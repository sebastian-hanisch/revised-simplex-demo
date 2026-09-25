"""Zwei Simplex-Verfahren mit demselben Pivotpfad: das dichte Tableau (Referenz aus den Stücken 1-3) und der Revised Simplex.

Revised Simplex: die Ausgangsmatrix M = [A | Schlupf | künstliche] bleibt unverändert; gehalten wird nur eine Darstellung der Basis-Inverse B^-1 (`ExplicitInverse`: dichtes m x m; `ProductForm`: Eta-Datei mit
Neuinversion alle K Pivots). Je Iteration: BTRAN (y = c_B B^-1), Preisgebung (r_j = y a_j - c_j nur über M), FTRAN (d = B^-1 a_q), Quotiententest, Update. Aufwand in Gleitkomma-Operationen, gezählt auf den
tatsächlichen Nichtnullen (Multiply-Add = 2), je Komponente."""

import math
import random
from dataclasses import dataclass, field

import numpy as np

import rev_scenario as S

LE, GE, EQ = S.LE, S.GE, S.EQ
TOL = 1e-9
TIE = 1e-9                                  # relative Toleranz für Gleichstände bei der Spaltenwahl (kleinster Index gewinnt)
MAX_PIVOTS = 20_000
RULES_TABLEAU = ("dantzig", "greatest", "steepest", "bland", "random")
RULES = ("dantzig", "steepest", "bland", "random")
BASES = ("explicit", "pfi")
OP_KEYS = ("btran", "price", "ftran", "ratio", "update", "refactor", "weights")


class SingularBasis(Exception):
    pass


@dataclass
class Pivot:
    k: int                       # laufende Nummer (ab 1)
    phase: int
    enter: int                   # eintretende Spalte
    leave_row: int               # Position (Zeile der Basis) der austretenden Variable
    leave_var: int               # Spalte der austretenden Basisvariable
    ratio: float                 # Schrittweite (kleinster Quotient)
    ties: int                    # Zahl der Zeilen mit demselben kleinsten Quotienten
    obj: float                   # Zielwert der aktuellen Phase nach dem Pivot
    degenerate: bool             # Schrittweite null
    basis: tuple                 # Basisvariablen nach dem Pivot (in Positionsreihenfolge)
    x: tuple                     # Entscheidungsvariablen der Basislösung nach dem Pivot
    feasible: bool               # alle künstlichen Variablen null
    zero_basics: int
    ops: int = 0                 # Operationen dieser Iteration (Revised: alle Komponenten; Tableau: Pivot nur auf Nichtnullen gezählt)
    eta: int = 0                 # gespeicherte Nichtnullen der Eta-Datei nach dem Pivot (nur Produktform)
    drift: float = 0.0           # max |B B^-1 - I| nach dem Pivot (Diagnose, nicht mitgezählt)
    nnz: int = 0                 # gespeicherte Nichtnullen nach dem Pivot (Tableau: alle Nichtnullen des Tableaus; Revised: Nichtnullen von M plus Inverse bzw. Eta-Datei)


@dataclass
class Result:
    status: str                  # "optimal" | "infeasible" | "unbounded" | "cycled" | "limit"
    x: tuple = ()
    obj: float = float("nan")
    duals: tuple = ()
    pivots: list = field(default_factory=list)
    phase1_pivots: int = 0
    redundant_rows: int = 0
    m: int = 0
    n_cols: int = 0
    rule: str = "dantzig"
    method: str = "tableau"      # "tableau" | "explicit" | "pfi"
    refactor_every: int = 0
    partial: float = 1.0
    flops: int = 0               # Tableau: Operationen der Pivots (dichtes Modell der Stücke 1-3)
    flops_nz: int = 0            # Tableau: Operationen der Pivots, nur auf Nichtnullen gezählt (Pivotzeile mal Zeilen mit Faktor != 0)
    price_flops: int = 0         # Tableau: Preisgebung (dichtes Modell wie in Stück 2)
    price_nz: int = 0            # Tableau: Preisgebung, nur auf Nichtnullen der Spalten gezählt
    ops: dict = field(default_factory=lambda: {k: 0 for k in OP_KEYS})    # Revised: Operationen je Komponente
    refactors: int = 0
    stored: int = 0              # gespeicherte Zahlen (Tableau: alle Einträge; Revised: Nichtnullen von M + Inverse bzw. Eta-Datei, Spitze)
    cycled: bool = False
    cycle_len: int = 0
    repeats: int = 0
    stall_runs: list = field(default_factory=list)
    col_names: tuple = ()
    y: object = None             # Revised: Duale y = c_B B^-1 der letzten Phase (Standardform-Orientierung)

    @property
    def total_pivots(self):
        return len(self.pivots)

    @property
    def degenerate_pivots(self):
        return sum(p.degenerate for p in self.pivots)

    @property
    def total_ops(self):
        return self.flops + self.price_flops if self.method == "tableau" else sum(self.ops.values())

    @property
    def total_ops_nz(self):
        """Tableau nur auf Nichtnullen (Pivots und Preisgebung); beim Revised gleich `total_ops`."""
        return self.flops_nz + self.price_nz if self.method == "tableau" else self.total_ops

    @property
    def ops_per_pivot(self):
        return self.total_ops / len(self.pivots) if self.pivots else 0.0


# --- Standardform (aus den Vorgängerstücken) --------------------------------------------------------------------------------------------------------


def standard_form(inst):
    """Gleichungsform mit Schlupf-, Überschuss- und künstlichen Variablen; Zeilen mit negativem b werden mit -1 multipliziert (Sinn dreht sich). Gibt (T, basis, info) mit T = [A' | rhs] (m Zeilen)."""
    A, b, c = inst.arrays()
    m, n = A.shape
    rows, rhs, senses, sign = [], [], [], []
    for i in range(m):
        a, r, s = A[i].copy(), float(b[i]), inst.senses[i]
        sg = 1
        if r < 0:
            a, r, sg = -a, -r, -1
            s = {LE: GE, GE: LE, EQ: EQ}[s]
        rows.append(a), rhs.append(r), senses.append(s), sign.append(sg)
    slack_col, art_col = {}, {}
    ncols = n
    for i, s in enumerate(senses):
        if s in (LE, GE):
            slack_col[i] = ncols
            ncols += 1
    for i, s in enumerate(senses):
        if s in (GE, EQ):
            art_col[i] = ncols
            ncols += 1
    T = np.zeros((m, ncols + 1))
    basis = []
    for i in range(m):
        T[i, :n] = rows[i]
        T[i, -1] = rhs[i]
        if i in slack_col:
            T[i, slack_col[i]] = 1.0 if senses[i] == LE else -1.0
        if i in art_col:
            T[i, art_col[i]] = 1.0
        basis.append(art_col[i] if i in art_col else slack_col[i])
    names = list(inst.names) + [None] * (ncols - n)
    for i, j in slack_col.items():
        names[j] = f"s{i + 1}" if senses[i] == LE else f"e{i + 1}"
    for i, j in art_col.items():
        names[j] = f"a{i + 1}"
    info = {"n": n, "m": m, "ncols": ncols, "slack_col": slack_col, "art_col": art_col, "sign": sign, "senses": senses, "names": names, "c": c}
    return T, basis, info


def _col_label(names, n, j):
    return names[j] if j >= n else f"x{j + 1}"


def _pick_min(cand, val):
    """Kleinster Wert; Gleichstände (relative Toleranz TIE) gewinnt der kleinste Spaltenindex. `cand` ist aufsteigend sortiert."""
    best = min(val[j] for j in cand)
    thr = best + TIE * max(1.0, abs(best))
    return next(j for j in cand if val[j] <= thr)


def _duals(info, y):
    """Schattenpreise in der ursprünglichen Orientierung: y_i mal Vorzeichen der Zeile (Umformung von b < 0)."""
    return tuple(float(y[i] * info["sign"][i]) for i in range(info["m"]))


# --- Dichtes Tableau (Referenz) ---------------------------------------------------------------------------------------------------------------------


def _pivot(T, row, col):
    T[row] /= T[row, col]
    factors = T[:, col].copy()
    factors[row] = 0.0
    idx = np.nonzero(factors)[0]
    if len(idx):
        T[idx] -= np.outer(factors[idx], T[row])


def _pivot_flops(m, ncols):
    return (ncols + 1) + 2 * m * (ncols + 1)


def _pivot_flops_nz(T, row, col):
    """Tableau-Pivot nur auf Nichtnullen: Division der Pivotzeile (Nichtnullen) plus 2 je Nichtnull der Pivotzeile für jede andere Zeile (auch die Zielzeile) mit Faktor != 0."""
    nz_row = int(np.count_nonzero(T[row]))
    nz_factors = int(np.count_nonzero(T[:, col])) - (1 if T[row, col] != 0 else 0)
    return nz_row + 2 * nz_row * nz_factors


def tableau_simplex(inst, rule="dantzig", seed="0", max_pivots=MAX_PIVOTS, trace=None):
    """Dichtes Tableau wie in den Stücken 1-3 (Quotiententest: kleinster Index), mit Gleichständen der Spaltenwahl nach dem kleinsten Index (Toleranz). Aufwand: Pivot (Spalten + 1) + 2 m (Spalten + 1),
    dazu die Preisgebung der Regel wie in Stück 2. Mit `trace` (Liste) werden je Iteration Basis und Nullzeile abgelegt (für Tests)."""
    if rule not in RULES_TABLEAU:
        raise ValueError(rule)
    T, basis, info = standard_form(inst)
    m, n, ncols = info["m"], info["n"], info["ncols"]
    art = set(info["art_col"].values())
    res = Result(status="optimal", col_names=tuple(_col_label(info["names"], n, j) for j in range(ncols)), m=m, n_cols=ncols, rule=rule, method="tableau")
    full = np.zeros((m + 1, ncols + 1))
    full[:m] = T
    T = full
    rng = random.Random(f"rev-{seed}-{rule}")
    res.stored = (m + 1) * (ncols + 1)

    def x_of_basis():
        xf = np.zeros(ncols)
        for i, j in enumerate(basis):
            xf[j] = T[i, -1]
        return xf

    def choose_entering(cand, r):
        if rule == "bland":
            return cand[0]
        if rule == "dantzig":
            return _pick_min(cand, r)
        if rule == "random":
            return cand[rng.randrange(len(cand))]
        if rule == "steepest":
            res.price_flops += (2 * m + 2) * len(cand)
            res.price_nz += sum(2 * int(np.count_nonzero(T[:m, j])) + 2 for j in cand)
            val = {j: r[j] / (1.0 + float(np.dot(T[:m, j], T[:m, j]))) ** 0.5 for j in cand}
            return _pick_min(cand, val)
        best, best_gain = None, -1.0
        res.price_flops += 2 * m * len(cand)
        res.price_nz += sum(2 * int(np.count_nonzero(T[:m, j])) for j in cand)
        for j in cand:
            col = T[:m, j]
            pos = [i for i in range(m) if col[i] > TOL]
            gain = float("inf") if not pos else -r[j] * min(T[i, -1] / col[i] for i in pos)
            if gain > best_gain + 1e-12:
                best, best_gain = j, gain
        return best

    def run(cvec, allowed, phase):
        cB = cvec[basis]
        T[m, :] = cB @ T[:m, :]
        T[m, :-1] -= cvec
        seen = {frozenset(basis): len(res.pivots)}
        zero_run = 0
        deterministic = rule != "random"
        while True:
            r = T[m, :-1]
            if trace is not None:
                trace.append({"basis": tuple(basis), "red": r.copy()})
            cand = [j for j in allowed if r[j] < -TOL]
            if not cand:
                if zero_run:
                    res.stall_runs.append(zero_run)
                return "optimal"
            enter = choose_entering(cand, r)
            col = T[:m, enter]
            pos = [i for i in range(m) if col[i] > TOL]
            if not pos:
                if zero_run:
                    res.stall_runs.append(zero_run)
                return "unbounded"
            ratios = {i: T[i, -1] / col[i] for i in pos}
            rmin = min(ratios.values())
            tied = [i for i in pos if ratios[i] <= rmin + TOL * max(1.0, abs(rmin))]
            leave = min(tied, key=lambda i: basis[i])
            leave_var = basis[leave]
            step = max(rmin, 0.0)
            nzc = _pivot_flops_nz(T, leave, enter)
            res.flops_nz += nzc
            _pivot(T, leave, enter)
            res.flops += _pivot_flops(m, ncols)
            basis[leave] = enter
            xf = x_of_basis()
            degenerate = step <= TOL
            if degenerate:
                zero_run += 1
            elif zero_run:
                res.stall_runs.append(zero_run)
                zero_run = 0
            art_sum = float(sum(xf[j] for j in art)) if art else 0.0
            piv = Pivot(len(res.pivots) + 1, phase, enter, leave, leave_var, float(step), len(tied), float(T[m, -1]), degenerate, tuple(basis), tuple(float(v) for v in xf[:n]), art_sum <= 1e-7,
                        sum(1 for j in basis if xf[j] <= TOL), ops=nzc, nnz=int(np.count_nonzero(T)))
            res.pivots.append(piv)
            key = frozenset(basis)
            if key in seen:
                if deterministic:
                    res.cycled, res.cycle_len = True, piv.k - seen[key]
                    if zero_run:
                        res.stall_runs.append(zero_run)
                    return "cycled"
                res.repeats += 1
            else:
                seen[key] = piv.k
            if len(res.pivots) > max_pivots:
                return "limit"

    if art:
        c1 = np.zeros(ncols)
        for j in art:
            c1[j] = -1.0
        status1 = run(c1, list(range(ncols)), 1)
        res.phase1_pivots = len(res.pivots)
        if status1 in ("cycled", "limit"):
            res.status = status1
            return res
        if T[m, -1] < -1e-7:
            res.status = "infeasible"
            return res
        for a in sorted(art):
            if a not in basis:
                continue
            i = basis.index(a)
            cand = [j for j in range(ncols) if j not in art and abs(T[i, j]) > TOL]
            if cand:
                best = max(abs(T[i, j]) for j in cand)
                enter = next(j for j in cand if abs(T[i, j]) >= best - TIE * max(1.0, best))
                nzc = _pivot_flops_nz(T, i, enter)
                res.flops_nz += nzc
                _pivot(T, i, enter)
                res.flops += _pivot_flops(m, ncols)
                basis[i] = enter
                xf = x_of_basis()
                res.pivots.append(Pivot(len(res.pivots) + 1, 1, enter, i, a, 0.0, 1, float(T[m, -1]), True, tuple(basis), tuple(float(v) for v in xf[:n]), True,
                                        sum(1 for j in basis if xf[j] <= TOL), ops=nzc, nnz=int(np.count_nonzero(T))))
            else:
                res.redundant_rows += 1
        res.phase1_pivots = len(res.pivots)
    c2 = np.zeros(ncols)
    c2[:n] = info["c"]
    status = run(c2, [j for j in range(ncols) if j not in art], 2)
    if status != "optimal":
        res.status = status
        return res
    xf = x_of_basis()
    res.x = tuple(float(v) for v in xf[:n])
    res.obj = float(np.dot(info["c"], xf[:n]))
    y = [float(T[m, info["art_col"][i]]) if i in info["art_col"] else (float(T[m, info["slack_col"][i]]) if info["senses"][i] == LE else -float(T[m, info["slack_col"][i]])) for i in range(m)]
    res.duals = _duals(info, y)
    return res


# --- Darstellungen der Basis-Inverse ----------------------------------------------------------------------------------------------------------------


CLEAN = 1e-12                               # Beträge darunter gelten als Rundungsrauschen (null): macht die gezählten Nichtnullen plattformunabhängig


def _clean(v):
    """Setzt Rundungsrauschen (|v| <= 1e-12) auf null, damit die Zählung der Nichtnullen nicht von der BLAS-Bibliothek der Plattform abhängt."""
    v = np.asarray(v, dtype=float).copy()
    v[np.abs(v) <= CLEAN] = 0.0
    return v


class ExplicitInverse:
    """Dichtes B^-1 (m x m), Rang-1-Update auf den Zeilen mit d_i != 0. Kosten: FTRAN/BTRAN 2 m je Nichtnull des Vektors, Update m + 2 m je Nichtnull von d (ohne die Pivotzeile)."""
    kind = "explicit"

    def __init__(self, m, ops):
        self.m, self.ops = m, ops
        self.Binv = np.eye(m)
        self.identity = True

    def ftran(self, v, key="ftran"):
        self.ops[key] += 2 * self.m * int(np.count_nonzero(v))
        return _clean(self.Binv @ v)

    def btran(self, u, key="btran"):
        self.ops[key] += 2 * self.m * int(np.count_nonzero(u))
        return _clean(u @ self.Binv)

    def update(self, p, d):
        nz = int(np.count_nonzero(d))
        self.ops["update"] += self.m + 2 * self.m * (nz - (1 if d[p] != 0 else 0))
        self.Binv[p] /= d[p]
        idx = np.nonzero(d)[0]
        idx = idx[idx != p]
        if len(idx):
            self.Binv[idx] -= np.outer(d[idx], self.Binv[p])
        self.identity = False

    def stored(self):
        return self.m * self.m

    def full_inverse(self):
        return self.Binv.copy()


class ProductForm:
    """Produktform der Inverse (Dantzig und Orchard-Hays): B^-1 = E_k ... E_1 mit Eta-Matrizen (Einheitsmatrix mit ersetzter Spalte p). Jedes Update hängt eine Eta an; FTRAN und BTRAN laufen die Datei ab und
    zählen nur Nichtnullen. Neuinversion: Gauß-Jordan-Etas für die aktuelle Basis (Einheitsspalten zuerst, dann die übrigen nach Spaltenbelegung, Pivotzeile größter Betrag)."""
    kind = "pfi"

    def __init__(self, m, ops):
        self.m, self.ops = m, ops
        self.etas = []                     # (Pivotposition, Eta-Vektor, Nichtnullen)
        self.identity = True

    def ftran(self, v, key="ftran"):
        x = v.astype(float).copy()
        for p, eta, nnz in self.etas:
            xp = x[p]
            if abs(xp) > CLEAN:
                x += xp * eta
                x[p] = xp * eta[p]
                self.ops[key] += 2 * (nnz - 1) + 1
        return _clean(x)

    def btran(self, u, key="btran"):
        x = u.astype(float).copy()
        for p, eta, _nnz in reversed(self.etas):
            pairs = int(np.count_nonzero((np.abs(x) > CLEAN) & (eta != 0.0)))
            self.ops[key] += 2 * pairs
            x[p] = float(x @ eta)
        return _clean(x)

    def _push(self, p, d, key):
        eta = _clean(-d / d[p])
        eta[p] = 1.0 / d[p]
        nnz = int(np.count_nonzero(eta))
        self.ops[key] += nnz
        self.etas.append((p, eta, nnz))
        self.identity = False

    def update(self, p, d):
        self._push(p, d, "update")

    def refactor(self, M, basis_vars):
        """Baut die Eta-Datei für die Spalten `basis_vars` neu auf; gibt die neue Positionsreihenfolge der Variablen zurück (Positionen dürfen permutieren)."""
        self.etas = []
        self.identity = True
        m = self.m
        assigned, rest = {}, []
        for var in basis_vars:
            col = M[:, var]
            nz = np.flatnonzero(col)
            if len(nz) == 1 and col[nz[0]] == 1.0 and int(nz[0]) not in assigned:
                assigned[int(nz[0])] = var
            else:
                rest.append(var)
        rest.sort(key=lambda v: (int(np.count_nonzero(M[:, v])), v))
        free = [p for p in range(m) if p not in assigned]
        for var in rest:
            d = self.ftran(M[:, var], key="refactor")
            p = max(free, key=lambda q: (abs(d[q]), -q))
            if abs(d[p]) < 1e-12:
                raise SingularBasis(var)
            self._push(p, d, "refactor")
            free.remove(p)
            assigned[p] = var
        return [assigned[p] for p in range(m)]

    def stored(self):
        return sum(nnz for _p, _e, nnz in self.etas)

    def full_inverse(self):
        X = np.eye(self.m)
        for p, eta, _nnz in self.etas:
            row = X[p].copy()
            X += np.outer(eta, row)
            X[p] = row * eta[p]
        return X


# --- Revised Simplex --------------------------------------------------------------------------------------------------------------------------------


def revised_simplex(inst, rule="dantzig", basis_kind="explicit", refactor_every=0, partial=1.0, seed="0", max_pivots=MAX_PIVOTS, track_drift=False, trace=None):
    """Revised Simplex mit Regel `rule` (dantzig, steepest nach Forrest-Goldfarb, bland, random), Basisdarstellung `basis_kind` (explicit | pfi), Neuinversion alle `refactor_every` Updates (nur pfi; 0 = nie)
    und partieller Preisgebung `partial` (Anteil der Nichtbasisspalten je Block; nur Dantzig). Pivotpfad wie `tableau_simplex` (bei gleicher Regel); Aufwand je Komponente in `Result.ops`. Mit `trace` (Liste) werden je Iteration Basis, reduzierte Kosten und Gewichte abgelegt (für Tests)."""
    if rule not in RULES or basis_kind not in BASES:
        raise ValueError((rule, basis_kind))
    if not 0.0 < partial <= 1.0 or (partial < 1.0 and rule != "dantzig"):
        raise ValueError(("partial", partial, rule))
    T, basis, info = standard_form(inst)
    m, n, ncols = info["m"], info["n"], info["ncols"]
    M, bvec = T[:, :ncols].copy(), T[:, -1].copy()
    colnnz = np.count_nonzero(M, axis=0)
    art = set(info["art_col"].values())
    res = Result(status="optimal", col_names=tuple(_col_label(info["names"], n, j) for j in range(ncols)), m=m, n_cols=ncols, rule=rule, method=basis_kind, refactor_every=refactor_every, partial=partial)
    ops = res.ops
    B = ExplicitInverse(m, ops) if basis_kind == "explicit" else ProductForm(m, ops)
    a_nnz = int(colnnz.sum())
    res.stored = a_nnz + B.stored()
    xB = bvec.copy()
    inbasis = np.zeros(ncols, dtype=bool)
    inbasis[basis] = True
    rng = random.Random(f"rev-{seed}-{rule}")
    gamma = np.ones(ncols)                 # Steepest-Edge-Gewichte 1 + |B^-1 a_j|^2 der Nichtbasisspalten
    state = {"since": 0, "block": 0}

    def x_of_basis():
        xf = np.zeros(ncols)
        xf[basis] = xB
        return xf

    def drift_now():
        Bm = M[:, basis]
        return float(np.abs(B.full_inverse() @ Bm - np.eye(m)).max())

    def init_weights(cols):
        for j in cols:
            if inbasis[j]:
                continue
            if B.identity:
                ops["weights"] += 2 * int(colnnz[j])
                gamma[j] = 1.0 + float(M[:, j] @ M[:, j])
            else:
                dj = B.ftran(M[:, j], key="weights")
                ops["weights"] += 2 * int(np.count_nonzero(dj))
                gamma[j] = 1.0 + float(dj @ dj)

    def price(cols, y, cvec):
        """Reduzierte Kosten (Vorzeichen wie im Tableau: y a_j - c_j) der Spalten `cols`."""
        ops["price"] += 2 * int(colnnz[cols].sum()) + len(cols)
        return y @ M[:, cols] - cvec[cols]

    def do_pivot(pos, enter, d):
        """Basiswechsel: Quotient, Lösungsvektor, Inverse. Gibt die Schrittweite zurück."""
        theta = xB[pos] / d[pos]
        nz = int(np.count_nonzero(d))
        ops["update"] += 2 * nz
        xB[:] = xB - theta * d
        xB[pos] = theta
        B.update(pos, d)
        inbasis[basis[pos]] = False
        inbasis[enter] = True
        basis[pos] = enter
        return theta

    def maybe_refactor():
        nonlocal xB
        if basis_kind != "pfi" or refactor_every <= 0:
            return
        state["since"] += 1
        if state["since"] < refactor_every:
            return
        state["since"] = 0
        order = B.refactor(M, list(basis))
        basis[:] = order
        xB = B.ftran(bvec, key="refactor")
        res.refactors += 1

    def run(cvec, allowed, phase):
        seen = {frozenset(basis): len(res.pivots)}
        zero_run = 0
        deterministic = rule != "random"
        if rule == "steepest":
            init_weights(allowed)
        while True:
            it_start = sum(ops.values())
            cB = cvec[basis]
            y = B.btran(cB) if np.count_nonzero(cB) else np.zeros(m)
            nonbasic = [j for j in allowed if not inbasis[j]]
            cand, red = [], {}
            if partial < 1.0 and nonbasic:
                size = max(1, math.ceil(partial * len(nonbasic)))
                nblocks = math.ceil(len(nonbasic) / size)
                for step_i in range(nblocks):
                    bidx = (state["block"] + step_i) % nblocks
                    cols = nonbasic[bidx * size:(bidx + 1) * size]
                    vals = price(cols, y, cvec)
                    cand = [j for j, v in zip(cols, vals) if v < -TOL]
                    red = {j: float(v) for j, v in zip(cols, vals)}
                    if cand:
                        state["block"] = (bidx + 1) % nblocks
                        break
            elif nonbasic:
                vals = price(nonbasic, y, cvec)
                red = {j: float(v) for j, v in zip(nonbasic, vals)}
                cand = [j for j in nonbasic if red[j] < -TOL]
            if trace is not None:
                trace.append({"basis": tuple(basis), "red": dict(red), "gamma": gamma.copy(), "y": y.copy()})
            if not cand:
                if zero_run:
                    res.stall_runs.append(zero_run)
                res.y = y
                return "optimal"
            if rule == "bland":
                enter = cand[0]
            elif rule == "dantzig":
                enter = _pick_min(cand, red)
            elif rule == "random":
                enter = cand[rng.randrange(len(cand))]
            else:
                enter = _pick_min(cand, {j: red[j] / gamma[j] ** 0.5 for j in cand})
            d = B.ftran(M[:, enter])
            pos = [i for i in range(m) if d[i] > TOL]
            if not pos:
                if zero_run:
                    res.stall_runs.append(zero_run)
                return "unbounded"
            ops["ratio"] += len(pos)
            ratios = {i: xB[i] / d[i] for i in pos}
            rmin = min(ratios.values())
            tied = [i for i in pos if ratios[i] <= rmin + TOL * max(1.0, abs(rmin))]
            leave = min(tied, key=lambda i: basis[i])
            leave_var = basis[leave]
            step = max(rmin, 0.0)
            if rule == "steepest":
                rho = np.zeros(m)
                rho[leave] = 1.0
                rho = B.btran(rho, key="weights")
                others = [j for j in nonbasic if j != enter]
                alpha = rho @ M[:, others] if others else np.zeros(0)
                ops["weights"] += 2 * int(colnnz[others].sum()) if others else 0
                gq = 1.0 + float(d @ d)
                ops["weights"] += 2 * int(np.count_nonzero(d))
                v = B.btran(d, key="weights")
                touched = [(j, a) for j, a in zip(others, alpha) if abs(a) > 1e-12]
                if touched:
                    tcols = [j for j, _a in touched]
                    ops["weights"] += 2 * int(colnnz[tcols].sum()) + 5 * len(touched)
                    av = v @ M[:, tcols]
                    for (j, a), aj in zip(touched, av):
                        t = a / d[leave]
                        gamma[j] = max(gamma[j] - 2.0 * t * aj + t * t * gq, 1.0 + t * t)
                new_leave_gamma = gq / (d[leave] * d[leave])
            do_pivot(leave, enter, d)
            if rule == "steepest":
                gamma[leave_var] = new_leave_gamma
            maybe_refactor()
            xf = x_of_basis()
            degenerate = step <= TOL
            if degenerate:
                zero_run += 1
            elif zero_run:
                res.stall_runs.append(zero_run)
                zero_run = 0
            art_sum = float(sum(xf[j] for j in art)) if art else 0.0
            obj = float(cvec[basis] @ xB)
            ops["update"] += 2 * int(np.count_nonzero(cvec[basis]))
            piv = Pivot(len(res.pivots) + 1, phase, enter, leave, leave_var, float(step), len(tied), obj, degenerate, tuple(basis), tuple(float(w) for w in xf[:n]), art_sum <= 1e-7,
                        sum(1 for j in basis if xf[j] <= TOL), sum(ops.values()) - it_start, B.stored() if basis_kind == "pfi" else 0, drift_now() if track_drift else 0.0, a_nnz + B.stored())
            res.pivots.append(piv)
            res.stored = max(res.stored, a_nnz + B.stored())
            key = frozenset(basis)
            if key in seen:
                if deterministic:
                    res.cycled, res.cycle_len = True, piv.k - seen[key]
                    if zero_run:
                        res.stall_runs.append(zero_run)
                    return "cycled"
                res.repeats += 1
            else:
                seen[key] = piv.k
            if len(res.pivots) > max_pivots:
                return "limit"

    if art:
        c1 = np.zeros(ncols)
        for j in art:
            c1[j] = -1.0
        status1 = run(c1, list(range(ncols)), 1)
        res.phase1_pivots = len(res.pivots)
        if status1 in ("cycled", "limit"):
            res.status = status1
            return res
        if float(c1[basis] @ xB) < -1e-7:
            res.status = "infeasible"
            return res
        for a in sorted(art):
            if not inbasis[a]:
                continue
            i = basis.index(a)
            e = np.zeros(m)
            e[i] = 1.0
            rho = B.btran(e, key="update")
            cols = [j for j in range(ncols) if j not in art]
            row = rho @ M[:, cols]
            ops["update"] += 2 * int(colnnz[cols].sum())
            cand = [j for j, v in zip(cols, row) if abs(v) > TOL]
            if cand:
                rowabs = {j: abs(v) for j, v in zip(cols, row)}
                best = max(rowabs[j] for j in cand)
                enter = next(j for j in cand if rowabs[j] >= best - TIE * max(1.0, best))
                d = B.ftran(M[:, enter], key="update")
                do_pivot(i, enter, d)
                maybe_refactor()
                xf = x_of_basis()
                res.pivots.append(Pivot(len(res.pivots) + 1, 1, enter, i, a, 0.0, 1, float(c1[basis] @ xB), True, tuple(basis), tuple(float(w) for w in xf[:n]), True,
                                        sum(1 for j in basis if xf[j] <= TOL)))
            else:
                res.redundant_rows += 1
        res.phase1_pivots = len(res.pivots)
    c2 = np.zeros(ncols)
    c2[:n] = info["c"]
    status = run(c2, [j for j in range(ncols) if j not in art], 2)
    if status != "optimal":
        res.status = status
        return res
    xf = x_of_basis()
    res.x = tuple(float(v) for v in xf[:n])
    res.obj = float(np.dot(info["c"], xf[:n]))
    res.duals = _duals(info, res.y)
    return res


# --- Prüfgrößen (für Tests und Auswertung) ---------------------------------------------------------------------------------------------------------


def primal_violation(inst, x):
    """Größte Verletzung der Bedingungen und der Nichtnegativität durch x (0 = zulässig)."""
    A, b, _c = inst.arrays()
    x = np.asarray(x, dtype=float)
    worst = max(0.0, float(-x.min())) if len(x) else 0.0
    for i in range(inst.m):
        act, s = float(A[i] @ x), inst.senses[i]
        worst = max(worst, max(0.0, act - b[i]) if s == LE else (max(0.0, b[i] - act) if s == GE else abs(act - b[i])))
    return worst


def dual_violation(inst, y):
    """Größte Verletzung der Dual-Zulässigkeit (A^T y >= c, y >= 0 bei <=, y <= 0 bei >=)."""
    A, _b, c = inst.arrays()
    y = np.asarray(y, dtype=float)
    worst = float(max(0.0, (c - A.T @ y).max())) if inst.n else 0.0
    for i in range(inst.m):
        s = inst.senses[i]
        worst = max(worst, max(0.0, -y[i]) if s == LE else (max(0.0, y[i]) if s == GE else 0.0))
    return worst
