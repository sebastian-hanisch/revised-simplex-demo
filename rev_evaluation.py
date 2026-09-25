"""Auswertung: Tableau gegen Revised (explizite Inverse, Produktform), Aufwand je Komponente, Neuinversion und Drift, partielle Preisgebung, Kreuzungspunkte."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import rev_algorithm as A
import rev_constants as C
import rev_scenario as S


@dataclass(frozen=True)
class Settings:
    kind: str = "textbook"
    m: int = C.DEFAULT_M                      # Ressourcen (Transport: Lager)
    n: int = C.DEFAULT_N                      # Dienste (Transport: Kunden)
    density: float = C.DEFAULT_DENSITY
    seed: int = C.DEFAULT_SEED
    rule: str = C.DEFAULT_RULE
    basis: str = C.DEFAULT_BASIS
    refactor: int = C.DEFAULT_REFACTOR
    partial: float = C.DEFAULT_PARTIAL

    @property
    def partial_eff(self):
        return self.partial if self.rule == "dantzig" else 1.0

    @property
    def refactor_eff(self):
        return self.refactor if self.basis == "pfi" else 0

    @property
    def k_eff(self):
        """Neuinversions-Abstand für die Vergleiche (0 = nie wird als Standard-K ersetzt, damit die Produktform mit Neuinversion existiert)."""
        return self.refactor if self.refactor > 0 else C.DEFAULT_REFACTOR


@lru_cache(maxsize=256)
def instance_of(settings):
    return S.generate(settings.kind, settings.m, settings.n, settings.density, settings.seed)


def run_revised(inst, rule, basis_kind, refactor, partial=1.0, seed="0", track_drift=False):
    return A.revised_simplex(inst, rule=rule, basis_kind=basis_kind, refactor_every=refactor if basis_kind == "pfi" else 0, partial=partial, seed=seed, track_drift=track_drift)


@dataclass
class Analysis:
    settings: Settings
    inst: object
    tab: object
    rev: object
    same_path: object             # True/False bei voller Preisgebung, None bei partieller

    @property
    def status(self):
        return self.rev.status

    @property
    def obj_gap(self):
        return abs(self.tab.obj - self.rev.obj) if self.tab.status == self.rev.status == "optimal" else float("nan")


def _path(res):
    return [(p.enter, p.leave_var, p.phase) for p in res.pivots]


@lru_cache(maxsize=128)
def analyse(settings):
    inst = instance_of(settings)
    seed = str(settings.seed)
    tab = A.tableau_simplex(inst, rule=settings.rule, seed=seed)
    rev = run_revised(inst, settings.rule, settings.basis, settings.refactor, settings.partial_eff, seed, track_drift=inst.m <= 60)
    same = None if settings.partial_eff < 1.0 else (tab.status == rev.status and _path(tab) == _path(rev))
    return Analysis(settings, inst, tab, rev, same)


@lru_cache(maxsize=128)
def methods(settings):
    """Alle Verfahren auf derselben Instanz und Regel: Tableau, Revised explizit, Produktform ohne und mit Neuinversion."""
    inst = instance_of(settings)
    seed, rule = str(settings.seed), settings.rule
    return {"tableau": A.tableau_simplex(inst, rule=rule, seed=seed), "explicit": run_revised(inst, rule, "explicit", 0, seed=seed), "pfi": run_revised(inst, rule, "pfi", 0, seed=seed),
            "pfik": run_revised(inst, rule, "pfi", settings.k_eff, seed=seed)}


def method_ops(name, res):
    """Gesamtoperationen je Verfahren (das Tableau einmal dicht, einmal nur auf Nichtnullen gezählt)."""
    return res.total_ops_nz if name == "tableau_nz" else res.total_ops


def method_rows(settings):
    """Tabellenzeilen: Verfahren, Pivots, Operationen, gespeicherte Nichtnullen (Spitze)."""
    mm = methods(settings)
    rows = []
    for name in ("tableau", "tableau_nz", "explicit", "pfi", "pfik"):
        res = mm["tableau" if name == "tableau_nz" else name]
        peak = res.stored if name == "tableau" else max((p.nnz for p in res.pivots), default=0)
        rows.append({"name": name, "label": C.METHOD_LABELS[name], "status": res.status, "pivots": res.total_pivots, "ops": method_ops(name, res), "ops_per_pivot": method_ops(name, res) / max(1, res.total_pivots),
                     "stored": peak, "ops_parts": dict(res.ops) if name in ("explicit", "pfi", "pfik") else None})
    return rows


def _median_row(rows):
    return {k: float(np.median([r[k] for r in rows])) for k in rows[0]}


def _seed_instances(settings, seeds=C.SWEEP_SEEDS):
    if settings.kind in S.FIXTURE_KINDS:
        return [(settings.seed, instance_of(settings))]
    return [(s, S.generate(settings.kind, settings.m, settings.n, settings.density, s)) for s in seeds]


@lru_cache(maxsize=128)
def run_config(settings):
    """Median über die fünf festen Instanzen (Seeds 100000-100004): Pivots, Operationen je Verfahren, Verhältnisse zum Tableau, gespeicherte Nichtnullen."""
    rows = []
    for s, inst in _seed_instances(settings):
        seed = str(s)
        t = A.tableau_simplex(inst, rule=settings.rule, seed=seed)
        e = run_revised(inst, settings.rule, "explicit", 0, seed=seed)
        q0 = run_revised(inst, settings.rule, "pfi", 0, seed=seed)
        qk = run_revised(inst, settings.rule, "pfi", settings.k_eff, seed=seed)
        if not (t.status == e.status == q0.status == qk.status == "optimal"):
            continue
        rows.append({"pivots": t.total_pivots, "dense": t.total_ops, "nz": t.total_ops_nz, "explicit": e.total_ops, "pfi": q0.total_ops, "pfik": qk.total_ops,
                     "stored_tab": max(p.nnz for p in t.pivots), "stored_dense": t.stored, "stored_explicit": max(p.nnz for p in e.pivots), "stored_pfik": max(p.nnz for p in qk.pivots)})
    if not rows:
        return None
    out = _median_row(rows)
    for k in ("explicit", "pfi", "pfik"):
        out[f"{k}_vs_dense"] = out[k] / out["dense"]
        out[f"{k}_vs_nz"] = out[k] / out["nz"]
    out["nz_vs_dense"] = out["nz"] / out["dense"]
    out["n_runs"] = len(rows)
    return out


CROSS_PARAMS = ("size", "width", "density")
TRANSPORT_SIZES = ((3, 5), (4, 8), (6, 10), (8, 12), (10, 20), (12, 30))
SIZE_VALUES = (10, 20, 40, 60, 80, 100)
WIDTH_FACTORS = (1, 2, 4, 8)
CROSS_SEEDS = tuple(range(100000, 100003))


@lru_cache(maxsize=32)
def crossover(settings, param):
    """Aufwand über einen Parameter (size: m = n; width: n = Faktor mal m; density) für Tableau (dicht, nur Nichtnullen), explizite Inverse und Produktform mit Neuinversion; Median über 3 Instanzen."""
    kind = settings.kind if settings.kind in ("random", "mixed", "transport") else "mixed"
    if kind == "transport":
        grid = [((k, l, 0.5), f"{k}x{l}") for k, l in TRANSPORT_SIZES]
    elif param == "size":
        grid = [((s, s, settings.density), str(s)) for s in SIZE_VALUES]
    elif param == "width":
        grid = [((settings.m, min(C.N_MAX, f * settings.m), settings.density), f"{f}m") for f in WIDTH_FACTORS]
    else:
        grid = [((settings.m, settings.n, d), f"{d:g}") for d in C.DENSITY_OPTIONS]
    rows = []
    for spec, label in grid:
        cells = []
        for s in CROSS_SEEDS:
            inst = S.generate(kind, spec[0], spec[1], spec[2], s)
            t = A.tableau_simplex(inst, rule=settings.rule, seed=str(s))
            e = run_revised(inst, settings.rule, "explicit", 0, seed=str(s))
            q = run_revised(inst, settings.rule, "pfi", settings.k_eff, seed=str(s))
            if t.status == e.status == q.status == "optimal":
                cells.append({"pivots": t.total_pivots, "dense": t.total_ops, "nz": t.total_ops_nz, "explicit": e.total_ops, "pfik": q.total_ops})
        if cells:
            row = _median_row(cells)
            row["label"] = label
            rows.append(row)
    return rows


@lru_cache(maxsize=32)
def refactor_curve(settings):
    """Produktform über die Neuinversions-Abstände K (0 = nie): Gesamtoperationen, davon Neuinversion, Spitze der gespeicherten Nichtnullen, Drift (Median über 5 Instanzen)."""
    rows = []
    for K in C.REFACTOR_OPTIONS:
        cells = []
        for s, inst in _seed_instances(settings):
            q = run_revised(inst, settings.rule, "pfi", K, seed=str(s), track_drift=inst.m <= 60)
            if q.status == "optimal":
                cells.append({"ops": q.total_ops, "refactor_ops": q.ops["refactor"], "stored": max(p.nnz for p in q.pivots), "drift": max(p.drift for p in q.pivots), "pivots": q.total_pivots,
                              "refactors": q.refactors})
        if cells:
            row = _median_row(cells)
            row["K"] = K
            rows.append(row)
    return rows


@lru_cache(maxsize=32)
def partial_curve(settings):
    """Dantzig mit partieller Preisgebung über den Anteil der je Block bepreisten Spalten (Produktform mit Neuinversion): Pivots, Preisgebung, Gesamt (Median über 5 Instanzen)."""
    rows = []
    for frac in C.PARTIAL_OPTIONS:
        cells = []
        for s, inst in _seed_instances(settings):
            q = run_revised(inst, "dantzig", "pfi", settings.k_eff, partial=frac, seed=str(s))
            if q.status == "optimal":
                cells.append({"pivots": q.total_pivots, "price": q.ops["price"], "total": q.total_ops})
        if cells:
            row = _median_row(cells)
            row["fraction"] = frac
            rows.append(row)
    return rows


@lru_cache(maxsize=32)
def steepest_cost(settings):
    """Dantzig gegen Steepest Edge in jedem Verfahren (Median über 5 Instanzen): Pivots, Operationen, Anteil der Gewichte."""
    out = {}
    for rule in ("dantzig", "steepest"):
        cells = []
        for s, inst in _seed_instances(settings):
            seed = str(s)
            t = A.tableau_simplex(inst, rule=rule, seed=seed)
            e = run_revised(inst, rule, "explicit", 0, seed=seed)
            q = run_revised(inst, rule, "pfi", settings.k_eff, seed=seed)
            if t.status == e.status == q.status == "optimal":
                cells.append({"pivots": t.total_pivots, "dense": t.total_ops, "nz": t.total_ops_nz, "explicit": e.total_ops, "pfik": q.total_ops, "weights_pfik": q.ops["weights"], "price_pfik": q.ops["price"]})
        out[rule] = _median_row(cells) if cells else None
    return out


def pivot_state(inst, res, k):
    """Was der Revised Simplex vor Pivot k + 1 hält (k = 0 ... Zahl der Pivots): Basis, B, B^-1, x_B, y, reduzierte Kosten, eintretende Spalte d und Quotienten des nächsten Pivots."""
    T, basis0, info = A.standard_form(inst)
    ncols, m = info["ncols"], info["m"]
    M, b = T[:, :ncols], T[:, -1]
    art = set(info["art_col"].values())
    basis = list(basis0) if k == 0 else list(res.pivots[k - 1].basis)
    nxt = res.pivots[k] if k < len(res.pivots) else None
    phase = nxt.phase if nxt is not None else 2
    cvec = np.zeros(ncols)
    if phase == 1:
        for j in art:
            cvec[j] = -1.0
    else:
        cvec[:info["n"]] = info["c"]
    Bm = M[:, basis]
    Binv = np.linalg.inv(Bm)
    xB = Binv @ b
    y = cvec[basis] @ Binv
    red = y @ M - cvec
    allowed = [j for j in range(ncols) if phase == 1 or j not in art]
    out = {"M": M, "b": b, "basis": basis, "B": Bm, "Binv": Binv, "xB": xB, "y": y, "red": red, "allowed": allowed, "phase": phase, "names": list(res.col_names), "next": nxt, "ncols": ncols, "m": m}
    if nxt is not None:
        d = Binv @ M[:, nxt.enter]
        out["d"] = d
        out["ratios"] = {i: float(xB[i] / d[i]) for i in range(m) if d[i] > A.TOL}
    return out


DRIFT_MAX_M = 60


@lru_cache(maxsize=32)
def drift_series(settings):
    """Abweichung |B B^-1 - I| nach jedem Pivot für explizite Inverse, Produktform ohne und mit Neuinversion (Diagnose, nicht mitgezählt); None ab m > 60 (die Diagnose kostet je Pivot eine volle Inverse)."""
    inst = instance_of(settings)
    if inst.m > DRIFT_MAX_M:
        return None
    seed = str(settings.seed)
    out = {}
    for name, kind, K in (("explicit", "explicit", 0), ("pfi", "pfi", 0), ("pfik", "pfi", settings.k_eff)):
        q = run_revised(inst, settings.rule, kind, K, seed=seed, track_drift=True)
        out[name] = [p.drift for p in q.pivots]
    return out
