"""Unabhängiges Orakel: ein exakter Simplex in rationaler Arithmetik (Fraction, eigene Schleife, keine gemeinsame Rechnung mit dem Demo-Code).

Geprüft werden (1) der Pivotpfad aller Verfahren gegen den exakten Pfad (Dantzig, Bland, Steepest Edge), (2) Optimalwert, Duale und Status, (3) die Zahl der Nichtnullen des Tableaus je Pivot und die
daraus gezählten Operationen des Nichtnull-Modells: Rundungsreste einer Auslöschung (1e-17) dürfen dort nicht als Eintrag zählen, sonst hängt die Zahl von der Plattform ab und bevorzugt den Revised."""

from fractions import Fraction as F

import numpy as np
import pytest

import rev_algorithm as A
import rev_scenario as S

LE, GE, EQ = S.LE, S.GE, S.EQ


def _fr(v, binary):
    return F(float(v)) if binary else F(str(float(v)))


def _exact(inst, rule, binary):
    """Zwei-Phasen-Tableau in Fraction mit denselben Regeln wie die Demo (kleinster Index bei Gleichstand), je Pivot: Pfad, Nichtnullen des (m+1) x (Spalten+1)-Tableaus, Operationen des Nichtnull-Modells."""
    m, n = inst.m, inst.n
    rows, rhs, ss, sign = [], [], [], []
    for i in range(m):
        a, r, s, sg = [_fr(v, binary) for v in inst.A[i]], _fr(inst.b[i], binary), inst.senses[i], 1
        if r < 0:
            a, r, sg, s = [-v for v in a], -r, -1, {LE: GE, GE: LE, EQ: EQ}[s]
        rows.append(a), rhs.append(r), ss.append(s), sign.append(sg)
    slack, art, nc = {}, {}, n
    for i, s in enumerate(ss):
        if s != EQ:
            slack[i], nc = nc, nc + 1
    for i, s in enumerate(ss):
        if s != LE:
            art[i], nc = nc, nc + 1
    T = [[F(0)] * (nc + 1) for _ in range(m)]
    basis = []
    for i in range(m):
        T[i][:n], T[i][nc] = rows[i], rhs[i]
        if i in slack:
            T[i][slack[i]] = F(1) if ss[i] == LE else F(-1)
        if i in art:
            T[i][art[i]] = F(1)
        basis.append(art[i] if i in art else slack[i])
    cost = [_fr(v, binary) for v in inst.c] + [F(0)] * (nc - n)
    artset = set(art.values())
    path, nnz, flops = [], [], [0]

    def zero_row(cvec):
        r = [sum((cvec[basis[i]] * T[i][j] for i in range(m)), F(0)) - cvec[j] for j in range(nc)]
        return r + [sum((cvec[basis[i]] * T[i][nc] for i in range(m)), F(0))]

    def pivot(row, col, cvec, phase):
        zr = zero_row(cvec)
        nz_row = sum(1 for v in T[row] if v != 0)
        flops[0] += nz_row + 2 * nz_row * (sum(1 for i in range(m) if T[i][col] != 0) + (1 if zr[col] != 0 else 0) - 1)
        leave = basis[row]
        p = T[row][col]
        T[row] = [v / p for v in T[row]]
        for i in range(m):
            if i != row and T[i][col] != 0:
                f = T[i][col]
                T[i] = [T[i][k] - f * T[row][k] for k in range(nc + 1)]
        basis[row] = col
        path.append((phase, col, leave))
        nnz.append(sum(1 for i in range(m) for v in T[i] if v != 0) + sum(1 for v in zero_row(cvec) if v != 0))

    def run(cvec, allowed, phase):
        while True:
            r = zero_row(cvec)[:nc]
            cand = [j for j in allowed if r[j] < 0]
            if not cand:
                return "optimal"
            if rule == "bland":
                enter = cand[0]
            elif rule == "dantzig":
                enter = next(j for j in cand if r[j] == min(r[k] for k in cand))
            else:
                enter = max(cand, key=lambda j: (r[j] * r[j] / (1 + sum(T[i][j] ** 2 for i in range(m))), -j))
            pos = [i for i in range(m) if T[i][enter] > 0]
            if not pos:
                return "unbounded"
            rmin = min(T[i][nc] / T[i][enter] for i in pos)
            lv = min((i for i in pos if T[i][nc] / T[i][enter] == rmin), key=lambda i: basis[i])
            pivot(lv, enter, cvec, phase)

    if artset:
        c1 = [F(-1) if j in artset else F(0) for j in range(nc)]
        run(c1, list(range(nc)), 1)
        if any(T[i][nc] > 0 for i in range(m) if basis[i] in artset):
            return {"status": "infeasible"}
        for a in sorted(artset):
            if a in basis:
                i = basis.index(a)
                cand = [j for j in range(nc) if j not in artset and T[i][j] != 0]
                if cand:
                    best = max(abs(T[i][j]) for j in cand)
                    pivot(i, next(j for j in cand if abs(T[i][j]) == best), c1, 1)
    status = run(cost, [j for j in range(nc) if j not in artset], 2)
    if status != "optimal":
        return {"status": status}
    r = zero_row(cost)[:nc]
    y = [(r[art[i]] if i in art else (r[slack[i]] if ss[i] == LE else -r[slack[i]])) * sign[i] for i in range(m)]
    xf = [F(0)] * nc
    for i, j in enumerate(basis):
        xf[j] = T[i][nc]
    return {"status": "optimal", "path": path, "nnz": nnz, "flops_nz": flops[0], "obj": sum(cost[j] * xf[j] for j in range(n)), "y": y}


def _custom(rows, b, c, senses):
    n = len(c)
    return S.Instance(tuple(tuple(float(v) for v in r) for r in rows), tuple(float(v) for v in b), tuple(float(v) for v in c), tuple(senses), tuple(f"x{j}" for j in range(n)),
                      tuple(f"r{i}" for i in range(len(b))), "custom")


def _instances():
    for seed in range(5):
        yield S.generate("mixed", 8, 8, 0.4, seed), True
        yield S.generate("mixed", 10, 20, 0.2, seed), True
        yield S.generate("random", 8, 12, 0.4, seed), False
        yield S.generate("transport", 3, 5, 0.5, seed), False
    yield S.textbook_instance(), False
    rng = np.random.default_rng(10)
    for _ in range(40):
        m, n = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        yield _custom(rng.integers(-3, 5, (m, n)), rng.integers(-4, 14, m), rng.integers(-3, 7, n), [str(s) for s in rng.choice([LE, GE, EQ], size=m, p=[.5, .3, .2])]), False
    yield _custom([[1, 1, 0], [2, 2, 0], [0, 1, 1]], [4, 8, 6], [1, 2, 1], [EQ, EQ, LE]), False     # redundante Gleichung


@pytest.mark.parametrize("rule", ("dantzig", "bland", "steepest"))
def test_all_methods_follow_the_exact_rational_path_and_count_true_nonzeros(rule):
    checked_pivots = checked_nnz = 0
    for inst, binary in _instances():
        ex = _exact(inst, rule, binary)
        runs = [A.tableau_simplex(inst, rule=rule), A.revised_simplex(inst, rule=rule, basis_kind="explicit"), A.revised_simplex(inst, rule=rule, basis_kind="pfi", refactor_every=3)]
        for r in runs:
            assert r.status == ex["status"], (inst.kind, inst.m, inst.n, rule)
            if ex["status"] != "optimal":
                continue
            assert [(p.phase, p.enter, p.leave_var) for p in r.pivots] == ex["path"], (inst.kind, inst.m, inst.n, rule, r.method)
            assert r.obj == pytest.approx(float(ex["obj"]), rel=1e-8, abs=1e-8)
            assert np.allclose(r.duals, [float(v) for v in ex["y"]], atol=1e-7)
            checked_pivots += len(ex["path"])
        t = runs[0]
        if ex["status"] == "optimal" and rule != "steepest":
            assert [p.nnz for p in t.pivots] == ex["nnz"], (inst.kind, inst.m, inst.n, rule)          # Nichtnullen des Tableaus je Pivot: exakt, ohne Rundungsreste
            assert t.flops_nz == ex["flops_nz"], (inst.kind, inst.m, inst.n, rule)
            checked_nnz += len(ex["nnz"])
    assert checked_pivots > 500 and (rule == "steepest" or checked_nnz > 150)
