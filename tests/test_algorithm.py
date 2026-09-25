"""Korrektheitskette: jede Kombination Basisdarstellung x Regel == HiGHS, gleicher Pivotpfad wie das Tableau, Basisdarstellungen gegen die dichte Inverse, reduzierte Kosten und Steepest-Edge-Gewichte gegen
unabhängige Neuberechnung, Buchführung (von Hand gezählt), partielle Preisgebung, Sonderfälle."""

import numpy as np
import pytest

import rev_algorithm as A
import rev_scenario as S
from tests.test_scenario import _highs, reference_status

CONFIGS = [("explicit", 0), ("pfi", 0), ("pfi", 3)]


def _custom(rows, b, c, senses):
    n = len(c)
    return S.Instance(tuple(tuple(float(v) for v in r) for r in rows), tuple(float(v) for v in b), tuple(float(v) for v in c), tuple(senses), tuple(f"x{j}" for j in range(n)), tuple(f"r{i}" for i in range(len(b))), "custom")


def _families():
    """Instanzen aller Arten in vielen Größen (dicht, dünn, breit, Transport, Mischung mit Phase 1)."""
    for seed in range(14):
        yield S.generate("random", 8, 12, 0.4, seed)
        yield S.generate("random", 6, 30, 0.1, seed)
        yield S.generate("mixed", 8, 8, 0.4, seed)
        yield S.generate("mixed", 10, 20, 0.2, seed)
        yield S.generate("transport", 3, 5, 0.5, seed)
        yield S.generate("transport", 2, 6, 0.5, seed)
        yield S.generate("random", 12, 12, 1.0, seed)
    yield S.textbook_instance()


def _solve(inst, **kw):
    return A.revised_simplex(inst, **kw)


@pytest.mark.parametrize("rule", A.RULES)
@pytest.mark.parametrize("basis_kind,K", CONFIGS)
def test_every_combination_reaches_the_highs_optimum_with_certificates(rule, basis_kind, K):
    count = 0
    for inst in _families():
        r, h = _solve(inst, rule=rule, basis_kind=basis_kind, refactor_every=K), _highs(inst)
        assert r.status == "optimal" and h.status == 0, (inst.kind, rule)
        assert r.obj == pytest.approx(-h.fun, rel=1e-7, abs=1e-7)
        assert A.primal_violation(inst, r.x) < 1e-7 and A.dual_violation(inst, r.duals) < 1e-6
        assert float(np.dot(r.duals, np.array(inst.b))) == pytest.approx(r.obj, rel=1e-7, abs=1e-7)
        count += 1
    assert count >= 90


def test_all_combinations_agree_on_infeasible_and_unbounded():
    for rule in A.RULES:
        for basis_kind, K in CONFIGS:
            assert _solve(S.infeasible_instance(), rule=rule, basis_kind=basis_kind, refactor_every=K).status == "infeasible"
            assert _solve(S.unbounded_instance(), rule=rule, basis_kind=basis_kind, refactor_every=K).status == "unbounded"


def test_random_lps_with_mixed_signs_agree_with_highs_for_every_combination():
    rng = np.random.default_rng(21)
    seen = set()
    for _ in range(80):
        m, n = int(rng.integers(1, 6)), int(rng.integers(1, 5))
        inst = _custom(np.round(rng.uniform(-3, 4, (m, n)), 1), np.round(rng.uniform(-5, 20, m), 1), np.round(rng.uniform(-3, 6, n), 1), [str(s) for s in rng.choice([S.LE, S.GE, S.EQ], size=m, p=[0.55, 0.3, 0.15])])
        ref = reference_status(inst)
        for rule in A.RULES:
            for basis_kind, K in CONFIGS:
                r = _solve(inst, rule=rule, basis_kind=basis_kind, refactor_every=K)
                assert r.status == ref, (rule, basis_kind, K)
                if ref == "optimal":
                    assert r.obj == pytest.approx(-_highs(inst).fun, rel=1e-6, abs=1e-6)
        seen.add(ref)
    assert seen == {"optimal", "infeasible", "unbounded"}


@pytest.mark.parametrize("rule", A.RULES)
@pytest.mark.parametrize("basis_kind,K", CONFIGS)
def test_revised_walks_exactly_the_tableau_path(rule, basis_kind, K):
    count, pivots = 0, 0
    for inst in _families():
        t = A.tableau_simplex(inst, rule=rule)
        r = _solve(inst, rule=rule, basis_kind=basis_kind, refactor_every=K)
        assert (r.status, r.phase1_pivots) == (t.status, t.phase1_pivots)
        assert [(p.enter, p.leave_var, p.phase) for p in r.pivots] == [(p.enter, p.leave_var, p.phase) for p in t.pivots], (inst.kind, inst.m, inst.n, rule)
        assert [p.obj for p in r.pivots] == pytest.approx([p.obj for p in t.pivots], rel=1e-7, abs=1e-7)
        assert r.obj == pytest.approx(t.obj, rel=1e-8, abs=1e-8) and r.redundant_rows == t.redundant_rows
        count += 1
        pivots += t.total_pivots
    assert count >= 90 and pivots > 500


def test_the_paths_include_phase_one_steepest_restart_and_refactorizations():
    mixed = [S.generate("mixed", 10, 20, 0.2, s) for s in range(10)]
    r = _solve(mixed[0], rule="steepest", basis_kind="pfi", refactor_every=3)
    assert any(_solve(i, rule="steepest").phase1_pivots > 0 for i in mixed) and r.refactors >= 1
    assert sum(_solve(i, rule="dantzig", basis_kind="pfi", refactor_every=2).refactors for i in mixed) >= 10


def test_drive_out_of_artificials_and_redundant_rows_match_the_tableau():
    inst = _custom([[1, 1, 0], [1, 1, 0], [0, 1, 1]], [4, 4, 6], [1, 2, 1], [S.EQ, S.EQ, S.LE])
    t = A.tableau_simplex(inst)
    assert t.redundant_rows == 1
    for basis_kind, K in CONFIGS:
        r = _solve(inst, basis_kind=basis_kind, refactor_every=K)
        assert r.redundant_rows == 1 and r.status == "optimal" and r.obj == pytest.approx(-_highs(inst).fun)
    drive = [i for i in (S.generate("mixed", 8, 8, 0.4, s) for s in range(40)) if A.tableau_simplex(i).phase1_pivots > 0 and any(p.phase == 1 and p.ratio == 0.0 and p.ties == 1 for p in A.tableau_simplex(i).pivots)]
    assert drive
    for inst in drive[:10]:
        assert [(p.enter, p.leave_var) for p in _solve(inst).pivots] == [(p.enter, p.leave_var) for p in A.tableau_simplex(inst).pivots]


# --- Basisdarstellungen ------------------------------------------------------------------------------------------------------------------------------


def _replay_updates(M, basis, kinds, rng, steps, refactor_every=0):
    """Wendet dieselbe Folge gültiger Basiswechsel auf alle Darstellungen an; prüft nach jedem Wechsel FTRAN, BTRAN und die volle Inverse gegen numpy."""
    m = M.shape[0]
    ops = [{k: 0 for k in A.OP_KEYS} for _ in kinds]
    reps = [A.ExplicitInverse(m, ops[0]) if k == "explicit" else A.ProductForm(m, ops[1]) for k in kinds]
    basis = list(basis)
    since = 0
    for _ in range(steps):
        Bm = M[:, basis]
        j = int(rng.integers(0, M.shape[1]))
        if j in basis:
            continue
        d = np.linalg.solve(Bm, M[:, j])
        p = int(np.argmax(np.abs(d)))
        if abs(d[p]) < 1e-6:
            continue
        for rep in reps:
            rep.update(p, d)
        basis[p] = j
        since += 1
        if refactor_every and since >= refactor_every:
            for rep in reps:
                if isinstance(rep, A.ProductForm):
                    order = rep.refactor(M, list(basis))
            basis = list(order)
            for rep in reps:
                if isinstance(rep, A.ExplicitInverse):
                    rep.Binv = np.linalg.inv(M[:, basis])
            since = 0
        Bm = M[:, basis]
        v, u = rng.normal(size=m), rng.normal(size=m)
        for rep in reps:
            assert np.allclose(rep.ftran(v), np.linalg.solve(Bm, v), atol=1e-8) and np.allclose(rep.btran(u), np.linalg.solve(Bm.T, u), atol=1e-8)
            assert np.abs(rep.full_inverse() @ Bm - np.eye(m)).max() < 1e-8
    return reps


def test_explicit_and_product_form_solve_like_numpy_after_every_update_and_refactor():
    rng = np.random.default_rng(3)
    for trial in range(12):
        m = 6
        M = np.hstack([np.eye(m), np.round(rng.normal(size=(m, 14)), 1) * (rng.random((m, 14)) < 0.6)])
        _replay_updates(M, list(range(m)), ["explicit", "pfi"], rng, 40)
        _replay_updates(M, list(range(m)), ["explicit", "pfi"], rng, 40, refactor_every=3)
        _replay_updates(M, list(range(m)), ["explicit", "pfi"], rng, 30, refactor_every=1)


def test_refactor_may_permute_positions_but_not_the_path():
    inst = S.generate("mixed", 10, 20, 0.2, 4)
    a = _solve(inst, basis_kind="pfi", refactor_every=1)
    b = _solve(inst, basis_kind="pfi", refactor_every=0)
    assert a.refactors >= 5 and [(p.enter, p.leave_var) for p in a.pivots] == [(p.enter, p.leave_var) for p in b.pivots]
    assert any(p.basis != q.basis for p, q in zip(a.pivots, b.pivots))                  # die Positionen sind wirklich permutiert


def test_singular_basis_is_reported():
    M = np.array([[1.0, 2.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    rep = A.ProductForm(3, {k: 0 for k in A.OP_KEYS})
    with pytest.raises(A.SingularBasis):
        rep.refactor(M, [0, 1, 2])


def test_drift_stays_tiny_and_matches_the_dense_inverse_at_every_pivot():
    for inst in (S.generate("mixed", 12, 20, 0.3, 2), S.generate("random", 15, 15, 0.5, 1), S.generate("transport", 4, 6, 0.5, 2)):
        for basis_kind, K in CONFIGS:
            r = _solve(inst, basis_kind=basis_kind, refactor_every=K, track_drift=True)
            assert r.total_pivots >= 5 and max(p.drift for p in r.pivots) < 1e-8


# --- reduzierte Kosten und Steepest-Edge-Gewichte ------------------------------------------------------------------------------------------------------


def test_reduced_costs_from_pricing_equal_the_tableau_zero_row_at_every_iteration():
    checked = 0
    for inst in list(_families())[:60]:
        tt, tr = [], []
        A.tableau_simplex(inst, trace=tt)
        _solve(inst, basis_kind="pfi", refactor_every=2, trace=tr)
        assert len(tt) == len(tr)
        for a, b in zip(tt, tr):
            assert set(a["basis"]) == set(b["basis"])
            for j, v in b["red"].items():
                assert v == pytest.approx(a["red"][j], abs=1e-7, rel=1e-7)
                checked += 1
    assert checked > 2000


def test_steepest_edge_weights_equal_the_recomputed_norms_at_every_iteration():
    checked = 0
    for inst in list(_families())[:70]:
        T, _basis, info = A.standard_form(inst)
        M = T[:, :-1]
        for basis_kind, K in (("explicit", 0), ("pfi", 3)):
            tr = []
            _solve(inst, rule="steepest", basis_kind=basis_kind, refactor_every=K, trace=tr)
            for it in tr:
                Bm = M[:, list(it["basis"])]
                for j in it["red"]:
                    exact = 1.0 + float(np.sum(np.linalg.solve(Bm, M[:, j]) ** 2))
                    assert it["gamma"][j] == pytest.approx(exact, rel=1e-7)
                    checked += 1
    assert checked > 3000


def test_duals_are_the_btran_result_of_the_last_iteration():
    inst = S.textbook_instance()
    tr = []
    r = _solve(inst, trace=tr)
    assert np.allclose(r.duals, [0.0, 1.5, 1.0]) and np.allclose(tr[-1]["y"], [0.0, 1.5, 1.0])


# --- Buchführung (von Hand gezählt) --------------------------------------------------------------------------------------------------------------------


def test_textbook_operation_counts_match_the_hand_count():
    inst = S.textbook_instance()
    t = A.tableau_simplex(inst)
    assert t.total_pivots == 2 and t.total_ops == 84 and t.stored == 4 * 6 and t.flops_nz == 15 + 20 and t.total_ops_nz == 35
    e = _solve(inst)
    assert e.total_pivots == 2 and e.ops == {"btran": 6 + 12, "price": 10 + 8 + 6, "ftran": 12 + 12, "ratio": 2 + 2, "update": 15 + 17, "refactor": 0, "weights": 0} and e.total_ops == 102
    p = _solve(inst, basis_kind="pfi")
    assert p.ops == {"btran": 2 + 6, "price": 24, "ftran": 0, "ratio": 4, "update": 8 + 10, "refactor": 0, "weights": 0} and p.total_ops == 54
    assert [q.ops for q in e.pivots] == [39, 45] and [q.ops for q in p.pivots] == [20, 22]
    assert e.stored == 7 + 9 and p.stored == 7 + 4                       # Nichtnullen von M (2 + 2 + 1 + 1 + 1 = 7) plus Inverse bzw. zwei Etas mit je 2 Nichtnullen


def test_components_sum_to_the_total_and_pivot_ops_never_exceed_it():
    for inst in list(_families())[::9]:
        for rule in A.RULES:
            r = _solve(inst, rule=rule, basis_kind="pfi", refactor_every=3)
            assert r.total_ops == sum(r.ops.values()) and sum(p.ops for p in r.pivots) <= r.total_ops
            assert (r.ops["weights"] > 0) == (rule == "steepest") and (r.ops["refactor"] > 0) == (r.refactors > 0)


def test_component_costs_of_the_representations_on_a_known_example():
    m = 3
    ops = {k: 0 for k in A.OP_KEYS}
    ex = A.ExplicitInverse(m, ops)
    ex.ftran(np.array([1.0, 0.0, 2.0]))
    ex.btran(np.array([0.0, 3.0, 0.0]))
    assert ops["ftran"] == 12 and ops["btran"] == 6
    ex.update(1, np.array([0.0, 2.0, 2.0]))
    assert ops["update"] == 3 + 2 * 3 * 1
    ops2 = {k: 0 for k in A.OP_KEYS}
    pf = A.ProductForm(m, ops2)
    pf.update(1, np.array([0.0, 2.0, 2.0]))
    assert ops2["update"] == 2 and pf.stored() == 2
    pf.ftran(np.array([0.0, 1.0, 0.0]))
    assert ops2["ftran"] == 2 * (2 - 1) + 1
    pf.ftran(np.array([1.0, 0.0, 0.0]))
    assert ops2["ftran"] == 3                                            # x[p] = 0: die Eta wird übersprungen


def test_tableau_model_reproduces_the_numbers_of_the_earlier_pieces():
    t = A.tableau_simplex(S.generate("random", 10, 10, 0.5, 35))
    assert t.flops == t.total_pivots * A._pivot_flops(10, 20) and t.price_flops == 0
    s = A.tableau_simplex(S.textbook_instance(), rule="steepest")
    assert s.total_ops == 108 and A.tableau_simplex(S.textbook_instance(), rule="greatest").total_ops == 102
    assert s.price_nz == 18 and s.total_ops_nz == 35 + 18


# --- partielle Preisgebung ------------------------------------------------------------------------------------------------------------------------------


def test_partial_one_equals_full_pricing_and_every_fraction_reaches_the_optimum():
    for inst in list(_families())[:40:3]:
        full = _solve(inst)
        assert [(p.enter, p.leave_var) for p in _solve(inst, partial=1.0).pivots] == [(p.enter, p.leave_var) for p in full.pivots]
        for frac in (0.5, 0.25, 0.125, 0.05):
            r = _solve(inst, partial=frac)
            assert r.status == "optimal" and r.obj == pytest.approx(full.obj, rel=1e-8, abs=1e-8)
    with pytest.raises(ValueError):
        _solve(S.textbook_instance(), rule="steepest", partial=0.5)
    with pytest.raises(ValueError):
        _solve(S.textbook_instance(), partial=0.0)


def test_partial_pricing_scans_every_block_before_declaring_optimality():
    # nur die letzte Spalte verbessert: der Lauf muss durch alle Blöcke laufen, bevor er die Spalte findet, und am Ende einen sauberen Umlauf machen
    A_ = [[1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6]]
    inst = _custom(A_, [10, 40], [0, 0, 0, 0, 0, 1], [S.LE, S.LE])
    r = _solve(inst, partial=1 / 6)
    assert r.status == "optimal" and r.obj == pytest.approx(-_highs(inst).fun) and r.obj == pytest.approx(6.666666666, rel=1e-6)
    full = _solve(inst)
    assert r.ops["price"] > 0 and r.total_pivots == full.total_pivots


def test_partial_pricing_lowers_the_pricing_cost_per_iteration_on_wide_instances():
    inst = S.generate("random", 10, 200, 0.05, 5)
    full, part = _solve(inst), _solve(inst, partial=0.125)
    assert full.status == part.status == "optimal" and part.ops["price"] / part.total_pivots < 0.6 * full.ops["price"] / full.total_pivots


# --- Sonderfälle -------------------------------------------------------------------------------------------------------------------------------------


def test_special_cases_m1_n1_limit_and_determinism():
    one = _custom([[2]], [6], [3], [S.LE])
    single = _custom([[1, 2, 3]], [12], [1, 2, 3], [S.LE])
    for basis_kind, K in CONFIGS:
        for rule in A.RULES:
            assert _solve(one, rule=rule, basis_kind=basis_kind, refactor_every=K).obj == pytest.approx(9.0)
            assert _solve(single, rule=rule, basis_kind=basis_kind, refactor_every=K).obj == pytest.approx(12.0)
    inst = next(i for i in _families() if A.tableau_simplex(i).total_pivots >= 6)
    assert _solve(inst, max_pivots=2).status == "limit" and A.tableau_simplex(inst, max_pivots=2).status == "limit"
    a, b = _solve(inst, basis_kind="pfi", refactor_every=3), _solve(inst, basis_kind="pfi", refactor_every=3)
    assert (a.total_pivots, a.total_ops, a.obj) == (b.total_pivots, b.total_ops, b.obj)
    x, y = _solve(inst, rule="random", seed="5"), _solve(inst, rule="random", seed="5")
    assert [p.enter for p in x.pivots] == [p.enter for p in y.pivots]
    with pytest.raises(ValueError):
        _solve(inst, rule="greatest")
    with pytest.raises(ValueError):
        _solve(inst, basis_kind="nope")


def test_zero_rhs_rows_and_degenerate_starts_still_agree_with_the_tableau():
    inst = _custom([[1, 1, 0], [0, 1, 1], [1, 0, 1]], [0, 0, 5], [1, 1, 1], [S.LE, S.LE, S.LE])
    t = A.tableau_simplex(inst, rule="bland")
    for basis_kind, K in CONFIGS:
        r = _solve(inst, rule="bland", basis_kind=basis_kind, refactor_every=K)
        assert r.status == t.status and [(p.enter, p.leave_var) for p in r.pivots] == [(p.enter, p.leave_var) for p in t.pivots]
