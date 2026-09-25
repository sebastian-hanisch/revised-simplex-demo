"""Instanzen: Bauart, Determinismus, Fixtures, Zulässigkeit der erzeugten Instanzen, Transport, Dünnheit."""

import numpy as np
import pytest
from scipy.optimize import linprog

import rev_scenario as S


def _highs(inst, zero_objective=False):
    A, b, c = inst.arrays()
    if zero_objective:
        c = np.zeros_like(c)
    ub_a, ub_b, eq_a, eq_b = [], [], [], []
    for i, s in enumerate(inst.senses):
        if s == S.LE:
            ub_a.append(A[i]), ub_b.append(b[i])
        elif s == S.GE:
            ub_a.append(-A[i]), ub_b.append(-b[i])
        else:
            eq_a.append(A[i]), eq_b.append(b[i])
    return linprog(-c, A_ub=np.array(ub_a) if ub_a else None, b_ub=ub_b or None, A_eq=np.array(eq_a) if eq_a else None, b_eq=eq_b or None, bounds=(0, None), method="highs")


def reference_status(inst):
    """Status laut HiGHS; bei zulässigen, unbeschränkten LPs meldet HiGHS gelegentlich "unzulässig" (Präsolve), darum wird über ein Zulässigkeitsproblem ohne Zielfunktion abgesichert."""
    h = _highs(inst)
    if h.status == 0:
        return "optimal"
    return "infeasible" if _highs(inst, zero_objective=True).status == 2 else "unbounded"


def test_textbook_is_the_classic_two_service_example():
    inst = S.textbook_instance()
    assert (inst.m, inst.n) == (3, 2) and inst.senses == (S.LE,) * 3 and inst.b == (4.0, 12.0, 18.0) and inst.c == (3.0, 5.0)
    assert -_highs(inst).fun == pytest.approx(36.0)


def test_fixtures_have_the_status_they_claim():
    assert _highs(S.infeasible_instance()).status == 2
    assert _highs(S.unbounded_instance()).status == 3


def test_generation_is_deterministic_and_seed_dependent_and_hashable():
    a = S.generate("random", 8, 6, 0.5, 3)
    assert a == S.generate("random", 8, 6, 0.5, 3) and a != S.generate("random", 8, 6, 0.5, 4) and hash(a) == hash(S.generate("random", 8, 6, 0.5, 3))


@pytest.mark.parametrize("kind", ["random", "mixed"])
def test_generated_instances_are_feasible_and_bounded(kind):
    for seed in range(40):
        for m, n in ((2, 2), (6, 4), (4, 9), (15, 15), (10, 80)):
            inst = S.generate(kind, m, n, 0.1, seed)
            assert (inst.m, inst.n) == (m, n) and len(inst.senses) == m and _highs(inst).status == 0, (kind, seed, m, n)


def test_every_row_and_column_is_used_and_row_zero_is_dense_even_at_low_density():
    for seed in range(20):
        inst = S.generate("random", 10, 40, 0.02, seed)
        A = np.array(inst.A)
        assert (A.sum(axis=1) > 0).all() and (A.sum(axis=0) > 0).all() and (A[0] > 0).all() and (A >= 0).all()


def test_density_controls_the_number_of_nonzeros():
    counts = [int(np.count_nonzero(np.array(S.generate("random", 20, 60, d, 3).A))) for d in (0.02, 0.1, 0.5, 1.0)]
    assert counts == sorted(counts) and counts[0] < 0.2 * 20 * 60 and counts[-1] == 20 * 60


def test_mixed_instances_contain_all_three_senses_sometimes():
    seen = set()
    for seed in range(30):
        seen |= set(S.generate("mixed", 12, 8, 0.5, seed).senses)
    assert seen == {S.LE, S.GE, S.EQ}
    for seed in range(30):
        inst = S.generate("mixed", 10, 4, 0.5, seed)
        assert sum(s == S.EQ for s in inst.senses) <= 2 and inst.senses[0] == S.LE


def test_transport_is_balanced_integer_feasible_and_has_two_entries_per_column():
    for seed in range(20):
        for k, l in ((2, 3), (4, 8), (8, 12)):
            inst = S.generate("transport", k, l, 0.5, seed)
            b, A = inst.b, np.array(inst.A)
            assert (inst.m, inst.n) == (k + l, k * l) and sum(b[:k]) == sum(b[k:]) and all(v == int(v) and v >= 1 for v in b)
            assert inst.senses == (S.LE,) * k + (S.GE,) * l and all(v <= 0 for v in inst.c) and (np.count_nonzero(A, axis=0) == 2).all()
            assert _highs(inst).status == 0
    assert S.generate("transport", 4, 8, 0.5, 3) == S.generate("transport", 4, 8, 0.1, 3) and S.generate("transport", 4, 8, 0.5, 3) != S.generate("transport", 4, 8, 0.5, 4)


def test_kinds_and_fixtures_are_consistent_and_unknown_kind_raises():
    assert set(S.FIXTURE_KINDS) <= set(S.KINDS) and set(S.KIND_LABELS) == set(S.KINDS)
    for kind in S.FIXTURE_KINDS:
        assert S.generate(kind, 9, 9, 0.5, 9) == S.generate(kind, 3, 3, 0.1, 1)
    with pytest.raises(ValueError):
        S.generate("nope", 3, 3, 0.5, 1)
