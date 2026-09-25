"""Auswertung: Analyse, Vergleichszeilen, Median über feste Instanzen, Kreuzungskurven, Neuinversion, partielle Preisgebung, Steepest-Kosten, Drift, Zustand vor einem Pivot."""

import numpy as np
import pytest

import rev_algorithm as A
import rev_constants as C
import rev_evaluation as ev
import rev_scenario as S
from rev_evaluation import Settings


def test_settings_effective_values():
    s = Settings("random", basis="explicit", refactor=5, rule="steepest", partial=0.5)
    assert s.refactor_eff == 0 and s.partial_eff == 1.0 and s.k_eff == 5
    t = Settings("random", basis="pfi", refactor=0, rule="dantzig", partial=0.25)
    assert t.refactor_eff == 0 and t.partial_eff == 0.25 and t.k_eff == C.DEFAULT_REFACTOR


def test_analyse_same_path_and_certificate_and_caching():
    a = ev.analyse(Settings("mixed", 20, 20, 0.2, 5))
    assert a.status == "optimal" and a.same_path is True and a.obj_gap < 1e-7 and ev.analyse(Settings("mixed", 20, 20, 0.2, 5)) is a
    assert max(p.drift for p in a.rev.pivots) < 1e-9
    b = ev.analyse(Settings("mixed", 20, 160, 0.1, 5, partial=0.125))
    assert b.same_path is None and b.status == "optimal" and b.rev.obj == pytest.approx(b.tab.obj, rel=1e-7)
    assert ev.analyse(Settings("infeasible")).status == "infeasible" and ev.analyse(Settings("unbounded")).status == "unbounded"
    assert ev.analyse(Settings("infeasible")).same_path is True and np.isnan(ev.analyse(Settings("unbounded")).obj_gap)


def test_methods_and_method_rows_have_consistent_numbers():
    s = Settings("transport", 4, 8, 0.5, 35)
    mm = ev.methods(s)
    assert list(mm) == ["tableau", "explicit", "pfi", "pfik"] and all(r.status == "optimal" for r in mm.values())
    assert len({r.total_pivots for r in mm.values()}) == 1
    rows = ev.method_rows(s)
    assert [r["name"] for r in rows] == ["tableau", "tableau_nz", "explicit", "pfi", "pfik"]
    assert rows[0]["ops"] == mm["tableau"].total_ops and rows[1]["ops"] == mm["tableau"].total_ops_nz < rows[0]["ops"]
    assert rows[2]["ops"] == mm["explicit"].total_ops and rows[4]["ops"] == mm["pfik"].total_ops and mm["pfik"].refactors > 0 and mm["pfi"].refactors == 0
    assert rows[0]["ops_parts"] is None and rows[2]["ops_parts"] == mm["explicit"].ops and all(r["ops_per_pivot"] == pytest.approx(r["ops"] / r["pivots"]) for r in rows)
    assert rows[0]["stored"] == mm["tableau"].stored and rows[1]["stored"] == max(p.nnz for p in mm["tableau"].pivots)


def test_run_config_medians_and_ratios():
    cfg = ev.run_config(Settings("random", 10, 40, 0.1, 35))
    assert cfg["n_runs"] == 5 and ev.run_config(Settings("random", 10, 40, 0.1, 35)) is cfg
    assert cfg["explicit_vs_dense"] == pytest.approx(cfg["explicit"] / cfg["dense"]) and cfg["pfik_vs_nz"] == pytest.approx(cfg["pfik"] / cfg["nz"]) and cfg["nz_vs_dense"] == pytest.approx(cfg["nz"] / cfg["dense"])
    assert cfg["dense"] > cfg["nz"] > 0 and cfg["stored_dense"] >= cfg["stored_tab"]
    single = ev.run_config(Settings("textbook"))
    assert single["n_runs"] == 1 and single["dense"] == 84 and single["nz"] == 35 and single["explicit"] == 102 and single["pfi"] == 54
    assert ev.run_config(Settings("infeasible")) is None


def test_crossover_rows_for_every_parameter_and_for_transport():
    s = Settings("mixed", 20, 20, 0.1, 35)
    size = ev.crossover(s, "size")
    assert [r["label"] for r in size] == [str(v) for v in ev.SIZE_VALUES]
    width = ev.crossover(Settings("random", 10, 10, 0.1, 35), "width")
    assert [r["label"] for r in width] == [f"{f}m" for f in ev.WIDTH_FACTORS]
    dens = ev.crossover(Settings("random", 10, 30, 0.1, 35), "density")
    assert [r["label"] for r in dens] == [f"{d:g}" for d in C.DENSITY_OPTIONS] and dens[-1]["dense"] > dens[0]["dense"]
    tr = ev.crossover(Settings("transport", 4, 8, 0.5, 35), "size")
    assert [r["label"] for r in tr] == [f"{k}x{l}" for k, l in ev.TRANSPORT_SIZES] and tr[-1]["pivots"] > tr[0]["pivots"]
    assert ev.crossover(Settings("textbook"), "size")[0]["label"] == "10"                     # Fixture: Vergleich auf gemischten Instanzen


def test_refactor_curve_is_u_shaped_and_counts_refactors():
    rows = ev.refactor_curve(Settings("mixed", 40, 40, 0.1, 35))
    assert [r["K"] for r in rows] == list(C.REFACTOR_OPTIONS)
    best = min(rows, key=lambda r: r["ops"])
    assert best["K"] in (3, 5, 10, 20) and rows[0]["ops"] > 1.5 * best["ops"] and rows[1]["ops"] > 1.5 * best["ops"] and rows[-1]["ops"] > best["ops"]
    assert rows[0]["refactor_ops"] == 0 and rows[1]["refactors"] == rows[1]["pivots"] and rows[1]["refactor_ops"] > rows[-1]["refactor_ops"] > 0
    assert all(r["drift"] < 1e-9 for r in rows) and rows[0]["stored"] > best["stored"]


def test_partial_curve_lowers_the_pricing_cost_per_pivot():
    rows = ev.partial_curve(Settings("mixed", 20, 160, 0.1, 35))
    assert [r["fraction"] for r in rows] == list(C.PARTIAL_OPTIONS)
    assert rows[-1]["price"] / rows[-1]["pivots"] < 0.4 * rows[0]["price"] / rows[0]["pivots"] and min(r["total"] for r in rows) < rows[0]["total"]


def test_steepest_cost_keys_and_pivot_savings():
    c = ev.steepest_cost(Settings("mixed", 40, 40, 0.1, 35))
    assert set(c) == {"dantzig", "steepest"} and c["steepest"]["pivots"] < c["dantzig"]["pivots"] and c["dantzig"]["weights_pfik"] == 0 < c["steepest"]["weights_pfik"]
    assert ev.steepest_cost(Settings("infeasible"))["steepest"] is None


def test_drift_series_and_limit():
    s = ev.drift_series(Settings("mixed", 20, 20, 0.2, 35))
    assert set(s) == {"explicit", "pfi", "pfik"} and len({len(v) for v in s.values()}) == 1 and max(max(v) for v in s.values()) < 1e-9
    assert ev.drift_series(Settings("random", 61, 61, 0.02, 35)) is None


def test_pivot_state_on_the_textbook_example_by_hand():
    inst = S.textbook_instance()
    res = A.revised_simplex(inst)
    s0 = ev.pivot_state(inst, res, 0)
    assert s0["basis"] == [2, 3, 4] and np.allclose(s0["Binv"], np.eye(3)) and np.allclose(s0["y"], 0) and np.allclose(s0["red"][:2], [-3, -5])
    assert s0["next"].enter == 1 and np.allclose(s0["d"], [0, 2, 2]) and s0["ratios"] == {1: 6.0, 2: 9.0}
    s1 = ev.pivot_state(inst, res, 1)
    assert s1["basis"] == [2, 1, 4] and np.allclose(s1["y"], [0, 2.5, 0]) and np.allclose(s1["xB"], [4, 6, 6]) and s1["next"].enter == 0 and s1["ratios"] == {0: 4.0, 2: 2.0}
    s2 = ev.pivot_state(inst, res, 2)
    assert s2["next"] is None and np.allclose(s2["y"], [0, 1.5, 1]) and (s2["red"][s2["allowed"]] >= -1e-9).all() and "d" not in s2


def test_pivot_state_phase_one_uses_the_artificial_costs():
    inst = next(i for i in (S.generate("mixed", 8, 8, 0.4, s) for s in range(40)) if A.tableau_simplex(i).phase1_pivots > 0)
    res = A.revised_simplex(inst)
    s0 = ev.pivot_state(inst, res, 0)
    assert s0["phase"] == 1 and s0["next"].phase == 1 and s0["y"].any()
    last = ev.pivot_state(inst, res, res.total_pivots)
    assert last["phase"] == 2 and last["next"] is None
