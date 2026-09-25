"""Jede Zahl aus README und App-Texten gegen die echten Auswertungsfunktionen (dieselben, die die App aufruft)."""

from pathlib import Path


import rev_algorithm as A
import rev_evaluation as ev
import rev_scenario as S
from rev_evaluation import Settings

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
APP_SRC = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")


def _has(*values):
    for v in values:
        assert v in README, v


def _ratios(rows, key="pfik", base="nz"):
    return [round(r[key] / r[base], 2) for r in rows]


def test_textbook_component_counts():
    inst = S.textbook_instance()
    e, p, t = A.revised_simplex(inst), A.revised_simplex(inst, basis_kind="pfi"), A.tableau_simplex(inst)
    assert (t.total_ops, t.total_ops_nz, e.total_ops, p.total_ops) == (84, 35, 102, 54)
    assert [e.ops[k] for k in ("btran", "price", "ftran", "ratio", "update")] == [18, 24, 24, 4, 32] and [p.ops[k] for k in ("btran", "price", "ftran", "ratio", "update")] == [8, 24, 0, 4, 18]
    _has("**84** Operationen", "**35**", "**102** (BTRAN 18, Preisgebung 24, FTRAN 24, Quotient 4, Update 32)", "**54** (BTRAN 8, Preisgebung 24, FTRAN 0, Quotient 4, Update 18)")


def test_dense_model_against_nonzeros():
    dense = ev.run_config(Settings("random", 40, 40, 1.0, 35))
    sparse = ev.run_config(Settings("random", 40, 40, 0.1, 35))
    assert round(dense["nz_vs_dense"], 2) == 0.52 and round(sparse["nz_vs_dense"], 2) == 0.04
    _has("**0.52** des dichten Modells", "**0.04** (Dichte 0.1)")


def test_transport_crossover_series():
    rows = ev.crossover(Settings("transport", 4, 8, 0.5, 35), "size")
    assert _ratios(rows) == [1.40, 1.20, 1.00, 0.83, 0.60, 0.48] and [int(r["pivots"]) for r in rows] == [18, 34, 54, 80, 153, 282]
    assert _ratios(rows, base="dense") == [0.33, 0.22, 0.17, 0.14, 0.09, 0.06] and round(rows[-1]["explicit"] / rows[-1]["dense"], 2) == 0.13
    _has("**1.40 / 1.20 / 1.00 / 0.83 / 0.60 / 0.48**", "(18, 34, 54, 80, 153, 282 Pivots)", "0.33 bis 0.06", "Transport 12 × 30 (282 Pivots) 0.48", "0.13 bei Transport 8 × 12")


def test_mixed_crossover_series_and_the_density_flip():
    rows = ev.crossover(Settings("mixed", 20, 20, 0.05, 35), "size")
    assert _ratios(rows) == [2.54, 4.72, 4.32, 1.56, 0.32, 0.47] and [int(r["pivots"]) for r in rows] == [7, 12, 38, 49, 223, 364]
    assert round(rows[-1]["explicit"] / rows[-1]["nz"], 2) == 1.45 and round(rows[2]["explicit"] / rows[2]["nz"], 1) == 23.1
    _has("**2.54 / 4.72 / 4.32 / 1.56 / 0.32 / 0.47**", "(7, 12, 38, 49, 223, 364 Pivots)", "Mischung 100 × 100: 1.45-fach", "bis 23-fach", "Mischung 20 × 20: 4.72")
    flip = ev.crossover(Settings("mixed", 20, 20, 0.1, 35), "size")
    assert _ratios(flip)[-1] == 1.42
    _has("(1.42)", "0.47 bei Dichte 0.05, 1.42 bei Dichte 0.1")
    dens = ev.crossover(Settings("mixed", 30, 120, 0.05, 35), "density")
    assert round(next(r for r in dens if r["label"] == "0.1")["explicit"] / next(r for r in dens if r["label"] == "0.1")["nz"], 2) == 0.68
    _has("Mischung 30 × 120, Dichte 0.1: 0.68")


def test_explicit_against_dense_on_a_dense_instance():
    cfg = ev.run_config(Settings("random", 20, 20, 1.0, 35))
    assert round(cfg["explicit_vs_dense"], 2) == 1.47
    _has("1.47-fach (Zufall 20 × 20 dicht)")


def test_klee_minty_cube_has_exactly_zero_drift():
    n = 9
    rows = [[2.0 ** (i - j + 1) if j < i else (1.0 if j == i else 0.0) for j in range(1, n + 1)] for i in range(1, n + 1)]
    inst = S.Instance(tuple(tuple(r) for r in rows), tuple(5.0 ** i for i in range(1, n + 1)), tuple(2.0 ** (n - j) for j in range(1, n + 1)), (S.LE,) * n, tuple(f"x{j}" for j in range(n)),
                      tuple(f"r{i}" for i in range(n)), "cube")
    for kind, K in (("explicit", 0), ("pfi", 0), ("pfi", 10)):
        r = A.revised_simplex(inst, basis_kind=kind, refactor_every=K, track_drift=True)
        assert r.total_pivots == 2 ** n - 1 and max(p.drift for p in r.pivots) == 0.0
    _has("Auf Klee-Minty-Würfeln (ganzzahlig) ist die Drift exakt null")


def test_refactor_and_drift_numbers():
    rows = {int(r["K"]): r for r in ev.refactor_curve(Settings("mixed", 60, 60, 0.1, 35))}
    assert round(rows[10]["refactor_ops"]) == 261790 and round(rows[10]["ops"]) == 980375 and round(rows[0]["ops"]) == 3618469
    _has("die Neuinversion kostet bei K = 10 261 790 Operationen", "**K = 10 980 375**", "nie 3 618 469", "K = 1 3 156 874", "K = 50 1 263 998", "980 375 Operationen gegen 3 618 469")
    _has("höchstens 3.0e-12", "**1.4e-12**", "**3.0e-12**", "**1.4e-13**", "162 Pivots")


def test_fill_in_and_storage_numbers():
    rows = {r["name"]: r for r in ev.method_rows(Settings("mixed", 100, 100, 0.05, 35, "dantzig", "pfi", 10))}
    assert (rows["tableau"]["stored"], rows["tableau_nz"]["stored"], rows["pfik"]["stored"]) == (23937, 13537, 5324)
    _has("**13 537** Nichtnullen (dicht gespeichert 23 937)", "**5 324**")


def test_steepest_numbers_and_the_mixed_counter_example():
    c = ev.steepest_cost(Settings("transport", 6, 10, 0.5, 35))
    d, s = c["dantzig"], c["steepest"]
    assert (round(s["pivots"] / d["pivots"], 2), round(s["nz"] / d["nz"], 2), round(s["dense"] / d["dense"], 2), round(s["pfik"] / d["pfik"], 2)) == (0.57, 0.55, 0.73, 1.19)
    assert (s["pfik"], d["pfik"], s["weights_pfik"]) == (31336, 26328, 17726)
    _has("(**0.57**)", "**0.55**", "im dichten Modell 0.73", "**1.19** (31 336 gegen 26 328)", "17 726 Operationen")
    m = ev.steepest_cost(Settings("mixed", 40, 40, 0.1, 35))
    assert (round(m["steepest"]["pivots"] / m["dantzig"]["pivots"], 2), round(m["steepest"]["dense"] / m["dantzig"]["dense"], 2), round(m["steepest"]["pfik"] / m["dantzig"]["pfik"], 2)) == (0.47, 0.59, 0.56)
    _has("0.47 der Pivots und 0.59 der Operationen im dichten Tableau, 0.56 im Revised mit Produktform")


def test_partial_pricing_numbers_and_the_grenzen_table():
    rows = {r["fraction"]: r for r in ev.partial_curve(Settings("mixed", 20, 160, 0.1, 35))}
    assert [round(r["total"]) for r in rows.values()] == [57000, 43151, 44912, 18701, 26120] and [int(r["pivots"]) for r in rows.values()] == [40, 45, 63, 45, 58]
    _has("**57 000 / 43 151 / 44 912 / 18 701 / 26 120**", "40 / 45 / 63 / 45 / 58 Pivots", "18 701 statt 57 000")
    cfg = ev.run_config(Settings("random", 60, 480, 0.05, 35))
    assert round(cfg["pfik_vs_nz"], 2) == 2.23 and round(cfg["pfik_vs_dense"], 2) == 0.07
    mixed = ev.run_config(Settings("mixed", 80, 80, 0.05, 35))
    assert round(mixed["pfik_vs_nz"], 2) == 0.32 and round(mixed["pfik_vs_dense"], 2) == 0.13
    for text in ("Median 0.32", "2.23-fach", "0.07 bei Zufall 60 × 480", "0.13 bei Transport 8 × 12 und Mischung 80 × 80", "3.0e-12", "0.57", "1.19"):
        assert text in APP_SRC, text
    _has("Zufall 60 × 480: 2.23", "0.07 bei Zufall 60 × 480, 0.13 bei Transport 8 × 12 und bei Mischung 80 × 80")


def test_literature_lines_are_in_the_readme_and_the_app():
    _has("Dantzig, G. B., & Orchard-Hays, W. (1954)", "Forrest, J. J. H., & Tomlin, J. A. (1972)", "Forrest, J. J., & Goldfarb, D. (1992)")
    for s in ("Dantzig, G. B., & Orchard-Hays, W. (1954)", "Forrest, J. J., & Goldfarb, D. (1992)", "Forrest, J. J. H., & Tomlin, J. A. (1972)"):
        assert s in APP_SRC
