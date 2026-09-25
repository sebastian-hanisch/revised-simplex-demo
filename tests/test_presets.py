"""Presets: gültige Werte und jede Zahl im Hilfetext gegen die echten Auswertungsfunktionen (Zählgrößen sind exakt; Verhältnisse gerundet)."""

import pytest

import rev_constants as C
import rev_evaluation as ev
from rev_presets import PRESET_KEYS, SETTING_SPECS


def _settings(name):
    p = C.PRESETS[name]
    m, n = (p["k"], p["l"]) if p["kind"] == "transport" else (p["m"], p["n"])
    return ev.Settings(p["kind"], m, n, p["density"], p["seed"], p["rule"], p["basis"], p["refactor"], p["partial"])


def _rows(name):
    return {r["name"]: r for r in ev.method_rows(_settings(name))}


def _has(name, *values):
    for v in values:
        assert v in C.PRESET_HELP[name], (name, v)


def th(x):
    return f"{int(round(x)):,}".replace(",", " ")


def test_every_preset_has_valid_values_and_a_help_text():
    assert list(C.PRESETS) == list(C.PRESET_HELP) and len(C.PRESETS) == 10
    for name, p in C.PRESETS.items():
        assert set(p) <= set(PRESET_KEYS) and {"kind", "step", "rule", "basis"} <= set(p), name
        for key, state_key in PRESET_KEYS.items():
            if key in p and state_key in SETTING_SPECS:
                spec = SETTING_SPECS[state_key]
                assert spec.caster(p[key]) == p[key], (name, key)
                if spec.lo is not None:
                    assert spec.lo <= p[key] <= spec.hi, (name, key)
        assert C.PRESET_HELP[name].strip()
        if "pivot_k" in p:
            assert p["step"] == 1 and p["pivot_k"] > 0


def test_help_textbook_and_transport():
    r = _rows("Lehrbuchbeispiel von Hand")
    assert (r["tableau"]["ops"], r["tableau_nz"]["ops"], r["explicit"]["ops"], r["pfi"]["ops"]) == (84, 35, 102, 54)
    _has("Lehrbuchbeispiel von Hand", "Optimum 36", "84 Operationen", "nur auf Nichtnullen 35", "102", "54")
    assert C.PRESETS["Lehrbuchbeispiel von Hand"]["pivot_k"] <= r["tableau"]["pivots"]
    r = _rows("Transport: Revised gewinnt")
    assert r["tableau"]["pivots"] == 86 and (r["tableau"]["ops"], r["tableau_nz"]["ops"], r["pfik"]["ops"]) == (454854, 77399, 60440)
    assert r["pfik"]["ops"] / r["tableau_nz"]["ops"] == pytest.approx(0.78, abs=0.005) and r["pfik"]["ops"] / r["tableau"]["ops"] == pytest.approx(0.13, abs=0.005)
    cfg = ev.run_config(_settings("Transport: Revised gewinnt"))
    assert cfg["pfik_vs_nz"] == pytest.approx(0.78, abs=0.005) and cfg["pfik_vs_dense"] == pytest.approx(0.13, abs=0.005)
    _has("Transport: Revised gewinnt", "8 Lagern und 12 Kunden", "86 Pivots", "454 854", "77 399", "60 440", "0.78", "0.13", "2 Einträge")


def test_help_long_and_short_runs():
    r = _rows("Lange Läufe: Revised gewinnt")
    assert r["tableau"]["pivots"] == 235 and (r["tableau"]["ops"], r["tableau_nz"]["ops"], r["pfik"]["ops"]) == (7377825, 2900025, 827497)
    assert r["pfik"]["ops"] / r["tableau_nz"]["ops"] == pytest.approx(0.29, abs=0.005)
    cfg = ev.run_config(_settings("Lange Läufe: Revised gewinnt"))
    assert cfg["pfik_vs_nz"] == pytest.approx(0.32, abs=0.005) and cfg["explicit_vs_nz"] == pytest.approx(1.48, abs=0.005)
    _has("Lange Läufe: Revised gewinnt", "80 Ressourcen und 80 Diensten", "235 Pivots", "7 377 825", "2 900 025", "827 497", "(0.29)", "0.32", "1.48")
    r = _rows("Kurzer Lauf: Tableau gewinnt")
    assert r["tableau"]["pivots"] == 6 and (r["tableau"]["ops"], r["tableau_nz"]["ops"], r["pfik"]["ops"]) == (392766, 7075, 28791)
    assert r["pfik"]["ops"] / r["tableau_nz"]["ops"] == pytest.approx(4.07, abs=0.005) and r["pfik"]["ops"] / r["tableau"]["ops"] == pytest.approx(0.07, abs=0.005)
    cfg = ev.run_config(_settings("Kurzer Lauf: Tableau gewinnt"))
    assert cfg["pfik_vs_nz"] == pytest.approx(2.23, abs=0.005)
    a = ev.analyse(_settings("Kurzer Lauf: Tableau gewinnt")).rev
    assert a.ops["price"] / a.total_ops > 0.99
    _has("Kurzer Lauf: Tableau gewinnt", "60 Ressourcen und 480 Diensten", "nur 6 Pivots", "7 075", "28 791", "4.1-fach", "2.23", "392 766", "0.07")


def test_help_explicit_inverse_and_fill_in():
    r = _rows("Explizite Inverse verliert")
    assert r["tableau"]["pivots"] == 8 and (r["explicit"]["ops"], r["tableau"]["ops"], r["tableau_nz"]["ops"], r["pfi"]["ops"]) == (20603, 13448, 7216, 8323)
    assert r["explicit"]["ops"] / r["tableau"]["ops"] == pytest.approx(1.53, abs=0.005) and r["explicit"]["ops"] / r["tableau_nz"]["ops"] == pytest.approx(2.86, abs=0.005)
    _has("Explizite Inverse verliert", "20 Ressourcen und 20 Diensten", "8 Pivots", "20 603", "13 448", "1.53-fach", "7 216", "2.86-fach", "8 323")
    r = _rows("Fill-in des Tableaus")
    assert r["tableau"]["pivots"] == 425 and (r["tableau"]["stored"], r["tableau_nz"]["stored"], r["pfik"]["stored"]) == (23937, 13537, 5324)
    assert (r["tableau_nz"]["ops"], r["pfik"]["ops"]) == (9363559, 4844404) and r["pfik"]["ops"] / r["tableau_nz"]["ops"] == pytest.approx(0.52, abs=0.005)
    assert ev.run_config(_settings("Fill-in des Tableaus"))["pfik_vs_nz"] == pytest.approx(0.47, abs=0.005)
    _has("Fill-in des Tableaus", "425 Pivots", "13 537", "23 937", "5 324", "9 363 559", "4 844 404", "0.52", "0.47")


def test_help_refactor_curve_and_drift():
    rows = {int(r["K"]): r for r in ev.refactor_curve(_settings("Neuinversion: U-Kurve"))}
    assert [round(rows[K]["ops"]) for K in (0, 1, 2, 5, 10, 20, 50)] == [3618469, 3156874, 1905893, 1174015, 980375, 983608, 1263998] and min(rows, key=lambda K: rows[K]["ops"]) == 10
    _has("Neuinversion: U-Kurve", "60 Ressourcen und 60 Diensten", *[th(rows[K]["ops"]) for K in (0, 1, 2, 5, 10, 20, 50)], "(Minimum)")
    s = _settings("Drift bleibt klein")
    d = ev.drift_series(s)
    assert len(d["pfi"]) == 162 and s.refactor == 0 and s.k_eff == 10
    assert (f"{max(d['explicit']):.1e}", f"{max(d['pfi']):.1e}", f"{max(d['pfik']):.1e}") == ("1.4e-12", "3.0e-12", "1.4e-13")
    _has("Drift bleibt klein", "162 Pivots", "1.4e-12", "3.0e-12", "1.4e-13")


def test_help_steepest_and_partial():
    c = ev.steepest_cost(_settings("Steepest Edge im Revised"))
    d, s = c["dantzig"], c["steepest"]
    assert (d["pivots"], s["pivots"]) == (54, 31) and s["pivots"] / d["pivots"] == pytest.approx(0.57, abs=0.005) and s["nz"] / d["nz"] == pytest.approx(0.55, abs=0.005)
    assert s["pfik"] / d["pfik"] == pytest.approx(1.19, abs=0.005) and (s["pfik"], d["pfik"], s["weights_pfik"]) == (31336, 26328, 17726)
    _has("Steepest Edge im Revised", "6 Lagern und 10 Kunden", "31 statt 54", "0.57", "0.55", "1.19", "31 336", "26 328", "17 726")
    s = _settings("Partielle Preisgebung")
    a = ev.analyse(s)
    assert (a.rev.total_pivots, a.tab.total_pivots) == (78, 59) and a.rev.total_ops == 45865 and ev.methods(s)["pfik"].total_ops == 81951
    rows = {r["fraction"]: r for r in ev.partial_curve(s)}
    assert [round(rows[f]["total"]) for f in C.PARTIAL_OPTIONS] == [57000, 43151, 44912, 18701, 26120]
    _has("Partielle Preisgebung", "20 Ressourcen und 160 Diensten", "0.125", "78 statt 59 Pivots", "45 865", "81 951", "57 000, 43 151, 44 912, 18 701, 26 120")
