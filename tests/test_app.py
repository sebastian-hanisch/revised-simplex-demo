"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt für jede Instanz, Regel und Basisdarstellung, Pivot-Regler, Randwerte, Permalink-Grenzen, bedingte Regler, Kurven auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import rev_algorithm as A
import rev_constants as C
import rev_scenario as S

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(step=1, **state):
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if step != 1:
        at.select_slider(key="rev_step").set_value(step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m for m in at.metric if m.label.startswith(label))


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()


def test_default_run_shows_the_textbook_example_with_the_hand_counted_operations():
    at = _run()
    _ok(at)
    assert {"Pivots", "Operationen", "Gegen Tableau", "Ergebnis"} <= {m.label for m in at.metric}
    assert _metric(at, "Pivots").value == "2" and _metric(at, "Operationen").value == "54" and _metric(at, "Gegen Tableau").value == "1.54" and _metric(at, "Ergebnis").value == "36.00"
    assert any("Derselbe Pivotpfad wie das Tableau" in s.value for s in at.success) and at.get("plotly_chart")


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    _click(at, f"preset_{name}")
    _ok(at)
    p = C.PRESETS[name]
    ss = at.session_state
    assert (ss["kind_select"], ss["rev_step"], ss["rule_select"], ss["basis_select"], ss["refactor_select"], ss["partial_select"]) == (p["kind"], p["step"], p["rule"], p["basis"], p["refactor"], p["partial"])
    if p["kind"] == "transport":
        assert (ss["k_slider"], ss["l_slider"]) == (p["k"], p["l"])
    elif p["kind"] not in S.FIXTURE_KINDS:
        assert (ss["m_slider"], ss["n_slider"], ss["density_select"]) == (p["m"], p["n"], p["density"])
    if "pivot_k" in p:
        assert ss["pivot_k"] == p["pivot_k"]


@pytest.mark.parametrize("step", [1, 2, 3, 4])
@pytest.mark.parametrize("kind", list(S.KINDS))
def test_every_step_runs_for_every_kind(step, kind):
    at = _run(kind_select=kind, m_slider=8, n_slider=10, k_slider=3, l_slider=5, density_select=0.2, rev_step=step)
    _ok(at)
    assert at.session_state["rev_step"] == step


@pytest.mark.parametrize("rule", A.RULES)
@pytest.mark.parametrize("basis", A.BASES)
def test_every_rule_and_basis_runs_on_the_first_two_steps(rule, basis):
    for kind in ("mixed", "transport"):
        for step in (1, 2):
            _ok(_run(step=step, kind_select=kind, m_slider=8, n_slider=8, k_slider=3, l_slider=4, rule_select=rule, basis_select=basis))


def test_pivot_slider_moves_through_the_pivots_and_shows_tables_for_small_m():
    at = _run()
    _ok(at)
    slider = next(s for s in at.slider if s.key == "pivot_k")
    assert slider.max == 2 and len(at.dataframe) == 2
    for k in (0, 1, 2):
        at.slider(key="pivot_k").set_value(k).run()
        _ok(at)
    assert len(at.get("plotly_chart")) == 3 and any("Nach dem letzten Pivot" in m.value for m in at.markdown)
    at.slider(key="pivot_k").set_value(0).run()
    assert any("Vor Pivot 1 (Phase 2)" in m.value and "x2" in m.value for m in at.markdown)


def test_larger_m_shows_patterns_without_tables_and_very_large_m_only_a_note():
    mid = _run(kind_select="mixed", m_slider=20, n_slider=20)
    _ok(mid)
    assert not mid.dataframe and len(mid.get("plotly_chart")) == 3
    big = _run(kind_select="mixed", m_slider=40, n_slider=40)
    _ok(big)
    assert any("bis m = 30" in i.value for i in big.info) and not any(s.key == "pivot_k" for s in big.slider)


def test_step_two_table_lists_all_five_methods_with_the_textbook_numbers():
    at = _run(step=2)
    _ok(at)
    table = at.dataframe[0].value
    assert len(table) == 5 and list(table["Pivots"]) == [2] * 5 and list(table["Operationen"]) == ["84", "35", "102", "54", "54"]
    assert any("Median über 1 feste Instanzen" in m.value for m in at.markdown)


@pytest.mark.parametrize("kw", [dict(kind_select="random", m_slider=C.M_MAX, n_slider=C.N_MAX, density_select=0.02), dict(kind_select="random", m_slider=C.M_MIN, n_slider=C.N_MIN),
                                dict(kind_select="mixed", m_slider=C.M_MAX, n_slider=C.M_MAX, density_select=0.05, basis_select="explicit"), dict(kind_select="transport", k_slider=C.K_MAX, l_slider=C.L_MAX),
                                dict(kind_select="transport", k_slider=C.K_MIN, l_slider=C.L_MIN), dict(kind_select="random", density_select=1.0, m_slider=20, n_slider=20, rule_select="steepest"),
                                dict(refactor_select=0), dict(refactor_select=50), dict(kind_select="mixed", m_slider=10, n_slider=40, rule_select="dantzig", partial_select=0.0625)])
def test_extreme_settings_run(kw):
    for step in (1, 2):
        _ok(_run(step=step, **kw))


def test_step_three_curve_and_drift_on_demand():
    at = _run(step=3, kind_select="mixed", m_slider=12, n_slider=12, density_select=0.2)
    _ok(at)
    assert at.get("plotly_chart") and any("Drift" in m.value for m in at.markdown)
    _click(at, "refactor_start")
    _ok(at)
    assert any("Minimum bei K" in m.value for m in at.markdown) and len(at.get("plotly_chart")) == 2 and any(len(d.value) == len(C.REFACTOR_OPTIONS) for d in at.dataframe)
    big = _run(step=3, kind_select="mixed", m_slider=70, n_slider=70, density_select=0.05)
    _ok(big)
    assert any("bis m = 60" in i.value for i in big.info)


def test_step_four_sections_on_demand():
    at = _run(step=4, kind_select="mixed", m_slider=10, n_slider=10, density_select=0.2)
    _ok(at)
    _click(at, "cross_start")
    _ok(at)
    assert at.get("plotly_chart") and any(len(d.value) == len(("10", "20", "40", "60", "80", "100")) for d in at.dataframe)
    _click(at, "steep_start")
    _ok(at)
    assert any("Pivots" in c.value and "Gewichte" in c.value for c in at.caption)
    _click(at, "partial_start")
    _ok(at)
    assert any("Am billigsten bei Anteil" in c.value for c in at.caption)
    tr = _run(step=4, kind_select="transport", k_slider=3, l_slider=5)
    assert next(s for s in tr.selectbox if s.key == "cross_param").disabled
    _click(tr, "cross_start")
    _ok(tr)


def test_dice_button_changes_the_seed():
    at = _run(kind_select="random")
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neuer Seed").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old and at.session_state["seed_widget"] == at.session_state["seed_input"]


def test_permalink_values_are_clamped_and_invalid_choices_fall_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(m="999", n="1", k="1", l="99", density="0.42", step="9", kind="nope", rule="nope", basis="nope", refactor="7", partial="0.3", seed="-4").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["m_slider"], ss["n_slider"], ss["k_slider"], ss["l_slider"], ss["density_select"], ss["rev_step"], ss["kind_select"], ss["rule_select"], ss["basis_select"], ss["refactor_select"],
            ss["partial_select"], ss["seed_input"]) == (C.M_MAX, C.N_MIN, C.K_MIN, C.L_MAX, C.DEFAULT_DENSITY, 1, "textbook", C.DEFAULT_RULE, C.DEFAULT_BASIS, C.DEFAULT_REFACTOR, C.DEFAULT_PARTIAL, 0)


def test_permalink_accepts_valid_values():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(kind="mixed", m="30", n="50", density="0.05", seed="7", rule="dantzig", basis="pfi", refactor="20", partial="0.25", step="2").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["kind_select"], ss["m_slider"], ss["n_slider"], ss["density_select"], ss["seed_input"], ss["basis_select"], ss["refactor_select"], ss["partial_select"], ss["rev_step"]) == (
        "mixed", 30, 50, 0.05, 7, "pfi", 20, 0.25, 2)


def test_sidebar_shows_the_controls_that_belong_to_the_setting():
    fixed = _run()
    assert not any(w.key in ("m_widget", "k_widget") for w in fixed.slider) and not any(n.key == "seed_widget" for n in fixed.number_input)
    assert any(w.key == "refactor_widget" for w in fixed.select_slider) and any(w.key == "partial_widget" for w in fixed.select_slider)
    rnd = _run(kind_select="random", basis_select="explicit", rule_select="steepest")
    assert any(w.key == "m_widget" for w in rnd.slider) and any(w.key == "density_widget" for w in rnd.select_slider) and any(n.key == "seed_widget" for n in rnd.number_input)
    assert not any(w.key in ("refactor_widget", "partial_widget") for w in rnd.select_slider)
    tr = _run(kind_select="transport")
    assert any(w.key == "k_widget" for w in tr.slider) and any(w.key == "l_widget" for w in tr.slider) and not any(w.key == "m_widget" for w in tr.slider)
    assert any(n.key == "seed_widget" for n in _run(rule_select="random").number_input)


def test_changing_kind_rule_and_basis_on_later_steps_does_not_crash():
    for step in (2, 3, 4):
        at = _run(step=step)
        _ok(at)
        for kw in (dict(kind_select="mixed", m_slider=10, n_slider=10), dict(kind_select="transport"), dict(rule_select="steepest"), dict(basis_select="explicit"), dict(kind_select="infeasible"),
                   dict(kind_select="unbounded"), dict(rule_select="random"), dict(kind_select="random", m_slider=6, n_slider=30, basis_select="pfi", rule_select="dantzig", partial_select=0.5)):
            for k, v in kw.items():
                at.session_state[k] = v
            at.run()
            _ok(at)


def test_footer_limits_and_literature_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Dantzig, G. B., & Orchard-Hays, W. (1954)" in m.value and "Forrest, J. J., & Goldfarb, D. (1992)" in m.value for m in at.markdown)
