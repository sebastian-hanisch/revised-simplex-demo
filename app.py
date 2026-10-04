"""Revised Simplex – dieselbe Rechnung, ein anderer Aufwand - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Viertes Stück der Lineare-Programmierung-Reihe der "Konzepte"-Reihe: Das dichte Tableau der Stücke 1-3 schreibt in jedem Pivot die ganze Matrix neu. Der Revised Simplex hält die Ausgangsmatrix fest und nur eine
Darstellung der Basis-Inverse. Die Demo zeigt einen Pivot von Hand, zählt den Aufwand je Komponente, vergleicht die Basisdarstellungen (explizite Inverse, Produktform mit Neuinversion) und misst, wann sich der Umbau lohnt.

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import rev_constants as C
import rev_evaluation as ev
import rev_scenario as S
from rev_evaluation import Settings, analyse
from rev_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    store_from_widget,
    sync_query_params,
)
from rev_visualization import (
    build_crossover,
    build_drift,
    build_nnz,
    build_ops_stack,
    build_partial,
    build_patterns,
    build_per_pivot,
    build_refactor,
    build_steepest,
)

st.set_page_config(page_title="Revised Simplex – Sebastian Hanisch", layout="wide")


def thousands(x):
    return f"{int(round(x)):,}".replace(",", " ")


def num(x, digits=2):
    return f"{x:.{digits}f}"


STATUS_TEXT = {"optimal": "Optimum", "infeasible": "unzulässig", "unbounded": "unbeschränkt", "cycled": "kreist", "limit": "Pivot-Grenze erreicht"}
CROSS_LABELS = {"size": "Größe (m = n)", "width": "Breite (n = Faktor mal m)", "density": "Dichte der Matrix"}

st.title("🧮 Revised Simplex – dieselbe Rechnung, ein anderer Aufwand")
st.markdown(
    """
**Viertes Stück der Lineare-Programmierung-Reihe.** Das dichte Tableau der ersten drei Stücke schreibt in jedem Pivot die ganze Matrix neu (etwa 2·m·(n+m) Operationen), obwohl nur **eine** Spalte eintritt. Der **Revised Simplex** hält
die Ausgangsmatrix A **unverändert** (dünn) und nur eine Darstellung der Basis-Inverse B⁻¹; je Pivot rechnet er die Duale y = c_B B⁻¹, bepreist die Spalten über A, berechnet die eintretende Spalte d = B⁻¹ a und aktualisiert die Basis.
Der **Pivotpfad ist derselbe**, nur die Rechnung ist eine andere. Vier Fragen, alle gemessen: **(1) Ein Pivot** - was hält der Revised Simplex, was das Tableau nicht? **(2) Aufwand** - was kostet jede Komponente, und wie füllt sich das Tableau?
**(3) Neuinversion** - explizite Inverse gegen Produktform, und wann lohnt eine neue Faktorisierung? **(4) Wann lohnt es** - Kreuzungspunkte, Steepest Edge und partielle Preisgebung.
"""
)
st.caption("Kind von [Tableau-Simplex](https://github.com/sebastian-hanisch/tableau-simplex-demo). Folgestücke (Dualität, Dualer Simplex, Innere Punkte, PDLP) sind inzwischen gebaut.")

with st.expander("So funktioniert der Revised Simplex", expanded=True):
    st.markdown(
        """
1. **Was gespeichert wird:** die feste Matrix M = [A | Schlupf | künstliche] (nur Nichtnullen zählen) und die Basis-Inverse B⁻¹ - **explizit** als dichte m×m-Matrix oder als **Produktform**: B⁻¹ ist ein Produkt von Eta-Matrizen (Einheitsmatrix mit einer ersetzten Spalte), jeder Pivot hängt eine an.
2. **Eine Iteration:** BTRAN (y = c_B B⁻¹) → **Preisgebung** (r_j = y·a_j − c_j, nur über M) → Regel wählt die eintretende Spalte q → FTRAN (d = B⁻¹ a_q) → Quotiententest → Update der Basis. Das Tableau rechnet dieselben Zahlen, schreibt dafür aber die ganze Matrix um.
3. **Neuinversion:** die Eta-Datei wächst mit jedem Pivot und macht BTRAN und FTRAN teurer. Nach K Pivots wird B neu faktorisiert (Gauß-Jordan-Etas für die aktuelle Basis): das kostet einmal, hält die Datei kurz und begrenzt Rundungsfehler.
4. **Steepest Edge im Revised:** die Gewichte γ_j = 1 + ‖B⁻¹a_j‖² müssen fortgeschrieben werden (Forrest und Goldfarb): das braucht die Pivotzeile und einen weiteren BTRAN, also eine zusätzliche Preisgebung. **Partielle Preisgebung:** nur ein Block der Spalten wird bepreist.
5. **Wie gezählt wird:** Multiply-Add = 2 Operationen, gezählt auf den tatsächlichen Nichtnullen; das Tableau einmal im dichten Modell der Stücke 1-3 und einmal ebenfalls nur auf Nichtnullen (fairer Vergleich). Ein Näherungsmaß, keine Laufzeit.
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:4], preset_names[4:7], preset_names[7:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

ss = st.session_state
with st.sidebar:
    st.header("⚙️ Einstellungen")
    kind = st.selectbox("Instanz", options=list(S.KINDS), format_func=lambda v: S.KIND_LABELS[v], key="kind_select",
                        help="Lehrbuchbeispiel und die Sonderfälle sind fest; Zufall, Mischung (Phase 1) und Transport (nur 2 Einträge je Spalte) sind regelbar.")
    random_kind = kind not in S.FIXTURE_KINDS
    if random_kind and kind == "transport":
        m = st.slider("Lager", *bounds("k_slider"), value=int(ss["k_slider"]), key="k_widget", on_change=store_from_widget, args=("k_slider",), help="Lager (Angebot); Lager mal Kunden sind die Variablen, jede Spalte hat 2 Einträge.")
        n = st.slider("Kunden", *bounds("l_slider"), value=int(ss["l_slider"]), key="l_widget", on_change=store_from_widget, args=("l_slider",), help="Kunden (Nachfrage).")
        density = C.DEFAULT_DENSITY
    elif random_kind:
        m = st.slider("Ressourcen m", *bounds("m_slider"), value=int(ss["m_slider"]), key="m_widget", on_change=store_from_widget, args=("m_slider",), help="Zahl der Bedingungen; bis 30 gibt es die Muster in Schritt 1.")
        n = st.slider("Dienste n", *bounds("n_slider"), value=int(ss["n_slider"]), key="n_widget", on_change=store_from_widget, args=("n_slider",), help="Zahl der Variablen: n größer als m heißt breit.")
        density = st.select_slider("Dichte der Matrix", options=list(C.DENSITY_OPTIONS), value=float(ss["density_select"]), key="density_widget", on_change=store_from_widget, args=("density_select",),
                                   help="Anteil der Dienste, die eine Ressource verbrauchen (Zeile 0 ist immer dicht).")
    else:
        m, n, density = C.DEFAULT_M, C.DEFAULT_N, C.DEFAULT_DENSITY
    rule = st.selectbox("Pivotregel", options=list(C.RULE_LABELS), format_func=lambda v: C.RULE_LABELS[v], key="rule_select",
                        help="Der Revised Simplex wählt dieselben Spalten wie das Tableau. Den größten Zuwachs gibt es hier nicht: er bräuchte je Kandidat einen FTRAN.")
    if random_kind or rule == "random":
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), value=int(ss["seed_input"]), key="seed_widget", step=1, on_change=store_from_widget, args=("seed_input",),
                               help="Seed der Instanz bzw. der Zufallsregel.")
        st.button("🎲 Neuer Seed", width="stretch", on_click=randomize_seed)
    else:
        seed = C.DEFAULT_SEED
    basis = st.radio("Basisdarstellung", options=list(C.BASIS_LABELS), format_func=lambda v: C.BASIS_LABELS[v], key="basis_select",
                     help="Explizit: dichtes B⁻¹, Update m² Operationen. Produktform: Eta-Datei, Update nur so viel wie Nichtnullen.")
    if basis == "pfi":
        refactor = st.select_slider("Neuinversion alle K Pivots", options=list(C.REFACTOR_OPTIONS), value=int(ss["refactor_select"]), key="refactor_widget", on_change=store_from_widget, args=("refactor_select",),
                                    format_func=lambda k: "nie" if k == 0 else str(k), help="0 = nie. Die Vergleiche in Schritt 2 und 4 verwenden bei 0 den Standardwert 10 für die Produktform mit Neuinversion.")
    else:
        refactor = 0
    if rule == "dantzig":
        partial = st.select_slider("Partielle Preisgebung: Anteil der Spalten je Block", options=list(C.PARTIAL_OPTIONS), value=float(ss["partial_select"]), key="partial_widget", on_change=store_from_widget,
                                   args=("partial_select",), help="1 = alle Spalten bepreisen. Kleinere Anteile bepreisen je Iteration nur einen Block (nur bei Dantzig); der Pfad ändert sich dann.")
    else:
        partial = 1.0

sync_query_params({"kind_select": kind, "m_slider": int(ss["m_slider"]), "n_slider": int(ss["n_slider"]), "k_slider": int(ss["k_slider"]), "l_slider": int(ss["l_slider"]), "density_select": float(ss["density_select"]),
                   "seed_input": int(ss["seed_input"]), "rule_select": rule, "basis_select": basis, "refactor_select": int(ss["refactor_select"]), "partial_select": float(ss["partial_select"]),
                   "rev_step": int(ss["rev_step"])})

settings = Settings(kind, int(m), int(n), float(density), int(seed), rule, basis, int(refactor), float(partial))
with st.spinner("Rechne..."):
    a = analyse(settings)
inst, tab, rev = a.inst, a.tab, a.rev

# --- In Aktion ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Dieselbe Rechnung, anders aufgebaut")
step = st.select_slider("Schritt", options=list(C.STEPS), key="rev_step", format_func=lambda s: C.STEPS[s])

if rev.status != "optimal":
    st.warning(f"Ergebnis: **{STATUS_TEXT[rev.status]}** nach {rev.total_pivots} Pivots.")
if a.same_path is True:
    st.success(f"✅ Derselbe Pivotpfad wie das Tableau: {rev.total_pivots} Pivots, gleiche eintretende und austretende Variablen, Optimum {num(rev.obj)}." if rev.status == "optimal"
               else f"✅ Derselbe Pivotpfad und derselbe Ausgang wie das Tableau ({STATUS_TEXT[rev.status]}).")
elif a.same_path is False:
    st.error("Der Pfad weicht vom Tableau ab (Gleichstand mit anderer Rundung) - bitte melden.")
else:
    st.info(f"Partielle Preisgebung ändert den Pfad: {rev.total_pivots} Pivots statt {tab.total_pivots} im Tableau; das Optimum ist dasselbe ({num(rev.obj)} gegen {num(tab.obj)})."
            if rev.status == tab.status == "optimal" else "Partielle Preisgebung ändert den Pfad.")

if step == 1:
    if inst.m > C.PIVOT_HEATMAP_MAX_M:
        st.info(f"Die Ein-Pivot-Ansicht gibt es bis m = {C.PIVOT_HEATMAP_MAX_M} Ressourcen (hier {inst.m}). Schritt 2 zeigt den Aufwand auch für große Instanzen.")
    else:
        total = rev.total_pivots
        if "pivot_k" in ss:
            ss["pivot_k"] = min(max(0, int(ss["pivot_k"])), total)
        k = st.slider("Vor Pivot", 0, total, key="pivot_k", help="0 = Startbasis; die Ansicht zeigt, was der Revised Simplex vor dem nächsten Pivot hält und rechnet.") if total > 0 else 0
        state = ev.pivot_state(inst, rev, k)
        names, basis_idx = state["names"], state["basis"]
        nxt = state["next"]
        if nxt is None:
            st.markdown(f"**Nach dem letzten Pivot** ({total}): keine Spalte mit negativen reduzierten Kosten, das Optimum ist erreicht. Duale y = ({', '.join(num(v, 2) for v in state['y'])}).")
        else:
            st.markdown(f"**Vor Pivot {k + 1} (Phase {state['phase']}):** {C.RULE_SHORT[rule]} wählt **{names[nxt.enter]}** (reduzierte Kosten {num(state['red'][nxt.enter])}); "
                        f"der Quotiententest lässt **{names[nxt.leave_var]}** austreten (Schrittweite {num(nxt.ratio)}).")
        if inst.m <= C.PIVOT_VIEW_MAX_M:
            c1, c2 = st.columns(2)
            rows_b = []
            for i in range(state["m"]):
                row = {"Basisvariable": names[basis_idx[i]], "x_B": num(state["xB"][i]), "y (Duale)": num(state["y"][i])}
                if nxt is not None:
                    row["d = B⁻¹ a"] = num(state["d"][i])
                    row["Quotient x_B / d"] = num(state["ratios"][i]) if i in state["ratios"] else "-"
                rows_b.append(row)
            with c1:
                st.markdown("**Basis:** Lösung, Duale (BTRAN) und eintretende Spalte (FTRAN)")
                st.dataframe(pd.DataFrame(rows_b), hide_index=True, width="stretch")
            nonbasic = [j for j in state["allowed"] if j not in basis_idx]
            order = sorted(nonbasic, key=lambda j: (state["red"][j], j))[:10]
            rows_r = [{"Spalte": names[j], "reduzierte Kosten y·a − c": num(state["red"][j]), "": "◀ tritt ein" if nxt is not None and j == nxt.enter else ""} for j in order]
            with c2:
                st.markdown(f"**Preisgebung** (nur über die feste Matrix; die {min(10, len(nonbasic))} kleinsten von {len(nonbasic)} Nichtbasisspalten)")
                st.dataframe(pd.DataFrame(rows_r), hide_index=True, width="stretch")
        figs = build_patterns(state)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(figs["tableau"], width="stretch", key=f"s1_tab_{k}")
        with c2:
            st.plotly_chart(figs["M"], width="stretch", key=f"s1_m_{k}")
        with c3:
            st.plotly_chart(figs["Binv"], width="stretch", key=f"s1_binv_{k}")
        st.caption("Farbig = Nichtnull (darunter: Nichtnullen von allen Einträgen). Links: das Tableau B⁻¹[A | b], das jeder Pivot des Tableau-Verfahrens komplett neu schreibt. Mitte: die feste Matrix M = [A | Schlupf | künstliche], die der "
                   "Revised Simplex nur liest. Rechts: die Inverse B⁻¹, die er aktualisiert (bei der Produktform nur als angehängte Eta-Spalte).")
elif step == 2:
    rows = ev.method_rows(settings)
    nzr = next(r for r in rows if r["name"] == "tableau_nz")["ops"]
    table = pd.DataFrame([{"Verfahren": r["label"], "Ergebnis": STATUS_TEXT[r["status"]], "Pivots": r["pivots"], "Operationen": thousands(r["ops"]), "je Pivot": thousands(r["ops_per_pivot"]),
                           "gegen Tableau (Nichtnullen)": f"{r['ops'] / nzr:.2f}" if nzr else "-", "gespeicherte Nichtnullen (Spitze)": thousands(r["stored"])} for r in rows])
    st.markdown(f"**Alle Verfahren auf derselben Instanz** ({S.KIND_LABELS[kind].split(' (')[0]}, {inst.m} Ressourcen, {inst.n} Variablen, {C.RULE_SHORT[rule]}; Produktform mit Neuinversion alle {settings.k_eff} Pivots):")
    st.dataframe(table, hide_index=True, width="stretch")
    st.plotly_chart(build_ops_stack(rows), width="stretch", key="s2_stack")
    st.caption("Operationen je Komponente. Das Tableau schreibt nur Pivots (grau). Beim Revised Simplex dominiert oft die Preisgebung (orange) oder die Neuinversion und der Abstieg durch die Eta-Datei (BTRAN, FTRAN).")
    mm = ev.methods(settings)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_per_pivot(mm), width="stretch", key="s2_perpivot")
        st.caption("Operationen je Pivot (logarithmisch). Die Spitzen der Produktform sind die Neuinversionen.")
    with c2:
        st.plotly_chart(build_nnz(mm), width="stretch", key="s2_nnz")
        st.caption("Gespeicherte Nichtnullen: das Tableau füllt sich (Fill-in), der Revised Simplex behält die feste Matrix und eine kleine Datei.")
    cfg = ev.run_config(settings)
    if cfg:
        st.markdown(f"**Median über {cfg['n_runs']} feste Instanzen** (Seeds 100000 bis 100004): {thousands(cfg['pivots'])} Pivots; Produktform mit Neuinversion **{cfg['pfik_vs_nz']:.2f}** des Nichtnull-Tableaus und **{cfg['pfik_vs_dense']:.2f}** "
                    f"des dichten Modells; explizite Inverse {cfg['explicit_vs_nz']:.2f} und {cfg['explicit_vs_dense']:.2f}.")
elif step == 3:
    st.markdown("Die **Produktform** über die Abstände K der Neuinversion (Median über 5 feste Instanzen; 🔬 auf Abruf): zu selten und die Eta-Datei wird lang, zu oft und die Neuinversion kostet mehr, als sie spart.")
    if st.button("Kurve berechnen (dauert einige Sekunden)", key="refactor_start"):
        ss["refactor_done"] = settings
    if ss.get("refactor_done") == settings:
        with st.spinner("Rechne..."):
            rows = ev.refactor_curve(settings)
        if rows:
            best = min(rows, key=lambda r: r["ops"])
            st.plotly_chart(build_refactor(rows), width="stretch", key="s3_refactor")
            st.markdown(f"**Minimum bei K = {int(best['K']) if best['K'] else 'nie'}:** {thousands(best['ops'])} Operationen, davon {thousands(best['refactor_ops'])} für die Neuinversion; ohne Neuinversion "
                        f"{thousands(next(r for r in rows if r['K'] == 0)['ops'])}.")
            st.dataframe(pd.DataFrame([{"K": "nie" if r["K"] == 0 else int(r["K"]), "Pivots": r["pivots"], "Operationen": thousands(r["ops"]), "davon Neuinversion": thousands(r["refactor_ops"]),
                                        "Neuinversionen": int(r["refactors"]), "gespeicherte Nichtnullen": thousands(r["stored"]), "max. Drift": f"{r['drift']:.1e}" if inst.m <= ev.DRIFT_MAX_M else "-"} for r in rows]),
                         hide_index=True, width="stretch")
        else:
            st.warning("Auf dieser Instanz gibt es keinen Lauf mit Optimum.")
    st.markdown("**Drift der Inverse:** |B·B⁻¹ − I| nach jedem Pivot (Diagnose, nicht mitgezählt).")
    series = ev.drift_series(settings)
    if series is None:
        st.info(f"Die Diagnose gibt es bis m = {ev.DRIFT_MAX_M} (sie braucht je Pivot die volle Inverse).")
    elif not series["pfi"]:
        st.info("Ohne Pivot gibt es keine Drift.")
    else:
        st.plotly_chart(build_drift(series), width="stretch", key="s3_drift")
        st.caption(f"Größter Fehler im Lauf: explizit {max(series['explicit']):.1e}, Produktform ohne Neuinversion {max(series['pfi']):.1e}, mit Neuinversion alle {settings.k_eff} Pivots {max(series['pfik']):.1e}. "
                   "Auf diesen Größen bleibt die Numerik harmlos; die Neuinversion lohnt wegen der Eta-Datei, nicht wegen des Fehlers.")
else:
    st.markdown("**Kreuzungspunkt:** Gesamtoperationen über die Größe, die Breite oder die Dichte (Median über 3 Instanzen; 🔬 auf Abruf; Transport: sechs Größen).")
    param = st.selectbox("Parameter", options=list(ev.CROSS_PARAMS), format_func=lambda p: CROSS_LABELS[p], key="cross_param", disabled=kind == "transport",
                         help="Größe: m = n = 10 bis 100; Breite: n = 1, 2, 4, 8 mal m; Dichte: 0.02 bis 1.")
    if st.button("Kurven berechnen (dauert einige Sekunden)", key="cross_start"):
        ss["cross_done"] = (settings, param)
    if ss.get("cross_done") == (settings, param):
        with st.spinner("Rechne..."):
            rows = ev.crossover(settings, param)
        xlabel = "Lager x Kunden" if kind == "transport" else CROSS_LABELS[param]
        st.plotly_chart(build_crossover(rows, xlabel), width="stretch", key="s4_cross")
        st.dataframe(pd.DataFrame([{xlabel: r["label"], "Pivots": r["pivots"], "Tableau dicht": thousands(r["dense"]), "Tableau Nichtnullen": thousands(r["nz"]), "explizit": thousands(r["explicit"]),
                                    "Produktform + Neuinversion": thousands(r["pfik"]), "Produktform / Tableau (Nichtnullen)": f"{r['pfik'] / r['nz']:.2f}", "Produktform / Tableau (dicht)": f"{r['pfik'] / r['dense']:.2f}"}
                                   for r in rows]), hide_index=True, width="stretch")
        st.caption("Unter 1 in den letzten beiden Spalten heißt: der Revised Simplex gewinnt. Der Verlauf ist nicht monoton: entscheidend sind die Länge des Laufs (Pivots) und wie stark sich das Tableau füllt.")
    st.markdown("**Steepest Edge im Revised Simplex** (Median über 5 feste Instanzen; 🔬 auf Abruf): weniger Pivots, aber die Gewichte kosten Preisgebung.")
    if st.button("Steepest-Edge-Vergleich berechnen", key="steep_start"):
        ss["steep_done"] = settings
    if ss.get("steep_done") == settings:
        with st.spinner("Rechne..."):
            cost = ev.steepest_cost(settings)
        if cost["dantzig"] and cost["steepest"]:
            st.plotly_chart(build_steepest(cost), width="stretch", key="s4_steepest")
            sp, sw = cost["steepest"], cost["dantzig"]
            st.caption(f"Pivots {thousands(sw['pivots'])} → {thousands(sp['pivots'])}; im Produktform-Lauf machen die Gewichte {thousands(sp['weights_pfik'])} von {thousands(sp['pfik'])} Operationen aus. "
                       "Unter der roten Linie (1) lohnt Steepest Edge, darüber nicht.")
        else:
            st.warning("Auf dieser Instanz gibt es keinen Lauf mit Optimum für beide Regeln.")
    st.markdown("**Partielle Preisgebung** (nur Dantzig, Produktform mit Neuinversion; Median über 5 feste Instanzen; 🔬 auf Abruf): kleinere Blöcke sparen Preisgebung, kosten aber Pivots.")
    if st.button("Partielle Preisgebung berechnen", key="partial_start"):
        ss["partial_done"] = settings
    if ss.get("partial_done") == settings:
        with st.spinner("Rechne..."):
            prow = ev.partial_curve(settings)
        if prow:
            st.plotly_chart(build_partial(prow), width="stretch", key="s4_partial")
            bestp = min(prow, key=lambda r: r["total"])
            st.caption(f"Am billigsten bei Anteil {bestp['fraction']:g}: {thousands(bestp['total'])} Operationen bei {thousands(bestp['pivots'])} Pivots (volle Preisgebung: {thousands(prow[0]['total'])}).")
        else:
            st.warning("Auf dieser Instanz gibt es keinen Lauf mit Optimum.")

st.markdown("---")

# --- Kennzahlen --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ⚙️ Der gewählte Lauf")
nz_ops = tab.total_ops_nz
m1, m2, m3, m4 = st.columns(4)
m1.metric("Pivots", thousands(rev.total_pivots), delta=f"Phase 1: {rev.phase1_pivots}", delta_color="off")
m2.metric("Operationen", thousands(rev.total_ops), delta=f"{thousands(rev.ops_per_pivot)} je Pivot", delta_color="off")
m3.metric("Gegen Tableau", f"{rev.total_ops / nz_ops:.2f}" if nz_ops else "-", delta="Nichtnullen", delta_color="off")
m4.metric("Ergebnis", ("-" if rev.status != "optimal" else (num(rev.obj) if abs(rev.obj) < 1000 else (thousands(rev.obj) if abs(rev.obj) < 1e7 else f"{rev.obj:.3g}"))), delta=STATUS_TEXT[rev.status], delta_color="off")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Revised Simplex ist billiger als das Tableau.** | Nur gegen das dichte Modell der Stücke 1-3 (Produktform mit Neuinversion: 0.07 bei Zufall 60 × 480, 0.13 bei Transport 8 × 12, 0.12 bei Mischung 80 × 80). Zählt man auch das Tableau nur auf Nichtnullen, gewinnt der Revised Simplex bei langen Läufen (Mischinstanz 80 × 80: Median 0.42) und verliert bei kurzen (Zufall 60 × 480, 8 Pivots: 2.52-fach). | Sparse-Tableau-Varianten, echte Löser |
| **Er spart Pivots.** | Nein: der Pfad ist derselbe. Er ändert nur die Kosten je Pivot. Nur die partielle Preisgebung ändert den Pfad. | Pivotregeln (Stück 2) |
| **Die explizite Inverse ist der Revised Simplex.** | Sie kostet m² je Update und verliert auf dichten und kurzen Läufen deutlich (Zufall 20 × 20: 1.47-fach des dichten Tableaus); die Produktform mit Neuinversion ist fast überall besser. | Sparse LU, Forrest-Tomlin |
| **Neuinversion schützt vor Rundungsfehlern.** | Auf diesen Größen ist der Fehler winzig (unter 1e-11 auch ohne Neuinversion bei Mischung 60 × 60); sie lohnt wegen der wachsenden Eta-Datei. | Numerik-Stück (Präsolve, Skalierung) |
| **Steepest Edge lohnt immer.** | Im Revised Simplex verursachen die Gewichte eine zusätzliche Preisgebung: bei Transport 6 × 10 sinken die Pivots auf 0.57, die Operationen steigen auf 1.19. | Devex, Referenzrahmen |
| **Das Zählmodell ist die Laufzeit.** | Es ist ein Näherungsmaß auf Nichtnullen; Cache, Speicherzugriffe und Sparse-Datenstrukturen fehlen. Echte Löser (Markowitz-LU, Forrest-Tomlin, Bound Flipping, Hypersparsity) sind nicht gebaut. | Echte Löser |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Standardform.** $\max c^\top x$ unter $Mx = b$, $x \ge 0$ mit $M = [A \mid \text{Schlupf} \mid \text{künstliche}]$; die Basis $B$ sind $m$ Spalten von $M$.

**Eine Iteration.** $y = c_B^\top B^{-1}$ (BTRAN), $r_j = y^\top a_j - c_j$ (Preisgebung; ein Kandidat hat $r_j < 0$), $d = B^{-1} a_q$ (FTRAN), Quotiententest $\theta = \min_{d_i > 0} x_{B,i} / d_i$ in Zeile $p$, $x_B \leftarrow x_B - \theta d$, $x_{B,p} = \theta$.

**Update der Inverse.** $B_\text{neu}^{-1} = E\,B^{-1}$ mit der Eta-Matrix $E$ (Einheitsmatrix, Spalte $p$ ersetzt durch $\eta$ mit $\eta_p = 1/d_p$ und $\eta_i = -d_i/d_p$). Explizit: $m$ + $2m$ je Nichtnull von $d$ Operationen. Produktform: $B^{-1} = E_k \cdots E_1$, FTRAN wendet die Etas nacheinander an, BTRAN in umgekehrter Reihenfolge.

**Steepest Edge (Forrest und Goldfarb).** $\gamma_j = 1 + \lVert B^{-1} a_j \rVert^2$. Nach dem Pivot ist mit $t = \alpha_{pj}/\alpha_{pq}$ und $v = B^{-\top} d$: $\tilde\gamma_j = \gamma_j - 2t\,(a_j^\top v) + t^2 \gamma_q$ (mindestens $1 + t^2$) und $\tilde\gamma_\text{austretend} = \gamma_q / \alpha_{pq}^2$.

**Aufwand.** BTRAN und FTRAN: 2 Operationen je Nichtnull des Vektors mal Spaltenlänge der Inverse bzw. je Nichtnull der Etas; Preisgebung: 2 je Nichtnull der bepreisten Spalten; Tableau: $(\text{Spalten}+1) + 2m(\text{Spalten}+1)$ dicht bzw. je Nichtnull der Pivotzeile und der Pivotspalte.

**Literatur.** Dantzig, G. B., & Orchard-Hays, W. (1954). *The product form for the inverse in the simplex method.* Mathematical Tables and Other Aids to Computation 8(46), 64-67. Forrest, J. J. H., & Tomlin, J. A. (1972). *Updated triangular factors of the basis to maintain sparsity
in the product form simplex method.* Mathematical Programming 2, 263-278 (nur genannt). Forrest, J. J., & Goldfarb, D. (1992). *Steepest-edge simplex algorithms for linear programming.* Mathematical Programming 57, 341-374.

Implementiert in `rev_algorithm.py` (Tableau, explizite Inverse, Produktform, Revised Simplex), `rev_scenario.py`, `rev_evaluation.py`.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Lineare Programmierung: vom Tableau zum Crossover](https://sebastianhanisch.net/konzepte-lineare-programmierung.html)."
)
