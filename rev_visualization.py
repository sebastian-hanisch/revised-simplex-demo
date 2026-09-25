"""Plotly-Abbildungen: Besetzungsmuster (Tableau gegen Revised), Operationen je Komponente, Verlauf je Pivot, Neuinversion, Drift, Kreuzungskurven, partielle Preisgebung, Steepest-Kosten.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

import rev_constants as C

TEAL, ORANGE, RED, BLUE, GREY, PURPLE, GREEN = "#2F6B65", "#e8a13a", "#d62728", "#1f4e9c", "#8a8f98", "#7b3fbf", "#3a8a3a"
COMP_COLORS = {"btran": BLUE, "price": ORANGE, "ftran": TEAL, "ratio": GREY, "update": PURPLE, "refactor": RED, "weights": GREEN}
METHOD_COLORS = {"tableau": GREY, "tableau_nz": "#5d6470", "explicit": ORANGE, "pfi": BLUE, "pfik": TEAL}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _pattern(z, title, color, xlabel=""):
    """Besetzungsmuster einer Matrix: farbig = Nichtnull; die Beschriftung nennt die Zahl der Nichtnullen."""
    z = np.asarray(z)
    nz = (np.abs(z) > 1e-12).astype(int)
    fig = go.Figure(go.Heatmap(z=nz, colorscale=[[0, "rgba(0,0,0,0.04)"], [1, color]], showscale=False, xgap=1, ygap=1, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed", showticklabels=False, showgrid=False, zeroline=False)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, title_text=f"{title}: {int(nz.sum())} von {nz.size}" if not xlabel else xlabel)
    fig.update_layout(height=260, margin=dict(l=5, r=5, t=5, b=40), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_patterns(state):
    """Drei Besetzungsmuster für den Zustand vor einem Pivot: das Tableau B^-1 [A | b] (das im Tableau-Verfahren je Pivot neu geschrieben wird), die feste Matrix M und die Inverse B^-1."""
    tab = np.hstack([state["Binv"] @ state["M"], (state["Binv"] @ state["b"]).reshape(-1, 1)])
    return {"tableau": _pattern(tab, "Tableau", ORANGE), "M": _pattern(state["M"], "Matrix M", TEAL), "Binv": _pattern(state["Binv"], "Inverse B⁻¹", BLUE)}


def build_ops_stack(rows):
    """Gestapelte Balken: Operationen je Verfahren, die Revised-Verfahren nach Komponenten."""
    fig = go.Figure()
    labels = [r["label"] for r in rows]
    for name in ("btran", "price", "ftran", "ratio", "update", "refactor", "weights"):
        ys = [(r["ops_parts"][name] if r["ops_parts"] else 0) for r in rows]
        if any(ys):
            fig.add_trace(go.Bar(y=labels, x=ys, orientation="h", name=C.COMPONENT_LABELS[name], marker_color=COMP_COLORS[name]))
    plain = [(r["ops"] if not r["ops_parts"] else 0) for r in rows]
    fig.add_trace(go.Bar(y=labels, x=plain, orientation="h", name="Pivot (Zeilenoperationen)", marker_color=GREY))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="Operationen (Nichtnull-gezählt bzw. dichtes Modell)")
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 300, legend_y=-0.45)


def build_per_pivot(mm):
    """Operationen je Pivot über den Lauf: Tableau (dicht: konstant; nur Nichtnullen), explizite Inverse, Produktform mit Neuinversion (Spitzen = Neuinversion)."""
    fig = go.Figure()
    tab = mm["tableau"]
    ncols, m = tab.n_cols, tab.m
    dense = (ncols + 1) + 2 * m * (ncols + 1)
    ks = list(range(1, tab.total_pivots + 1))
    fig.add_trace(go.Scatter(x=ks, y=[dense] * len(ks), mode="lines", line=dict(color=GREY, width=2, dash="dot"), name="Tableau, dichtes Modell"))
    fig.add_trace(go.Scatter(x=ks, y=[p.ops for p in tab.pivots], mode="lines", line=dict(color=METHOD_COLORS["tableau_nz"], width=2), name="Tableau, nur Nichtnullen"))
    for name, label in (("explicit", "Revised, explizite Inverse"), ("pfik", "Revised, Produktform + Neuinversion")):
        res = mm[name]
        fig.add_trace(go.Scatter(x=list(range(1, res.total_pivots + 1)), y=[p.ops for p in res.pivots], mode="lines", line=dict(color=METHOD_COLORS[name], width=2), name=label))
    fig.update_xaxes(title_text="Pivot")
    fig.update_yaxes(title_text="Operationen je Pivot (logarithmisch)", type="log")
    return _base(fig, 340, legend_y=-0.35)


def build_nnz(mm):
    """Gespeicherte Nichtnullen über den Lauf: das Tableau füllt sich (Fill-in), die Revised-Verfahren speichern die feste Matrix plus Inverse bzw. Eta-Datei."""
    fig = go.Figure()
    tab = mm["tableau"]
    fig.add_trace(go.Scatter(x=list(range(1, tab.total_pivots + 1)), y=[tab.stored] * tab.total_pivots, mode="lines", line=dict(color=GREY, width=2, dash="dot"), name="Tableau, dicht gespeichert"))
    for name, label in (("tableau", "Tableau, Nichtnullen"), ("explicit", "Revised, explizite Inverse"), ("pfik", "Revised, Produktform + Neuinversion")):
        res = mm[name]
        color = METHOD_COLORS["tableau_nz"] if name == "tableau" else METHOD_COLORS[name]
        fig.add_trace(go.Scatter(x=list(range(1, res.total_pivots + 1)), y=[p.nnz for p in res.pivots], mode="lines", line=dict(color=color, width=2), name=label))
    fig.update_xaxes(title_text="Pivot")
    fig.update_yaxes(title_text="Gespeicherte Nichtnullen")
    return _base(fig, 320, legend_y=-0.35)


def _kname(k):
    return "nie" if k == 0 else str(k)


def build_refactor(rows):
    """Produktform über die Neuinversions-Abstände K: Operationen gestapelt (ohne / mit Neuinversion), gespeicherte Nichtnullen als Linie."""
    xs = [_kname(int(r["K"])) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[r["ops"] - r["refactor_ops"] for r in rows], name="Iterationen", marker_color=TEAL))
    fig.add_trace(go.Bar(x=xs, y=[r["refactor_ops"] for r in rows], name="Neuinversion", marker_color=RED))
    fig.add_trace(go.Scatter(x=xs, y=[r["stored"] for r in rows], name="gespeicherte Nichtnullen (Spitze)", mode="lines+markers", yaxis="y2", line=dict(color=ORANGE, width=2.5)))
    fig.update_layout(barmode="stack", yaxis2=dict(overlaying="y", side="right", title="Nichtnullen", rangemode="tozero", showgrid=False, fixedrange=True))
    fig.update_xaxes(title_text="Neuinversion alle K Updates", type="category")
    fig.update_layout(yaxis_title_text="Operationen")
    return _base(fig, 340, legend_y=-0.35)


def build_drift(series):
    """Abweichung |B B^-1 - I| über die Pivots (logarithmisch); 0 wird als 1e-17 gezeichnet."""
    fig = go.Figure()
    for name, label in (("explicit", "explizite Inverse"), ("pfi", "Produktform ohne Neuinversion"), ("pfik", "Produktform mit Neuinversion")):
        ys = [max(v, 1e-17) for v in series[name]]
        fig.add_trace(go.Scatter(x=list(range(1, len(ys) + 1)), y=ys, mode="lines", line=dict(color=METHOD_COLORS[name], width=2), name=label))
    fig.update_xaxes(title_text="Pivot")
    fig.update_yaxes(title_text="max |B·B⁻¹ − I| (logarithmisch)", type="log", exponentformat="power")
    return _base(fig, 320, legend_y=-0.35)


def build_crossover(rows, xlabel):
    """Gesamtoperationen über einen Parameter (log-y) für das Tableau (dicht, nur Nichtnullen), die explizite Inverse und die Produktform mit Neuinversion."""
    xs = [r["label"] for r in rows]
    fig = go.Figure()
    for key, label, color, dash in (("dense", "Tableau, dichtes Modell", GREY, "dot"), ("nz", "Tableau, nur Nichtnullen", METHOD_COLORS["tableau_nz"], "solid"), ("explicit", "Revised, explizite Inverse", ORANGE, "solid"),
                                    ("pfik", "Revised, Produktform + Neuinversion", TEAL, "solid")):
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in rows], mode="lines+markers", line=dict(color=color, width=2.5, dash=dash), name=label))
    fig.update_xaxes(title_text=xlabel, type="category")
    fig.update_yaxes(title_text="Operationen (logarithmisch)", type="log")
    return _base(fig, 360, legend_y=-0.35)


def build_partial(rows):
    """Partielle Preisgebung: Pivots (Balken) sowie Preisgebung und Gesamtoperationen (Linien, zweite Achse)."""
    xs = [f"{r['fraction']:g}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[r["pivots"] for r in rows], name="Pivots", marker_color=GREY, opacity=0.6))
    fig.add_trace(go.Scatter(x=xs, y=[r["price"] for r in rows], name="Preisgebung", mode="lines+markers", yaxis="y2", line=dict(color=ORANGE, width=2.5)))
    fig.add_trace(go.Scatter(x=xs, y=[r["total"] for r in rows], name="Gesamt", mode="lines+markers", yaxis="y2", line=dict(color=TEAL, width=2.5)))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Operationen", rangemode="tozero", showgrid=False, fixedrange=True))
    fig.update_xaxes(title_text="Anteil der Spalten je Block", type="category")
    fig.update_layout(yaxis_title_text="Pivots")
    return _base(fig, 340, legend_y=-0.35)


def build_steepest(cost):
    """Verhältnis Steepest Edge zu Dantzig je Verfahren (Pivots und Operationen); unter 1 heißt: Steepest Edge lohnt."""
    d, s = cost["dantzig"], cost["steepest"]
    labels = ["Pivots", "Tableau (dicht)", "Tableau (Nichtnullen)", "Revised, explizit", "Revised, Produktform"]
    keys = ["pivots", "dense", "nz", "explicit", "pfik"]
    ys = [s[k] / d[k] if d[k] else float("nan") for k in keys]
    fig = go.Figure(go.Bar(x=labels, y=ys, marker_color=[GREY, GREY, METHOD_COLORS["tableau_nz"], ORANGE, TEAL], text=[f"{v:.2f}" for v in ys], textposition="outside"))
    fig.add_hline(y=1.0, line=dict(color=RED, dash="dot"))
    fig.update_yaxes(title_text="Steepest Edge / Dantzig", rangemode="tozero")
    return _base(fig, 320, legend_y=-0.3)
