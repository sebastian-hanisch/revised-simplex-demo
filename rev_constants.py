"""Konstanten der Demo Revised Simplex: Regler-Bereiche, feste Instanzen, Optionen (Werte aus der Vormessung), Presets."""
M_MIN, M_MAX, DEFAULT_M = 2, 100, 20
N_MIN, N_MAX, DEFAULT_N = 2, 500, 20
K_MIN, K_MAX, DEFAULT_K = 2, 12, 8                   # Lager im Transportproblem
L_MIN, L_MAX, DEFAULT_L = 2, 30, 12                  # Kunden im Transportproblem
DENSITY_OPTIONS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
DEFAULT_DENSITY = 0.1
SEED_MAX = 999999
DEFAULT_SEED = 35
RULE_LABELS = {"dantzig": "Dantzig (kleinste reduzierte Kosten)", "steepest": "Steepest Edge (Forrest-Goldfarb)", "bland": "Bland (kleinster Index)", "random": "Zufall"}
RULE_SHORT = {"dantzig": "Dantzig", "steepest": "Steepest Edge", "bland": "Bland", "random": "Zufall"}
DEFAULT_RULE = "dantzig"
BASIS_LABELS = {"explicit": "Explizite Inverse (dichtes B⁻¹)", "pfi": "Produktform (Eta-Datei)"}
DEFAULT_BASIS = "pfi"
REFACTOR_OPTIONS = (0, 1, 2, 3, 5, 10, 20, 50)        # Neuinversion alle K Updates; 0 = nie
DEFAULT_REFACTOR = 10
PARTIAL_OPTIONS = (1.0, 0.5, 0.25, 0.125, 0.0625)
DEFAULT_PARTIAL = 1.0
METHOD_LABELS = {"tableau": "Tableau (dicht, Modell der Stücke 1-3)", "tableau_nz": "Tableau (nur Nichtnullen)", "explicit": "Revised, explizite Inverse", "pfi": "Revised, Produktform ohne Neuinversion",
                 "pfik": "Revised, Produktform mit Neuinversion"}
COMPONENT_LABELS = {"btran": "BTRAN (Duale y)", "price": "Preisgebung", "ftran": "FTRAN (eintretende Spalte)", "ratio": "Quotiententest", "update": "Basis-Update", "refactor": "Neuinversion", "weights": "Gewichte (Steepest Edge)"}
STEPS = {1: "1 · Ein Pivot", 2: "2 · Aufwand", 3: "3 · Neuinversion", 4: "4 · Wann lohnt es"}
SWEEP_SEEDS = tuple(range(100000, 100005))
PIVOT_VIEW_MAX_M = 12                                # Ein-Pivot-Ansicht: Tabellen bis m = 12, Heatmaps bis m = 30
PIVOT_HEATMAP_MAX_M = 30
_BASE = {"kind": "textbook", "m": 20, "n": 20, "k": 8, "l": 12, "density": 0.1, "seed": 35, "rule": "dantzig", "basis": "pfi", "refactor": 10, "partial": 1.0, "step": 1}
PRESETS = {
    "Lehrbuchbeispiel von Hand": {**_BASE, "basis": "explicit", "pivot_k": 1},
    "Transport: Revised gewinnt": {**_BASE, "kind": "transport", "k": 8, "l": 12, "step": 2},
    "Lange Läufe: Revised gewinnt": {**_BASE, "kind": "mixed", "m": 80, "n": 80, "density": 0.05, "step": 2},
    "Kurzer Lauf: Tableau gewinnt": {**_BASE, "kind": "random", "m": 60, "n": 480, "density": 0.05, "step": 2},
    "Explizite Inverse verliert": {**_BASE, "kind": "random", "m": 20, "n": 20, "density": 1.0, "basis": "explicit", "step": 2},
    "Fill-in des Tableaus": {**_BASE, "kind": "mixed", "m": 100, "n": 100, "density": 0.05, "step": 2},
    "Neuinversion: U-Kurve": {**_BASE, "kind": "mixed", "m": 60, "n": 60, "density": 0.1, "step": 3},
    "Drift bleibt klein": {**_BASE, "kind": "mixed", "m": 60, "n": 60, "density": 0.1, "refactor": 0, "step": 3},
    "Steepest Edge im Revised": {**_BASE, "kind": "transport", "k": 6, "l": 10, "rule": "steepest", "step": 4},
    "Partielle Preisgebung": {**_BASE, "kind": "mixed", "m": 20, "n": 160, "density": 0.1, "partial": 0.125, "step": 4},
}
PRESET_HELP = {
    "Lehrbuchbeispiel von Hand": "Zwei Dienste, drei Ressourcen, Optimum 36 nach zwei Pivots. Das dichte Tableau kostet 84 Operationen (nur auf Nichtnullen 35), der Revised Simplex mit expliziter Inverse 102 und mit Produktform 54. Der Pivot-Regler zeigt B⁻¹, die Duale y, die reduzierten Kosten und die eintretende Spalte zum Nachrechnen von Hand.",
    "Transport: Revised gewinnt": "Transportproblem mit 8 Lagern und 12 Kunden (Seed 35): 86 Pivots. Dichtes Tableau 454 854 Operationen, nur auf Nichtnullen 77 399, Revised mit Produktform und Neuinversion alle 10 Pivots 60 440: 0.78 des Nichtnull-Tableaus und 0.13 des dichten (Median der fünf festen Instanzen ebenso 0.78 und 0.13). Jede Spalte hat nur 2 Einträge.",
    "Lange Läufe: Revised gewinnt": "Mischinstanz mit 80 Ressourcen und 80 Diensten, Dichte 0.05: 235 Pivots. Dichtes Tableau 7 377 825 Operationen, nur auf Nichtnullen 1 788 804, Produktform mit Neuinversion alle 10 Pivots 752 090 (0.42); im Median der fünf festen Instanzen 0.42. Die explizite Inverse liegt mit 1.88 des Nichtnull-Tableaus (Median) darüber.",
    "Kurzer Lauf: Tableau gewinnt": "Zufallsinstanz mit 60 Ressourcen und 480 Diensten, Dichte 0.05: nur 6 Pivots. Das Nichtnull-Tableau kostet 7 013 Operationen, die Produktform 28 788 (4.1-fach; Median 2.52), fast alles davon Preisgebung. Gegen das dichte Modell (392 766) gewinnt Revised trotzdem klar (0.07).",
    "Explizite Inverse verliert": "Dichte Zufallsinstanz mit 20 Ressourcen und 20 Diensten: 8 Pivots. Explizite Inverse 20 603 Operationen, dichtes Tableau 13 448 (1.53-fach), nur auf Nichtnullen 7 216 (2.86-fach); die Produktform braucht 8 323. Jeder Update der dichten Inverse kostet m² Operationen.",
    "Fill-in des Tableaus": "Mischinstanz mit 100 Ressourcen und 100 Diensten, Dichte 0.05: 425 Pivots. Das Tableau füllt sich bis auf 13 149 Nichtnullen (dicht gespeichert wären es 23 937), die Produktform mit Neuinversion hält höchstens 5 042. Operationen: 8 209 467 im Nichtnull-Tableau gegen 4 504 324 (0.55; Median 0.52).",
    "Neuinversion: U-Kurve": "Mischinstanz mit 60 Ressourcen und 60 Diensten, Dichte 0.1: Operationen der Produktform (Median über fünf Instanzen) bei Neuinversion alle K Pivots: nie 2 984 287, K = 1 3 116 851, K = 2 1 882 352, K = 5 1 137 921, K = 10 953 297 (Minimum), K = 20 955 386, K = 50 1 208 814.",
    "Drift bleibt klein": "Dieselbe Mischinstanz (Seed 35, 162 Pivots) ohne Neuinversion: der größte Fehler |B·B⁻¹ − I| über den Lauf liegt bei der expliziten Inverse und bei der Produktform in der Größenordnung 1e-12 (unter 1e-11), mit Neuinversion alle 10 Pivots eine Größenordnung darunter. Auf diesen Größen ist die Numerik kein Problem.",
    "Steepest Edge im Revised": "Transport mit 6 Lagern und 10 Kunden (Median über fünf Instanzen): Steepest Edge braucht 31 statt 54 Pivots (0.57). Im Nichtnull-Tableau sinken die Operationen auf 0.55, im Revised mit Produktform steigen sie auf 1.19 (31 336 gegen 26 328), weil die Gewichte 17 726 Operationen kosten: der Vorsprung geht verloren.",
    "Partielle Preisgebung": "Mischinstanz mit 20 Ressourcen und 160 Diensten: Anteil 0.125 der Spalten je Block. Seed 35: 78 statt 59 Pivots, aber 43 340 Operationen (volle Preisgebung 80 158). Median über fünf Instanzen für Anteil 1, 0.5, 0.25, 0.125, 0.0625: 55 433, 42 124, 43 462, 18 262, 24 583 Operationen: kein monotoner Verlauf.",
}
