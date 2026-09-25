# Revised Simplex – dieselbe Rechnung, ein anderer Aufwand – Streamlit-Demo

Viertes Stück der **Lineare-Programmierung-Reihe** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Kind der Wurzel [tableau-simplex-demo](https://github.com/sebastian-hanisch/tableau-simplex-demo). Das dichte Tableau der ersten drei Stücke schreibt in jedem Pivot die ganze Matrix neu (etwa 2·m·(n+m) Operationen), obwohl nur **eine** Spalte eintritt. Der **Revised Simplex** hält die Ausgangsmatrix **unverändert** (dünn) und nur eine Darstellung der Basis-Inverse B⁻¹; je Pivot rechnet er die Duale y = c_B B⁻¹ (BTRAN), bepreist die Spalten über die feste Matrix, berechnet die eintretende Spalte d = B⁻¹ a (FTRAN) und aktualisiert die Basis. Der **Pivotpfad ist derselbe**, nur die Rechnung ist eine andere. Die Demo hat zwei Basisdarstellungen (**explizite Inverse** und **Produktform** mit Eta-Datei und Neuinversion alle K Pivots), Steepest Edge nach Forrest und Goldfarb (Stück 2 hatte offen gelassen, was es im Revised kostet) und **partielle Preisgebung**. Vier Fragen, alle gemessen: **(1) Ein Pivot** – was hält der Revised Simplex, was das Tableau nicht hält? **(2) Aufwand** – was kostet jede Komponente, und wie füllt sich das Tableau? **(3) Neuinversion** – wann lohnt eine neue Faktorisierung, und wie groß wird die Drift? **(4) Wann lohnt es** – Kreuzungspunkte, Steepest Edge, partielle Preisgebung.

**Einordnung in die Reihe:** geplant sind zwölf Stücke, dies ist das vierte (Details in `lp-planung/PLAN.md` des Portfolio-Ordners):

```
Tableau-Simplex (Wurzel)                                                                  [gebaut: tableau-simplex-demo]
 ├─ Pivotregeln & Entartung ─ Simplex im schlimmsten und im typischen Fall (Klee-Minty)   [gebaut: pivotregeln-demo, klee-minty-demo]
 ├─ Revised Simplex ─ Präsolve, Skalierung & Numerik                                     [DIESES STÜCK]  →  [nicht gebaut]
 ├─ Dualität & Sensitivität ─ Dualer Simplex & Neuoptimierung                            [nicht gebaut]
 ├─ Ellipsoid-Methode (Kontrast: polynomial in der Theorie)                              [nicht gebaut]
 └─ Innere Punkte ─ PDLP (Verfahren erster Ordnung) ─ Crossover & Simplex gegen Innere Punkte gegen PDLP  [nicht gebaut]
```

Ergebnis in Kürze: **Der Revised Simplex ist kein Gewinn für alle, und wie stark er gewinnt, hängt daran, gegen welches Tableau man ihn hält.** Gegen das **dichte Tableau-Modell** der Stücke 1 bis 3 gewinnt die Produktform mit Neuinversion fast überall (0.07 bei Zufall 60 × 480, 0.13 bei Transport 8 × 12 und bei Mischung 80 × 80). Zählt man das Tableau ebenfalls **nur auf Nichtnullen**, bleibt der Vorteil nur bei **langen Läufen** und wachsendem Fill-in: Mischung 80 × 80 0.32, Transport 12 × 30 (282 Pivots) 0.48; bei **kurzen Läufen** verliert der Revised Simplex bis zum Vierfachen (Zufall 60 × 480: 2.23; Mischung 20 × 20: 4.72). Die **explizite Inverse** schlägt das Nichtnull-Tableau fast nie (Mischung 100 × 100: 1.45-fach). Die **Neuinversion** hat ein U-förmiges Optimum (Mischung 60 × 60: K = 10 mit 980 375 Operationen gegen 3 618 469 ohne Neuinversion). **Drift** bleibt auf diesen Größen winzig (höchstens 3.0e-12). **Steepest Edge** spart im Revised seine Pivots (0.57 bei Transport 6 × 10), aber nicht die Operationen (1.19-fach). **Partielle Preisgebung** kann die Gesamtkosten stark senken (Mischung 20 × 160: 18 701 statt 57 000 Operationen bei Anteil 0.125), verläuft aber nicht monoton.

| Frage | Ergebnis (Distributionszentrum, Standard-LP max c·x; **Median** über 5 feste Instanzen, Seeds 100000–100004, Kreuzungskurven über 3 Instanzen; vollständig deterministisch; Aufwand in Gleitkomma-Operationen, gezählt auf den tatsächlichen Nichtnullen, Multiply-Add = 2; Produktform mit Neuinversion alle 10 Pivots, wenn nicht anders angegeben) |
|---|---|
| **Gleicher Pfad?** | ✅ Für Dantzig, Bland, Zufall und Steepest Edge und jede Basisdarstellung (explizit, Produktform, Produktform mit Neuinversion alle 3 Pivots) läuft der Revised Simplex **Pivot für Pivot** denselben Weg wie das Tableau (eintretende und austretende Variable, Phase, Zielwert; Zufall, Mischung mit Phase 1, Transport, breit und dünn; über 90 Instanzen je Kombination), das Optimum stimmt mit HiGHS überein (Zielwert, primal und dual zulässig, starke Dualität). Die Neuinversion darf die Zeilenzuordnung der Basis vertauschen, der Pfad ändert sich nicht |
| **Von Hand nachgezählt** | Lehrbuchbeispiel (2 Pivots, Optimum 36): dichtes Tableau **84** Operationen, nur auf Nichtnullen **35**, explizite Inverse **102** (BTRAN 18, Preisgebung 24, FTRAN 24, Quotient 4, Update 32), Produktform **54** (BTRAN 8, Preisgebung 24, FTRAN 0, Quotient 4, Update 18): jede Zahl im Test von Hand hergeleitet |
| **Dichtes Modell gegen Nichtnullen** | Das Tableau, nur auf Nichtnullen gezählt, kostet nur **0.52** des dichten Modells (Zufall 40 × 40 dicht) bzw. **0.04** (Dichte 0.1): der dichte Vergleich der Stücke 1 bis 3 überschätzt den Vorteil des Revised Simplex deutlich. Beide Vergleiche stehen nebeneinander |
| **Transport (2 Einträge je Spalte)** | Produktform gegen Nichtnull-Tableau bei 3 × 5 / 4 × 8 / 6 × 10 / 8 × 12 / 10 × 20 / 12 × 30: **1.40 / 1.20 / 1.00 / 0.83 / 0.60 / 0.48** (18, 34, 54, 80, 153, 282 Pivots); gegen das dichte Modell 0.33 bis 0.06. Der Vorteil wächst mit der Größe |
| **Mischung (Dichte 0.05, m = n)** | Produktform gegen Nichtnull-Tableau bei m = 10 / 20 / 40 / 60 / 80 / 100: **2.54 / 4.72 / 4.32 / 1.56 / 0.32 / 0.47** (7, 12, 38, 49, 223, 364 Pivots). Es gibt keinen einzelnen Kreuzungspunkt: bei kurzen Läufen bleibt das Tableau dünn und billig, bei langen füllt es sich; bei Dichte 0.1 kippt es bei m = 100 wieder (1.42), weil auch die Eta-Datei anwächst |
| **Explizite Inverse** | Kostet m² je Update: gegen das dichte Modell nur 1.47-fach (Zufall 20 × 20 dicht) bis 0.13; gegen das Nichtnull-Tableau bei Mischung 100 × 100 **1.45-fach**, bei kurzen Läufen bis 23-fach. Sie gewinnt nur bei breiten, mittel dichten Instanzen (Mischung 30 × 120, Dichte 0.1: 0.68) |
| **Speicher (Fill-in)** | Mischung 100 × 100, Dichte 0.05: das Tableau füllt sich bis auf **13 537** Nichtnullen (dicht gespeichert 23 937), die Produktform mit Neuinversion hält höchstens **5 324** (die feste Matrix eingerechnet) |
| **Neuinversion** | Mischung 60 × 60, Dichte 0.1, Operationen der Produktform bei Neuinversion alle K Pivots: nie 3 618 469, K = 1 3 156 874, K = 2 1 905 893, K = 5 1 174 015, **K = 10 980 375**, K = 20 983 608, K = 50 1 263 998: **U-förmig**; die Neuinversion kostet bei K = 10 261 790 Operationen, spart aber die langen Eta-Läufe |
| **Drift** | |B·B⁻¹ − I| über den Lauf (Mischung 60 × 60, 162 Pivots): explizit **1.4e-12**, Produktform ohne Neuinversion **3.0e-12**, mit Neuinversion alle 10 Pivots **1.4e-13**. Auf den Größen der Demo ist die Numerik kein Problem: die Neuinversion lohnt wegen der Eta-Datei, nicht wegen des Fehlers. Auf Klee-Minty-Würfeln (ganzzahlig) ist die Drift exakt null (in der Vormessung geprüft) |
| **Steepest Edge im Revised** | Transport 6 × 10: 31 statt 54 Pivots (**0.57**); Operationen im Nichtnull-Tableau **0.55**, im dichten Modell 0.73, im Revised mit Produktform **1.19** (31 336 gegen 26 328), weil die Gewichte 17 726 Operationen kosten (eine zusätzliche Preisgebung über die Pivotzeile, ein BTRAN mehr). Der Vorsprung aus Stück 2 (Mischung: 0.63 der Operationen im dichten Modell) ist im Revised bei Transport verloren |
| **Partielle Preisgebung** | Mischung 20 × 160, Dantzig, Produktform mit Neuinversion: Anteil der Spalten je Block 1 / 0.5 / 0.25 / **0.125** / 0.0625: **57 000 / 43 151 / 44 912 / 18 701 / 26 120** Operationen bei 40 / 45 / 63 / 45 / 58 Pivots. Weniger Preisgebung, aber ein anderer Pfad mit teils mehr Pivots: kein monotoner Verlauf, das Minimum liegt bei einem kleinen Block |

## Was die Demo zeigt

1. **Vier Schritte** (Schritt-Slider): **Ein Pivot** (Regler über die Pivots bis m = 30: Basis mit Lösung, Dualen und eintretender Spalte, die kleinsten reduzierten Kosten der Preisgebung, drei Besetzungsmuster: das Tableau, das jeder Pivot neu schreibt, die feste Matrix und die Inverse; für m ≤ 12 die Tabellen zum Nachrechnen von Hand) → **Aufwand** (alle Verfahren auf derselben Instanz: Tabelle, Operationen je Komponente gestapelt, Operationen je Pivot über den Lauf, gespeicherte Nichtnullen über den Lauf, Median über 5 feste Instanzen) → **Neuinversion** (Kurve über K, Drift der Inverse über die Pivots; auf Abruf) → **Wann lohnt es** (Kreuzungskurven über Größe, Breite und Dichte bzw. sechs Transportgrößen, Steepest-Edge-Vergleich, partielle Preisgebung; auf Abruf).
2. **Kennzahlen des gewählten Laufs:** Pivots, Operationen, Verhältnis zum Nichtnull-Tableau, Ergebnis; darüber die Meldung "derselbe Pivotpfad wie das Tableau".
3. **Instanzen:** Lehrbuchbeispiel, Unzulässig, Unbeschränkt, Zufall (Dichte 0.02 bis 1, bis 100 × 500), Mischung (Phase 1), Transport (2 Einträge je Spalte).

Presets (10): Lehrbuchbeispiel von Hand, Transport: Revised gewinnt, Lange Läufe: Revised gewinnt, Kurzer Lauf: Tableau gewinnt, Explizite Inverse verliert, Fill-in des Tableaus, Neuinversion: U-Kurve, Drift bleibt klein, Steepest Edge im Revised, Partielle Preisgebung.

## Messwerte der Presets

| Preset | Einstellungen | Ergebnis |
|---|---|---|
| **Lehrbuchbeispiel von Hand** | 2 Dienste, explizite Inverse | 2 Pivots, Optimum 36; 84 / 35 / 102 / 54 Operationen (Tableau dicht / Nichtnullen / explizit / Produktform) |
| **Transport: Revised gewinnt** | 8 Lager, 12 Kunden, Produktform K = 10 | 86 Pivots; 454 854 / 77 399 / 60 440 Operationen (0.78 des Nichtnull-Tableaus, 0.13 des dichten) |
| **Lange Läufe: Revised gewinnt** | Mischung 80 × 80, Dichte 0.05 | 235 Pivots; 7 377 825 / 2 900 025 / 827 497 (0.29; Median 0.32); explizit 1.48 des Nichtnull-Tableaus |
| **Kurzer Lauf: Tableau gewinnt** | Zufall 60 × 480, Dichte 0.05 | 6 Pivots; 392 766 / 7 075 / 28 791 (4.1-fach; Median 2.23), fast alles Preisgebung |
| **Explizite Inverse verliert** | Zufall 20 × 20, dicht, explizit | 8 Pivots; explizit 20 603, dicht 13 448 (1.53-fach), Nichtnullen 7 216 (2.86-fach); Produktform 8 323 |
| **Fill-in des Tableaus** | Mischung 100 × 100, Dichte 0.05 | 425 Pivots; Nichtnullen 13 537 gegen höchstens 5 324; Operationen 9 363 559 gegen 4 844 404 (0.52; Median 0.47) |
| **Neuinversion: U-Kurve** | Mischung 60 × 60, Dichte 0.1 | Minimum bei K = 10 (980 375 Operationen), nie 3 618 469 |
| **Drift bleibt klein** | wie oben, ohne Neuinversion | 162 Pivots; 1.4e-12 / 3.0e-12 / 1.4e-13 |
| **Steepest Edge im Revised** | Transport 6 × 10 | 31 statt 54 Pivots; Operationen 0.55 (Nichtnullen-Tableau), 1.19 (Revised) |
| **Partielle Preisgebung** | Mischung 20 × 160, Anteil 0.125 | Seed 35: 78 statt 59 Pivots, aber 45 865 statt 81 951 Operationen |

## Modell und Verfahren

- **Instanz** (`rev_scenario.py`): die Auslastungsplanung der Vorgängerstücke (Lehrbuchbeispiel, Zufall mit Dichte, Mischung mit ≥ und =, Transport mit ganzzahligem Angebot gleich Nachfrage und 2 Einträgen je Spalte, Unzulässig, Unbeschränkt); Zeile 0 ist dicht, damit alles beschränkt bleibt.
- **Tableau** (`tableau_simplex`): das dichte Tableau der Stücke 1 bis 3 als Referenz, mit zwei Zählmodellen: dicht (Spalten + 1) + 2 m (Spalten + 1) je Pivot wie bisher, und nur auf Nichtnullen (Division der Pivotzeile plus 2 je Nichtnull der Pivotzeile für jede Zeile mit Faktor ≠ 0, auch die Zielzeile); Preisgebung der Regel wie in Stück 2 bzw. auf den Nichtnullen der Spalten. Gleichstände bei der Spaltenwahl (relative Toleranz 1e-9) gewinnt der kleinste Index, damit beide Verfahren bei ganzzahligen Daten dieselbe Spalte wählen.
- **Revised Simplex** (`revised_simplex`): feste Matrix M = [A | Schlupf | künstliche], Zwei-Phasen-Start wie im Tableau, Regeln Dantzig, Steepest Edge (Forrest-Goldfarb-Gewichte γ_j = 1 + ‖B⁻¹a_j‖², fortgeschrieben mit γ̃_j = γ_j − 2t (a_j·v) + t² γ_q, v = B⁻ᵀ d, t = α_pj/α_pq; die austretende Variable bekommt γ_q/α_pq²), Bland, Zufall; **den größten Zuwachs gibt es nicht** (er bräuchte je Kandidat einen FTRAN). **Partielle Preisgebung** (nur Dantzig): zyklische Blöcke der Nichtbasisspalten, das Optimum gilt erst nach einem sauberen Umlauf über alle Blöcke.
- **Basisdarstellungen:** `ExplicitInverse` (dichtes B⁻¹, Rang-1-Update auf den Zeilen mit d_i ≠ 0) und `ProductForm` (B⁻¹ = E_k ··· E_1 mit Eta-Matrizen; FTRAN und BTRAN laufen die Datei ab und zählen nur Nichtnullen; Neuinversion nach K Updates als Gauß-Jordan-Etas für die aktuelle Basis: Einheitsspalten zuerst, dann die übrigen nach Spaltenbelegung, Pivotzeile größter Betrag).
- **Aufwand je Komponente** (`Result.ops`): BTRAN, Preisgebung (2 je Nichtnull der bepreisten Spalten), FTRAN, Quotiententest, Update, Neuinversion, Gewichte (Steepest Edge). Ein Näherungsmaß, keine Laufzeit.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Hypothese "der Revised Simplex gewinnt bei dünnen und breiten Instanzen" – nur gegen das dichte Modell bestätigt.** Gegen das Tableau, das nur Nichtnullen zählt, gewinnt er bei langen Läufen (Transport, Mischung ab m = 80) und verliert bei kurzen: der Vorteil hängt an der Länge des Laufs und am Fill-in, nicht an der Form der Instanz allein. Ein einzelner Kreuzungspunkt existiert nicht (Mischung 100 × 100: 0.47 bei Dichte 0.05, 1.42 bei Dichte 0.1).
- **Vorab-Hypothese "die explizite Inverse ist bei kleinem m nicht schlechter als die Produktform" – widerlegt.** Die Produktform gewinnt schon bei 20 × 20; die explizite Inverse gewinnt nur bei breiten mittel dichten Instanzen.
- **Vorab-Hypothese "Drift zeigt sich ohne Neuinversion" – ehrlich negativ.** Der größte Fehler bleibt bei 3.0e-12; die Neuinversion lohnt wegen der wachsenden Eta-Datei (U-Kurve), nicht wegen der Genauigkeit. Die Demo zeigt das mit der Diagnose |B·B⁻¹ − I| (nicht mitgezählt, bis m = 60).
- **Vorab-Hypothese "Steepest Edge verliert im Revised an Vorsprung" – bei Transport bestätigt, bei Mischung nicht.** Mischung 40 × 40 (Dichte 0.1): 0.47 der Pivots und 0.59 der Operationen im dichten Tableau, 0.56 im Revised mit Produktform (Vormessung).
- **Kein Sparse-LU.** Echte Löser faktorisieren B mit Markowitz-Pivotwahl und aktualisieren die Faktoren (Forrest-Tomlin, Bartels-Golub), nutzen Hypersparsity, Bound Flipping und Präsolve: hier nur genannt. Die Produktform ist die historische Darstellung (Dantzig und Orchard-Hays 1954).
- **Kein Devex, kein größter Zuwachs, keine partielle Preisgebung für die anderen Regeln.** Steepest Edge braucht die ganze Pivotzeile und verträgt sich nicht mit partieller Preisgebung.
- **Zählmodell, keine Laufzeit.** Cache, Speicherzugriffe, Sparse-Datenstrukturen und Python-Overhead fehlen; das Nichtnull-Tableau ist ein Modell (die Stücke 1 bis 3 haben dicht gerechnet), kein Verfahren aus einem echten Löser.
- **Synthetische, kleine Instanzen.** Bis 100 Ressourcen und 500 Dienste; die Läufe haben höchstens einige hundert Pivots.

## Verifikation

- `tests/test_algorithm.py`: jede Kombination aus Basisdarstellung und Regel gegen HiGHS (Zielwert, Zulässigkeit, Dualität), Unzulässig und Unbeschränkt, 80 kleine LPs mit gemischten Vorzeichen; **derselbe Pivotpfad wie das Tableau** für jede Regel und Basisdarstellung (über 90 Instanzen je Kombination, mit Phase 1, Steepest-Neustart und Neuinversionen); Ausleitung künstlicher Variablen und redundante Zeilen; explizite Inverse und Produktform gegen `numpy` nach jedem Update und jeder Neuinversion (auch mit permutierten Positionen); reduzierte Kosten gegen die Nullzeile des Tableaus in jeder Iteration; **Steepest-Edge-Gewichte gegen die neu berechneten Normen** in jeder Iteration; **Aufwand je Komponente von Hand nachgezählt** (Lehrbuchbeispiel: 84 / 35 / 102 / 54); partielle Preisgebung (Anteil 1 gleich voll, jeder Anteil erreicht das Optimum, sauberer Umlauf); Sonderfälle und Determinismus.
- `tests/test_scenario.py`, `test_evaluation.py`, `test_presets.py` (jede Zahl der Hilfetexte), `test_claims.py` (jede Zahl aus README und App über die echten `ev.*`-Funktionen), `test_app.py` (Streamlit-AppTest: Voreinstellung, jedes Preset, jeder Schritt für jede Instanz, jede Regel und Basisdarstellung, Pivot-Regler, Randwerte, Permalink-Grenzen, bedingte Regler, Kurven auf Abruf, Footer).
- Für die Prüfung genügt **pytest**; `scipy` dient nur als Gegenprobe (`requirements-dev.txt`), die App braucht nur numpy.

## Lokal starten

```bash
python -m venv venv && venv/Scripts/activate  # Windows; Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -W error::SyntaxWarning`.

## Literatur

- Dantzig, G. B., & Orchard-Hays, W. (1954). *The product form for the inverse in the simplex method.* Mathematical Tables and Other Aids to Computation 8(46), 64–67.
- Forrest, J. J. H., & Tomlin, J. A. (1972). *Updated triangular factors of the basis to maintain sparsity in the product form simplex method.* Mathematical Programming 2, 263–278 (nur genannt).
- Forrest, J. J., & Goldfarb, D. (1992). *Steepest-edge simplex algorithms for linear programming.* Mathematical Programming 57, 341–374.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
