# Changelog BA2

Dieses Dokument protokolliert abgeschlossene, fachlich relevante Implementierungen und wichtige technische Änderungen im Rahmen von BA2.

Jeder Eintrag enthält:
- das Datum der vollständigen Umsetzung
- die technische Änderung im Projekt
- die Relevanz für die Bachelorarbeit 2

## 2026-05-16

### Task 5.1: Usability-Instrumentierung mit `study-run`

Technisch:
- Ein neues Modul `src/chaoscrypto/analysis/usability.py` wurde ergänzt.
- Der neue CLI-Befehl `study-run` führt beliebige ChaosCrypto-Kommandos aus und protokolliert BA2-relevante Nutzungsmetriken in eine CSV-Logdatei.
- Pro Run werden automatisch erfasst:
- Command und Argumente (`command_argv_json`, Argumentanzahl, Flag-Anzahl)
- Laufzeit (`duration_s`)
- Status/Exit-Code (`success`/`fail`)
- optionale Artefakt-Hashes zur Reproduzierbarkeitsprüfung
- Anzahl vorheriger Fehlversuche vor einem erfolgreichen Run (`failed_attempts_before_success`)
- Bei erneutem Lauf mit gleichem Command und `repro_key` wird `repro_match_previous` gesetzt, wenn die Artefakt-Hashes identisch sind.
- Testabdeckung wurde mit `tests/test_usability_log.py` ergänzt.

BA-Relevanz:
- Damit ist der Usability-Pfeiler von BA2 quantitativ messbar statt nur qualitativ beschreibbar.
- Die erhobenen Metriken erlauben Aussagen über Bedienaufwand, Fehlerraten und Reproduzierbarkeit entlang realer Evaluationsläufe.
- Die Logs können direkt für Methodik- und Ergebniskapitel genutzt werden.

### Task 4.2: Methodik-Schärfung für `ba2-eval` (NIST-Interpretation)

Technisch:
- `src/chaoscrypto/analysis/ba2_eval.py` wurde erweitert um methodisch robustere NIST-Kennzahlen.
- Zusätzlich zur globalen Fail-Quote wird jetzt eine bereinigte Sicht ausgewiesen:
- Anzahl Core-Tests ohne die beiden bekannten Template-Tests (`non_overlapping_template_matching`, `overlapping_template_matching`)
- Anzahl Core-Tests mit mindestens einem Fail
- Neue Fail-Hotspot-Analyse wurde ergänzt (Top-Gruppen nach mittlerer Fail-Anzahl pro Variante) für:
- `memory_type`
- `seed_strategy`
- `quant_k`
- `warmup`
- `dt`
- Die Markdown-Ausgabe wurde erweitert, damit diese Befunde direkt in die BA2-Diskussion übertragbar sind.
- Tests für den BA2-Eval-Workflow bleiben grün.

BA-Relevanz:
- Die Auswertung reduziert das Risiko von Fehlinterpretationen, bei denen eine bekannte Testdomäne (Template-Tests) die Gesamtbewertung verzerrt.
- Parameter-Hotspots liefern eine direkte Brücke von Messdaten zu Forschungsfragen (welche Einstellungen systematisch problematisch sind).
- Dadurch wird die Ergebnisdiskussion in BA2 argumentativ klarer und wissenschaftlich belastbarer.

### Task 4.1: Konsolidierte BA2-Evaluationsausgabe (`ba2-eval`)

Technisch:
- Ein neues Modul `src/chaoscrypto/analysis/ba2_eval.py` wurde ergänzt.
- Der neue CLI-Befehl `ba2-eval` konsolidiert bestehende Ergebnisdateien:
- `nist_runs.csv`
- `nist_summary.csv`
- `avalanche.csv` bzw. `avalanche_v2.csv`
- `periodicity.csv`
- Als Ausgabe werden automatisch erzeugt:
- ein BA2-Readout in Markdown (`ba2_eval_summary.md`)
- eine kompakte KPI-Tabelle (`ba2_eval_summary.csv`)
- optional ein JSON-Export (`ba2_eval_summary.json`)
- Die Auswertung enthält zentrale Kennzahlen für:
- NIST (Fail-Share, durchschnittlich bestandene/fehlgeschlagene Tests, kritischste Tests)
- Avalanche (Mean/Min/Max Hamming-Ratio, Skip/Evaluate-Zählung, Variant-Ranking nach Nähe zu 0.5)
- Periodizität (Lag-Match-Ratio, Chunk-Hash-Repetitionen, erkannte Präfixperioden)
- Die Testabdeckung wurde mit `tests/test_ba2_eval.py` um einen End-to-End-CLI-Test erweitert.

BA-Relevanz:
- Dieser Schritt macht aus mehreren separaten Rohdateien eine direkte, zitierfähige BA2-Auswertung.
- Ergebnisse zu Usability (automatisierbare Auswertung), Performance/Security-Indikatoren und Methodenvergleich werden konsistent in einem Artefakt zusammengeführt.
- Die Trennung zwischen Rohmessung und konsolidierter Interpretation ist für Nachvollziehbarkeit und Reproduzierbarkeit der BA2-Argumentation zentral.

### Task 3.3: NIST-Batch-Workflow und BA2-Aggregation

Technisch:
- Ein neues Modul `src/chaoscrypto/analysis/nist_batch.py` wurde ergänzt.
- Der neue CLI-Befehl `nist-batch` läuft die vollständige Variantenmatrix aus der bestehenden Analyze-Konfiguration durch und führt pro Variantenlauf die NIST-Suite aus.
- Die Ausgabe ist in zwei BA2-orientierte CSVs getrennt:
- Run-Ebene: `nist_runs.csv` mit Parametern, Gesamtzählern und testweisen `status`/`p_value`/`skip_reason`.
- Test-Ebene: `nist_summary.csv` mit aggregierten Kennzahlen (`n_runs`, `n_pass`, `n_fail`, `n_skip`, `pass_rate`, `pvalue_mean`, `pvalue_std`, `pvalue_min`, `pvalue_max`).
- Optional kann ein kombinierter JSON-Export (`--out-json`) erzeugt werden.
- In `nist_validator.py` wurde eine explizite Mindestlängen-Guardrail pro Test ergänzt; bei zu kurzen Bitfolgen wird konsistent `status=skip`, `reason=insufficient_bits` und `required_min_bits` protokolliert.
- Die Testabdeckung wurde mit `tests/test_nist_batch.py` erweitert (Smoke-Test für Run-/Summary-Outputs und Verifikation der Skip-Reason-Logik).

BA-Relevanz:
- Die NIST-Auswertung ist damit nicht mehr nur als Einzelmessung vorhanden, sondern als reproduzierbarer Batch-Prozess für Variantenvergleiche.
- Die getrennte Run- und Testaggregation vereinfacht die direkte Übernahme in BA2-Tabellen und statistische Diskussionen.
- Die expliziten Skip-Gründe erhöhen die methodische Transparenz, da fehlende Mindestlängen nicht als "echte" Test-Fails fehlinterpretiert werden.

### NIST-Validierung und BA2-Analysepipeline

Technisch:
- Ein NIST-Validierungsmodul wurde in `src/chaoscrypto/analysis/nist_validator.py` ergänzt.
- Die Analysepipeline wurde so erweitert, dass ein generierter Keystream optional automatisch durch die integrierte NIST-Suite läuft.
- Die Ergebnisse werden in `analysis.csv` und `analysis.json` geschrieben, inklusive P-Values, Statusfeldern und aggregierten NIST-Summenwerten.
- Der Report wurde um eine eigene NIST-Zusammenfassung erweitert.
- BA2-Beispielkonfigurationen und Batch-Skripte wurden ergänzt, damit Mehrvariantenläufe reproduzierbar ausführbar sind.
- Die NIST-Implementierung wurde für größere Keystreams optimiert, insbesondere bei `linear_complexity`.
- Die Testabdeckung wurde für NIST, Analyse, Benchmark-Parallelisierung und Reporting erweitert.

BA-Relevanz:
- Diese Erweiterung schafft die Grundlage für eine systematische statistische Bewertung der Keystreams.
- Die Ergebnisse können direkt in die BA2-Auswertung übernommen werden, ohne manuelle Nachbearbeitung.
- Der Prototyp wird dadurch als Forschungsplattform deutlich belastbarer, weil statistische Auffälligkeiten dokumentierbar und zwischen Varianten vergleichbar sind.

### Repository-Bereinigung und Reproduzierbarkeit

Technisch:
- Eine `.gitignore` wurde eingeführt.
- Generierte Artefakte wie `.venv`, `__pycache__`, `*.pyc`, `out/`, `*.egg-info` und lokale Tool-Dateien wurden aus der Versionsverwaltung herausgenommen oder ignoriert.
- Die Python-Umgebung wurde auf den aktuellen Dependency-Stand gebracht, damit die neue NIST-Integration lokal wieder lauffähig ist.
- Der Arbeitsbaum wurde so bereinigt, dass nur noch echte Projektänderungen sichtbar bleiben.

BA-Relevanz:
- Diese Bereinigung verbessert die Nachvollziehbarkeit und Reproduzierbarkeit der Implementierung.
- Für BA2 ist das wichtig, weil Evaluationsläufe, Batch-Experimente und Reports nicht durch lokale Artefakte oder Repository-Rauschen verfälscht werden sollen.

### Task 2.1: Ressourcen-Split im Benchmark

Technisch:
- Der Benchmark wurde so erweitert, dass `t_seed_s` separat gemessen und exportiert wird.
- `t_keystream_s` misst nun nur noch die Keystream-Generierung nach der Seed-Ableitung.
- Der XOR-Pfad im Benchmark verwendet jetzt konsistent die zentrale `xor_bytes`-Implementierung.
- Die Benchmark-Logik für `field_regen_each_repeat=true` wurde korrigiert, damit das Feld nicht versehentlich doppelt erzeugt wird.
- Der Benchmark-Report zeigt jetzt zusätzlich eine `Phase timing overview` mit gemittelten Laufzeiten für Feldgenerierung, Seed-Ableitung, Keystream und XOR.
- Tests für Benchmark und Report wurden entsprechend erweitert.

BA-Relevanz:
- Diese Änderung ist die Grundlage für die Performance-Evaluation in BA2.
- Statt nur Gesamtdurchsatz zu betrachten, kann jetzt analysiert werden, welcher Verarbeitungsschritt den größten Anteil an der Laufzeit hat.
- Das ist besonders wichtig für spätere Vergleiche zwischen Parametervarianten, Plattformen und möglichen Referenzverfahren.

### Task 2.2: Plattform-Divergenz-Test

Technisch:
- Es wurde ein neues Modul `src/chaoscrypto/analysis/platform_divergence.py` ergänzt.
- Der neue CLI-Befehl `platform-check` erzeugt pro Variantenlauf einen Export mit Laufzeitumgebung, `field_fingerprint` und `keystream_sha256`.
- Der neue CLI-Befehl `platform-compare` vergleicht zwei Plattform-Exports automatisiert und protokolliert pro Variante, ob `field_fingerprint` und `keystream_sha256` übereinstimmen.
- Für den Plattform-Check wird die bestehende `analyze`-Konfigurationsstruktur wiederverwendet, damit keine zweite Matrixlogik gepflegt werden muss.
- Eine Beispielkonfiguration wurde in `examples/ba2_platform_check.yaml` ergänzt.
- Die Testabdeckung wurde um einen Smoke-Test für den Export und einen Divergenz-Test für absichtlich manipulierte Hashes erweitert.

BA-Relevanz:
- Diese Erweiterung beantwortet eine zentrale Forschungsfrage von BA2: ob identische Eingaben auf unterschiedlichen Laufzeitumgebungen tatsächlich identische Keystreams erzeugen.
- Damit kann systematisch geprüft werden, ob Floating-Point- oder Plattformunterschiede die behauptete Deterministik des Systems beeinträchtigen.
- Die Ergebnisse lassen sich direkt als Nachweis in die BA2-Diskussion zu Reproduzierbarkeit, Robustheit und Grenzen des Prototyps übernehmen.

### Task 3.2: Sensitivity / Avalanche-Effekt

Technisch:
- Es wurde ein neues Modul `src/chaoscrypto/analysis/avalanche.py` ergänzt.
- Der neue CLI-Befehl `avalanche` erzeugt systematisch Perturbationen durch Single-Bit-Flips im Token sowie in `coord_x` und `coord_y`.
- Pro Perturbation werden Basis- und Vergleichskeystream berechnet und über die Hamming-Distanz auf Bitebene ausgewertet.
- Die Ergebnisse werden in CSV/JSON exportiert, inklusive `hamming_distance_bits` und `hamming_distance_ratio`.
- Eine BA2-Beispielkonfiguration wurde in `examples/ba2_avalanche.yaml` ergänzt.
- Die Testabdeckung wurde um CLI-Smoke- und Validierungstests für den neuen Avalanche-Workflow erweitert.

BA-Relevanz:
- Diese Erweiterung liefert eine direkte empirische Messung der Sensitivität gegenüber minimalen Eingabeänderungen.
- Damit kann für BA2 nachvollziehbar geprüft werden, wie nah das System am erwarteten Avalanche-Verhalten (ca. 50% Bitänderung) liegt.
- Die Ausgaben sind direkt für Resultatkapitel und Diskussion nutzbar, weil sie pro Variantenlauf reproduzierbar und vergleichbar sind.

### Task 3.1: Finite Precision & Periodizitäts-Analyse

Technisch:
- Es wurde ein neues Modul `src/chaoscrypto/analysis/periodicity.py` ergänzt.
- Der neue CLI-Befehl `periodicity` erzeugt pro Variante einen Keystream und berechnet mehrere Periodizitätsindikatoren:
- Chunk-Hash-Wiederholungen (inklusive `unique_chunk_hashes`, `repeated_chunk_hash_count`, `max_chunk_hash_frequency`)
- Byte-Lag-Match-Ratio über konfigurierbaren Abstand (`lag_match_ratio`)
- Einfache exakte Präfix-Periodensuche bis zu einer konfigurierbaren Obergrenze (`detected_prefix_period_bytes`)
- Die Ergebnisse werden in CSV/JSON exportiert und sind matrix-kompatibel mit dem bestehenden BA2-Workflow.
- Eine BA2-Beispielkonfiguration wurde in `examples/ba2_periodicity.yaml` ergänzt.
- Die Testabdeckung wurde mit CLI-Smoke- und Parameter-Validierungstests erweitert.

BA-Relevanz:
- Diese Erweiterung operationalisiert die BA2-Frage nach endlicher Präzision und möglicher Zyklusbildung über lange Bitfolgen.
- Statt rein qualitativer Aussagen liefert sie reproduzierbare, quantifizierbare Hinweise auf Wiederholungsmuster.
- Die Metriken sind direkt für Resultatdarstellung und Sicherheitsdiskussion nutzbar und ergänzen NIST- und Avalanche-Befunde methodisch sinnvoll.

### Task 3.2: Methodik-Fix für Koordinaten-Bitflips

Technisch:
- Der Avalanche-Workflow wurde um eine Invarianz-Prüfung für Koordinaten-Bitflips ergänzt.
- Bitflips, die modulo Feldgröße keine effektive Koordinatenänderung verursachen, werden jetzt als `perturbation_skipped=True` markiert.
- Für übersprungene Fälle werden Grund (`perturbation_skip_reason`) und Basiswerte protokolliert, statt ein künstliches `hamming_distance_ratio=0.0` zu erzeugen.
- CSV/JSON-Ausgabe und CLI-JSON-Summary wurden angepasst (`rows_skipped`, `rows_evaluated`).
- Die Testabdeckung wurde um einen gezielten Skip-Edge-Case erweitert.

BA-Relevanz:
- Dadurch wird der Avalanche-Befund methodisch korrekt und nicht durch modulo-invariante Artefakte verzerrt.
- Die Auswertung trennt jetzt sauber zwischen echten Sensitivitätsmessungen und formal erzeugten, aber inhaltlich nicht wirksamen Perturbationen.
- Das erhöht die wissenschaftliche Aussagekraft der BA2-Sicherheitsanalyse deutlich.
