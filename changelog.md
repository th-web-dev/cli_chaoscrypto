# Changelog BA2

Dieses Dokument protokolliert abgeschlossene, fachlich relevante Implementierungen und wichtige technische Änderungen im Rahmen von BA2.

Jeder Eintrag enthält:
- das Datum der vollständigen Umsetzung
- die technische Änderung im Projekt
- die Relevanz für die Bachelorarbeit 2

## 2026-05-16

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
