# ChaosCrypto WP2 (CLI)

Dieses Repo enthält den minimalen, deterministischen ChaosCrypto-CLI-Prototyp (WP2-MVP).

Kernschritte:
- Token → Memory (deterministisches OpenSimplex-Noise-Feld)
- Seed aus Koordinate (Neighborhood3)
- Lorenz-System (Euler-Integration)
- Keystream aus x-Komponente (quantisiert)
- XOR für Ver- und Entschlüsselung

## Schnellstart (WSL, ohne Vorkenntnisse)

1) **In WSL ins Projektverzeichnis wechseln**  
   Beispiel: `cd /mnt/c/Users/WP2`

2) **Virtuelle Umgebung anlegen (sauber und lokal)**  
   ```bash
   python3 -m venv .venv --without-pip
   python3 -m pip install --break-system-packages --target .venv/lib/python3.12/site-packages pip
   .venv/bin/python -m pip install --break-system-packages -r requirements.txt
   .venv/bin/python -m pip install --break-system-packages -e .
   ```

3) **Funktionstest ausführen**  
   ```bash
   .venv/bin/python -m pytest -q
   ```  
   (Sollte “3 passed” melden.)

4) **Beispielnutzung**  
   - Profil anlegen (speichert Fingerprint, keine Klartext-Token):
     ```bash
     .venv/bin/python -m chaoscrypto.cli.app init --profile alice --token "secret" --size 128 --scale 0.1
     ```
   - Datei verschlüsseln:
     ```bash
     .venv/bin/python -m chaoscrypto.cli.app encrypt --profile alice --token "secret" --coord 12,34 --in msg.txt --out enc.json
     ```
   - Wieder entschlüsseln:
      ```bash
      .venv/bin/python -m chaoscrypto.cli.app decrypt --profile alice --token "secret" --in enc.json --out dec.txt
      ```
   - Optional steuerbar: `--dt`, `--warmup`, `--quant-k` bei `encrypt` setzen (werden in `enc.json` gespeichert und bei `decrypt` geprüft).
   - Seed-Strategien:
     - `neighborhood3` (Baseline)
     - `window_mean_3x3` (3x3-Fenstermittelwerte)
     Beispiel:
     ```bash
     .venv/bin/python -m chaoscrypto.cli.app encrypt --profile alice --token "secret" --coord 12,34 --in msg.txt --out enc.json --seed-strategy window_mean_3x3
     ```
   - Keystream (für Bench/Analyse, deterministisch):
     ```bash
     # SHA-256 Hash (Default-Output)
     .venv/bin/python -m chaoscrypto.cli.app keystream --profile alice --token "secret" --coord 12,34 --nbytes 1024
     # Rohbytes in Datei
     .venv/bin/python -m chaoscrypto.cli.app keystream --profile alice --token "secret" --coord 12,34 --nbytes 1024 --out ks.bin
     # Hex/Base64 nach stdout
     .venv/bin/python -m chaoscrypto.cli.app keystream --profile alice --token "secret" --coord 12,34 --nbytes 16 --hex
     ```
   - Benchmark (YAML-gesteuert, keine enc.json; rein im RAM):
     ```bash
     .venv/bin/python -m chaoscrypto.cli.app benchmark --config examples/bench.yaml --out results.csv --out-json results.json
     ```
     `--json` gibt eine kurze Summary nach stdout; `--jobs` parallelisiert Varianten (deterministische Sortierung).
   - Analyze (YAML-gesteuerte Keystream-Statistiken, RAM-only):
     ```bash
     .venv/bin/python -m chaoscrypto.cli.app analyze --config examples/analyze.yaml --out analysis.csv --out-json analysis.json
     ```
   - Report (Markdown + optional Plots/JSON) aus Benchmark/Analyze-Outputs:
     ```bash
     .venv/bin/python -m chaoscrypto.cli.app report --bench-csv results.csv --analysis-csv analysis.csv --out report.md --plots-dir plots
     ```

5) **Ergebnis prüfen**  
   ```bash
   cmp -s msg.txt dec.txt && echo "identisch"
   ```

## Hinweise für den Betrieb
- Alles läuft deterministisch: gleicher Token + gleiche Parameter → identische Fingerprints und Keystreams.
- Profile liegen in `~/.chaoscrypto/wp2/<profile>/` (WSL-Home).
- Das Noise-Feld wird bei Bedarf aus Token/Parametern deterministisch neu erzeugt (kein Cache).
- CLI-Parameter können bei `encrypt` (dt, warmup, quant_k) gesetzt werden; Lorenz-Parameter bleiben fix.
- Hilfsbefehle: `profile list`, `profile show`, `selftest` (Golden-Vector).
- `keystream` ist die Grundlage für Benchmark und Analyse und garantiert reproduzierbare Keystreams (identisch zur Encrypt-Pipeline).
- `benchmark` nutzt eine YAML-Matrix, generiert deterministische Keystreams im RAM, misst Zeiten/Hashes und schreibt CSV/JSON. Token wird nie im Klartext gespeichert (nur Fingerprint).
- `analyze` baut auf derselben Matrix-Logik auf und berechnet deterministische Keystream-Metriken (Bit-Balance, Histogram/Chi², Autocorr, Runs, Hamming-Weights); Outputs CSV/JSON, keine enc.json, Token bleibt verborgen.
- `report` fasst Benchmark/Analyze-Outputs in Markdown zusammen, kann PNG-Plots erzeugen und bleibt deterministisch (optional ohne Timestamp für Reprotests).
- Seed-Strategien: `neighborhood3` (Baseline), `window_mean_3x3` (3x3 Fenster-Mittelwerte). YAML-Matrix (`seed_strategy`) und CLI (`--seed-strategy`) wählen die Strategie; `enc.json` speichert die genutzte Strategie.
- Memory-Modelle: `opensimplex` (Default) und `perlin` (deterministisch, tokenbasierte Permutation). Auswahl via `--memory-type` bei `init`/CLI und `memory_type` in YAML-Matrix; CSV/JSON nutzen die Spalte `memory_type`.
- BA1-Experiment-Kit (WSL): `bash scripts/run_ba1.sh` (nach `pip install -e .` und optional venv-Aktivierung). Outputs landen in `out/ba1/` (bench/analyze/report/plots + run_meta.txt). Make-Target: `make ba1`, aufräumen: `make clean-out`.
