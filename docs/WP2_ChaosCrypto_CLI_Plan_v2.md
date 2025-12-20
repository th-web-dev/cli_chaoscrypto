# WP2 – ChaosCrypto CLI
## Projektkontext & Technische Planung

---

## 1. Kontext & Einordnung

### 1.1 Akademischer Kontext
Dieses Projekt ist Teil des Wahlfachprojekts 2 (WP2) und baut konzeptionell auf dem Wahlfachprojekt 1 (WP1) auf. Während WP1 einen webbasierten Prototypen zur deterministisch-synchronen Verschlüsselung mit chaotischen Systemen realisiert hat, verfolgt WP2 das Ziel, das Konzept **strukturiert, reproduzierbar und vergleichbar** neu aufzusetzen.

WP2 dient dabei als:
- technische Konsolidierung des ChaosCrypto-Ansatzes
- Experimentier- und Benchmark-Plattform
- Grundlage für die wissenschaftliche Evaluation in BA1

### 1.2 Motivation
Der Web-GUI-Ansatz aus WP1 erwies sich als komplex und schwer wartbar. Für WP2 wird daher bewusst auf eine grafische Oberfläche verzichtet und stattdessen ein **CLI-basiertes Tool** entwickelt, das:
- deterministisch reproduzierbare Experimente erlaubt
- klare Schnittstellen zwischen Modulen erzwingt
- automatisierte Benchmarks und Analysen ermöglicht

### 1.3 Übergeordnetes Ziel
Ziel ist es, ein **modulares CLI-Toolkit** zu entwickeln, das verschiedene Varianten chaotischer, deterministisch-synchroner Schlüsselstromerzeugung implementiert und vergleichbar macht.

Der Fokus liegt nicht auf Produktreife, sondern auf:
- Nachvollziehbarkeit
- Vergleichbarkeit
- Erweiterbarkeit

---

## 2. Technische Zielsetzung

### 2.1 Funktionale Ziele
- deterministische Erzeugung eines gemeinsamen Gedächtnismodells (Memory Model)
- synchrone Schlüsselstromerzeugung ohne expliziten Schlüsselaustausch
- Ver- und Entschlüsselung per XOR
- Vergleich mehrerer Varianten hinsichtlich Performance und Eigenschaften

### 2.2 Nicht‑funktionale Ziele
- deterministische Reproduzierbarkeit über Plattformen hinweg
- klare Modultrennung (kein impliziter State)
- vollständige Automatisierbarkeit (CLI only)
- einfache Erweiterbarkeit für BA1 / BA2

---

## 3. Architekturübersicht

### 3.1 High-Level Architektur

Das System ist strikt schichten- und modulbasiert aufgebaut:

1. CLI Layer
2. Orchestrations-/Service Layer
3. Kryptographischer Core
4. Analyse- & Benchmark-Layer
5. Persistenz & Export

```
┌────────────┐
│    CLI     │
└─────┬──────┘
      │
┌─────▼──────┐
│ Orchestr.  │
└─────┬──────┘
      │
┌─────▼────────────────────────┐
│   Crypto Core (modular)       │
│  - Memory Models              │
│  - Seed Derivation             │
│  - Chaotic Systems             │
│  - Keystream Generators        │
└─────┬────────────────────────┘
      │
┌─────▼─────────┐
│ Analysis/Bench│
└─────┬─────────┘
      │
┌─────▼─────────┐
│ Export (CSV)  │
└───────────────┘
```

### 3.2 Trennung der Verantwortlichkeiten
- CLI: Ein- und Ausgabe, Argument Parsing
- Orchestrierung: Ablaufsteuerung, Parameterkombinationen
- Core: deterministische Berechnung ohne Seiteneffekte
- Analyse: Messung & Auswertung
- Persistenz: reproduzierbare Speicherung von Artefakten

---

## 4. Technologieentscheidungen

### 4.1 Programmiersprache
**Python** wird für WP2 gewählt, da:
- schnelle Iteration möglich ist
- NumPy effiziente numerische Berechnungen erlaubt
- Benchmarking & CSV/JSON-Export trivial sind
- Fokus auf Methodik statt Low-Level-Optimierung liegt

(Optimierungen / Rust-Port sind explizit BA2-Thema.)

### 4.2 Wichtige Libraries
- NumPy – numerische Berechnungen
- hashlib – Hashfunktionen (SHA‑256)
- argparse / typer – CLI
- csv / json – Exportformate
- time / timeit – Performance-Messungen

### 4.3 Determinismus-Prinzipien
- keine Zufallsfunktionen ohne expliziten Seed
- feste Floating-Point-Operationen
- klare Normalisierungsschritte
- definierte Warm‑Up‑Phasen

---

## 5. Modulstruktur (geplant)

```
chaoscrypto_wp2/
├── cli/
│   ├── main.py
│   └── commands.py
├── core/
│   ├── memory/
│   │   ├── base.py
│   │   └── opensimplex.py
│   ├── seed/
│   │   ├── base.py
│   │   └── coordinate_seed.py
│   ├── chaos/
│   │   ├── base.py
│   │   └── lorenz.py
│   ├── keystream/
│   │   └── xor_stream.py
│   └── crypto.py
├── analysis/
│   ├── entropy.py
│   ├── autocorrelation.py
│   └── sensitivity.py
├── bench/
│   ├── runner.py
│   └── configs.py
├── export/
│   └── csv_export.py
├── tests/
│   └── test_vectors.py
└── README.md
```

---

## 6. Zentrale Konzepte

### 6.1 Memory Model
- deterministisch aus Token erzeugt
- 2D Noise-Feld (z. B. OpenSimplex)
- dient als gemeinsamer Gedächtnisraum

### 6.2 Seed-Ableitung
- Koordinaten (x,y) als öffentlicher Trigger
- Extraktion mehrerer Werte aus Memory Model
- Normalisierung auf gültige Startwerte

### 6.3 Chaotisches System
- Lorenz-Attraktor als Basissystem
- diskrete Iteration
- Warm‑Up‑Phase zur Entkopplung vom Initialzustand

### 6.4 Schlüsselstrom
- Ableitung von Bits aus Trajektorienwerten
- definierte Sampling-Strategie
- deterministisch reproduzierbar

---

## 7. Grober Programmablauf (End-to-End)

Dieser Ablauf beschreibt den *minimalen* Kernpfad (MVP) des WP2-CLI-Tools, damit die Core-Funktionalität früh stabil wird.

### 7.1 Startbedingungen
- Das Tool arbeitet **profilbasiert** (z. B. `alice`, `bob`), um Artefakte eindeutig zu speichern.
- Alle Berechnungen sind **deterministisch**: gleiche Inputs → gleiche Outputs.

### 7.2 Ablauf A: Setup / Initialisierung
1. **CLI-Call**: `chaoscrypto init --profile alice --token <TOKEN> --memory opensimplex --size 128 --scale 0.1`
2. **Token-Normalisierung**
   - Token als Bytes interpretieren (UTF-8) und via SHA-256 auf feste Länge bringen.
3. **Memory Model erzeugen**
   - Noise-Feld deterministisch aus Token + Parametern generieren.
4. **Validierungs-Fingerprint berechnen**
   - z. B. SHA-256 über definierte Teilmenge des Feldes (erste Zeile oder gesamtes Feld) → `field_fingerprint`.
5. **Persistieren**
   - Speichere: `memory_meta.json` (Parameter, fingerprint, version) + optional `field.bin` (wenn Field gecached wird).

**Output (Setup):**
- `profile ready` + `field_fingerprint` + verwendete Parameter

### 7.3 Ablauf B: Verschlüsselung (Core)
1. **CLI-Call**: `chaoscrypto encrypt --profile alice --cipher lorenz --coord 12,34 --in msg.txt --out enc.json`
2. **Profil laden**
   - Lade `memory_meta.json` und ggf. das gecachte Feld.
   - Falls kein Cache: Feld deterministisch neu generieren (aus Token/Parametern).
3. **Seed / Initialwerte ableiten**
   - Extrahiere Werte aus dem Feld um `(x,y)` (z. B. 3 benachbarte Zellen).
   - Mappe/normalisiere auf `(x0,y0,z0)` in definierte Bereiche.
4. **Chaossystem initialisieren**
   - Lorenz-Parameter (σ, ρ, β) fest oder konfigurierbar.
   - Setze Startzustand `(x0,y0,z0)`.
5. **Warm-up Iterationen**
   - Iteriere N Schritte ohne Keybit-Entnahme (Determinismus + Stabilität).
6. **Keystream generieren**
   - Generiere exakt `len(plaintext_bytes)` Bytes.
7. **XOR anwenden**
   - `ciphertext = plaintext XOR keystream`
8. **Metadaten erstellen**
   - `version`, `cipher`, `coord`, `warmup`, `iters`, `sampling`, `field_fingerprint`, `ciphertext_encoding`.
9. **Output schreiben**
   - Speichere `enc.json` (Metadaten + ciphertext in Base64/Hex).

**Output (Encrypt):**
- `enc.json` mit ciphertext + vollständigen Parametern zur Reproduktion

### 7.4 Ablauf C: Entschlüsselung (Core)
1. **CLI-Call**: `chaoscrypto decrypt --profile alice --in enc.json --out dec.txt`
2. **Profil + Memory laden**
3. **Metadaten aus `enc.json` lesen**
4. **Seed/Initialwerte identisch ableiten**
5. **Keystream identisch regenerieren**
6. **XOR anwenden → Plaintext**
7. **Optional: Validierungschecks**
   - z. B. plaintext hash / expected markers (nur wenn definiert).

**Output (Decrypt):**
- `dec.txt` (muss byte-ident mit Original sein)

### 7.5 Ablauf D: Benchmark (Vergleichbarkeit)
1. **CLI-Call**: `chaoscrypto benchmark --config bench.yaml --out results.csv`
2. Lade Test-Inputs (fixe Token, Koordinaten, Nachrichtenlängen)
3. Iteriere Variantenmatrix:
   - Memory params (size/scale)
   - Seed derivation strategy
   - Warmup/iters/sampling
4. Messe:
   - runtime encrypt/decrypt
   - throughput (bytes/s)
   - memory usage (approx.)
5. Export `results.csv` + optional `results.json`

### 7.6 Ablauf E: Analyse (Basis)
1. **CLI-Call**: `chaoscrypto analyze --profile alice --cipher lorenz --coord 12,34 --nbytes 1048576 --out analysis.json`
2. Generiere Keystream
3. Berechne Metriken (Basis):
   - Bit-Balance
   - Autokorrelation (lag 1..k)
   - Runs (grob)
4. Export `analysis.json`

---

## 8. Implementierungsschritte (Roadmap)

1. Projekt-Scaffold & CLI-Grundstruktur
2. Memory Model (OpenSimplex, deterministisch)
3. Seed-Ableitung aus Koordinaten
4. Lorenz-System + Keystream-Generator
5. Encrypt/Decrypt Pipeline
6. Golden Test Vectors
7. Benchmark Runner
8. Analyse-Module (Basisstatistiken)
9. CSV/JSON Export
10. Dokumentation

---

## 8. Abgrenzung zu BA1 & BA2

### WP2
- Implementierung
- Vergleichbarkeit
- Basisanalysen

### BA1
- wissenschaftliche Evaluation
- Randomness-Tests
- Sicherheitsanalyse

### BA2
- Protokollhärtung
- PQC-/Hybridansätze
- Performance-Optimierung (z. B. Rust)

---

## 9. Vergleichsdimensionen & Evaluationsrahmen

Dieser Abschnitt definiert **verbindlich**, welche Aspekte in WP2 verglichen werden. Er dient als Scope-Lock, um spätere Ausuferung zu vermeiden.

---

## 9.1 Vergleichsmodell (Pipeline-Sicht)

Das ChaosCrypto-System wird als modulare Pipeline verstanden. Jede austauschbare Stufe ist potenziell vergleichbar:

```
Token
  ↓
Memory Model
  ↓
Seed-/Init-Ableitung
  ↓
Chaotisches System
  ↓
Sampling / Keystream
  ↓
XOR → Ciphertext
```

WP2 fokussiert sich auf **technisch-experimentelle Vergleiche**, nicht auf formale Beweise.

---

## 9.2 Memory Models (Gedächtnisraum)

### Zweck
Das Memory Model stellt einen deterministischen, tokenbasierten Gedächtnisraum dar, aus dem Initialwerte abgeleitet werden.

### Vergleichsvarianten
- **OpenSimplex Noise (2D)** – Baseline (Pflicht)
- **Perlin Noise (2D)** – Vergleichsmodell
- **Value Noise (2D)** – optional, falls Aufwand vertretbar

### Vergleichsparameter
- Feldgröße: 64×64, 128×128
- Scale/Frequency: niedrig vs. hoch
- Wertebereich: normalisiert vs. quantisiert

### Vergleichsmetriken
- Wertverteilung (Histogramm)
- lokale Korrelation benachbarter Zellen
- Sensitivität bei Token-Bitflip

---

## 9.3 Seed- / Initialwert-Ableitung

### Zweck
Abbildung von `(Memory Model, Koordinate)` auf die Startwerte des chaotischen Systems.

### Vergleichsstrategien
- **Single-Cell**: direkte Nutzung einer Zelle
- **Neighborhood (3 Zellen)**: Mittelwert mehrerer Nachbarzellen
- **Window (NxN)**: Aggregation eines kleinen Fensters

### Aggregationsmethoden
- arithmetischer Mittelwert
- gewichteter Mittelwert
- Hash-basierte Aggregation

### Vergleichsmetriken
- Sensitivität bei Koordinatenänderung
- Kollisionsnähe (experimentell)
- numerische Stabilität

---

## 9.4 Chaotische Systeme

### Zweck
Erzeugung eines pseudozufälligen Schlüsselstroms aus einem dynamischen System.

### Vergleichsstrategie in WP2
- **Lorenz-Attraktor** als einziges voll implementiertes System
- Architektur erlaubt Plug-in weiterer Systeme (z. B. Rössler, Logistic Map)

### Vergleichsparameter (Lorenz)
- feste Parameter (σ, ρ, β)
- alternative Parametersets (optional)

### Vergleichsmetriken
- Sensitivität gegenüber Initialwerten
- numerische Stabilität
- Iterationen pro Sekunde

---

## 9.5 Sampling & Keystream-Extraktion

### Zweck
Transformation kontinuierlicher Zustandswerte in diskrete Bits/Bytes.

### Vergleichsstrategien
- Vorzeichen-basierte Extraktion (`sign(x)`)
- Quantisierung (`floor(|x| * k) mod 256`)
- Kombination mehrerer Zustände (`x ⊕ y ⊕ z`)

### Parameter
- Warm-up-Länge (z. B. 100, 1000 Iterationen)
- Downsampling-Faktor

### Vergleichsmetriken
- Bit-Balance (0/1-Verhältnis)
- Autokorrelation
- Runs-Test (Basis)
- Durchsatz (Bytes/s)

---

## 9.6 Betriebsmodi & Nutzungskonzepte

### Vergleichsvarianten
- statischer Token
- Token-Rotation (Hash-Kette)
- Koordinaten-Reuse vs. Tracking

### Vergleichsmetriken
- Wiederverwendungsrisiken
- zusätzlicher Rechenaufwand
- Komplexität vs. Sicherheitsgewinn

---

## 9.7 Performance & Reproduzierbarkeit

### Vergleichsaspekte
- Verschlüsselungs- / Entschlüsselungszeit
- Keystream-Generierungsrate
- Speicherverbrauch
- deterministische Reproduzierbarkeit über Runs hinweg

### Validierung
- Golden Test Vectors
- Hash-Vergleiche der Keystreams

---

## 9.8 Abgrenzung zu BA1 und BA2

### WP2
- modulare Implementierung
- vergleichbare Varianten
- Basisstatistiken & Benchmarks

### BA1
- tiefergehende Randomness-Tests
- Sicherheitsdiskussion
- Vergleich mit Standard-Ciphers

### BA2
- Protokollhärtung
- quantensichere / hybride Ansätze
- Performance-Optimierung (z. B. Rust)

---

## 10. Implementierungsleitfaden (WSL-first) + Codex-Anweisungen

Dieser Abschnitt beschreibt **konkret** wie das WP2-Projekt in PyCharm neu aufgesetzt, strukturiert und Schritt-für-Schritt implementiert wird. Ziel ist, den Core (init → encrypt → decrypt) schnell stabil zu bekommen und danach kontrolliert zu erweitern.

---

## 10.1 Entwicklung unter Windows: WSL vs. PowerShell

**Empfehlung:** Implementierung **WSL-first**.

Begründung:
- reproduzierbares Linux-Tooling (Python, venv, Make, pytest)
- weniger Unterschiede zu späteren Deployment-/CI-Umgebungen
- einheitliche Pfad- und Encoding-Welt (wichtig für deterministische Tests)

**Wichtig:** Es ist *nicht zwingend*, aber **für WP2 deutlich angenehmer**. PowerShell wäre möglich, erhöht aber oft den Overhead (Pfad-/Encoding-/Shell-Quoting-Differenzen).

---

## 10.2 Projekt-Setup in PyCharm (WSL)

### 10.2.1 WSL vorbereiten
- WSL2 aktivieren, Ubuntu installieren
- in WSL:
  - `sudo apt update && sudo apt install -y python3 python3-venv python3-pip make git`

### 10.2.2 Projekt anlegen
1. Neues Projekt in PyCharm erstellen.
2. Interpreter: **WSL / Ubuntu / Python 3.x** auswählen.
3. Projektpfad: bevorzugt im WSL-Dateisystem, z. B.:
   - `/home/<user>/projects/chaoscrypto-wp2`
   (nicht unter `C:\...`, um I/O und Path-Mappings zu vermeiden)
4. Die generierte Canvas-Datei (dieses Dokument) in den Content Root legen.

### 10.2.3 Virtual Environment
- im Projektroot:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `python -m pip install --upgrade pip`

### 10.2.4 Abhängigkeiten festlegen
- `requirements.txt` (minimal, WP2):
  - `numpy`
  - `typer` (oder `argparse`; Empfehlung: `typer`)
  - `rich` (optional für schöne CLI-Ausgaben)
  - `pytest`

Install:
- `pip install -r requirements.txt`

---

## 10.3 Repository-Struktur (verbindlich)

Diese Struktur ist **bewusst streng**: Core ist ohne CLI testbar.

```
chaoscrypto_wp2/
├── docs/
│   └── WP2_ChaosCrypto_CLI_Plan.md   (Canvas-Export / Copy)
├── src/
│   └── chaoscrypto/
│       ├── __init__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   └── app.py
│       ├── orchestrator/
│       │   ├── __init__.py
│       │   └── pipeline.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── constants.py
│       │   ├── memory/
│       │   │   ├── __init__.py
│       │   │   ├── base.py
│       │   │   └── opensimplex_like.py
│       │   ├── seed/
│       │   │   ├── __init__.py
│       │   │   ├── base.py
│       │   │   └── neighborhood3.py
│       │   ├── chaos/
│       │   │   ├── __init__.py
│       │   │   ├── base.py
│       │   │   └── lorenz.py
│       │   ├── sampling/
│       │   │   ├── __init__.py
│       │   │   ├── base.py
│       │   │   └── quantize_byte.py
│       │   └── crypto/
│       │       ├── __init__.py
│       │       └── xor.py
│       ├── io/
│       │   ├── __init__.py
│       │   ├── profiles.py
│       │   └── formats.py
│       ├── bench/
│       │   ├── __init__.py
│       │   └── runner.py
│       └── analysis/
│           ├── __init__.py
│           └── basic_stats.py
├── tests/
│   ├── test_vectors/
│   ├── test_init.py
│   ├── test_encrypt_decrypt.py
│   └── test_determinism.py
├── pyproject.toml  (oder setup.cfg)
├── requirements.txt
├── Makefile
└── README.md
```

---

## 10.4 Daten- und Artefaktmanagement (Profiles)

### 10.4.1 Profile-Verzeichnis
- Standardpfad unter WSL:
  - `~/.chaoscrypto/wp2/<profile>/`

### 10.4.2 Persistierte Dateien
- `profile.json`
  - memory_type, size, scale, version
  - field_fingerprint
  - optional: token_fingerprint (nie Klartext-Token speichern!)
- optional Cache:
  - `field.npy` (nur wenn explizit aktiviert)

### 10.4.3 Sicherheitsregel
- Token niemals im Klartext speichern.
- Nur Hash/Fingerprint speichern.

---

## 10.5 Output-Contract: enc.json (Minimalversion)

**Ziel:** Die Entschlüsselung muss ohne Zusatzwissen (außer Profile/Token) reproduzierbar sein.

Minimalfelder:
- `version`: z. B. `1`
- `cipher`: `lorenz`
- `memory`: `{type,size,scale}`
- `seed_strategy`: `neighborhood3`
- `sampling`: `{type, warmup, dt, quant_k}`
- `coord`: `{x,y}`
- `field_fingerprint`: `<hex>`
- `ciphertext_encoding`: `base64`
- `ciphertext`: `<base64>`

Optional (später):
- `plaintext_hash`: SHA-256 des plaintext (für Tests, nicht für echten Betrieb)
- `rotation`: Angaben zum Token-Rotation-Modus

---

## 10.6 Implementierungsschritte (extrem konkret)

### Phase 0: Arbeitsfähigkeit herstellen (30–60 Min)
1. Repo-Struktur anlegen
2. `requirements.txt` + `pytest` lauffähig
3. `Makefile`:
   - `make test` → `pytest -q`
   - `make fmt` (optional)

### Phase 1: Core-MVP (init → encrypt → decrypt)
**Ziel:** Ein Golden Test Vector muss stabil durchlaufen.

1) Memory Model (Baseline)
- Implementiere `MemoryModel.generate(token_bytes, size, scale) -> np.ndarray`
- Erzeuge `field_fingerprint = sha256(field_bytes)`
- Determinismus-Test: gleiche Inputs → gleicher fingerprint

2) Seed/Init Ableitung
- Implementiere `derive_init(field, coord) -> (x0,y0,z0)` nach WP1-Logik:
  - `x0 = field[x,y]`
  - `y0 = field[x+1,y]`
  - `z0 = field[x,y+1]`

3) Lorenz-System
- Implementiere diskrete Integration:
  - Parameter: σ=10, ρ=28, β=8/3 (fix)
  - Integrator: Euler (für WP2 ok, solange deterministisch)
  - Step `dt` fix (z. B. 0.01)
  - Warm-up: N Schritte

4) Sampling → Bytes
- Implementiere `keystream_bytes(n)`:
  - aus x-Komponente, deterministisch normalisiert/quantisiert
  - z. B. `byte = floor(abs(x) * k) % 256`

5) XOR
- Implementiere `xor_bytes(data, keystream)`

6) CLI Commands (nur MVP)
- `init` erstellt Profile
- `encrypt` liest plaintext und schreibt enc.json
- `decrypt` liest enc.json und schreibt plaintext

7) Golden Test Vector
- Fixiere Test:
  - token = `"test-token"`
  - coord = (12,34)
  - msg = `"hello"`
- Erwartung:
  - decrypt(encrypt(msg)) == msg
  - fingerprints stabil

### Phase 2: Vergleichbarkeit + Bench
- `benchmark`-Command
- Variantenmatrix für:
  - size (64/128)
  - warmup (100/1000)
  - quant_k (z. B. 1e3/1e5)
- Export: CSV

### Phase 3: Analyse (Basis)
- bit-balance
- autocorrelation (lags 1..32)
- runs (grob)

---

## 10.7 Codex-Anweisungen (wie du Codex effektiv steuerst)

### 10.7.1 Grundregeln (sehr wichtig)
- Codex arbeitet **nur innerhalb** der oben definierten Struktur.
- Jede Änderung muss:
  - Tests grün halten (`make test`)
  - Determinismus nicht brechen
- Keine „kreativen“ Abkürzungen (kein globaler State, keine versteckten Caches)

### 10.7.2 Prompt-Vorlagen (copy/paste)

**Prompt A – Modul erstellen (sicher, klein):**
- „Erstelle in `src/chaoscrypto/core/seed/neighborhood3.py` eine Funktion `derive_init(field: np.ndarray, coord: tuple[int,int]) -> tuple[float,float,float]` gemäß: x0=field[x,y], y0=field[x+1,y], z0=field[x,y+1] mit Modulo-Indexing. Füge Typen, Docstring, und Unit-Test in `tests/test_seed.py` hinzu.“

**Prompt B – Determinismus-Test:**
- „Implementiere `field_fingerprint(field: np.ndarray) -> str` als SHA-256 über `field.tobytes(order='C')` und schreibe `tests/test_determinism.py`, das bei gleichem Token/Parametern exakt gleichen Fingerprint erwartet.“

**Prompt C – CLI Command (typer):**
- „Erweitere `src/chaoscrypto/cli/app.py` um Command `init` (profile, token, size, scale). Speichere Profile nach `~/.chaoscrypto/wp2/<profile>/profile.json` ohne Klartext-Token (nur token_fingerprint). Schreibe Tests mit temporärem HOME (pytest monkeypatch).“

**Prompt D – End-to-End Vector:**
- „Erstelle `tests/test_encrypt_decrypt.py` mit einem Golden Test Vector. Verwende token `test-token`, coord (12,34), message `hello`. Prüfe decrypt(encrypt(message)) == message. Schreibe außerdem enc.json nach Schema in docs.“

### 10.7.3 Code-Review Checkliste für Codex-Ausgaben
- Sind alle Inputs explizit (token bytes, coord, dt, warmup)?
- Gibt es irgendwo Randomness?
- Sind Arrays/Bytes eindeutig (order='C')?
- Werden Pfade WSL-konform verwendet?
- Sind Unit-Tests vorhanden?

---

## 10.8 Lessons Learned aus WP1 (Review)

Diese Punkte werden aus WP1 übernommen und bewusst adressiert:

- Synchronität ist extrem fragil: kleinste Abweichungen in Noise-Feld oder Floating-Point führen zu völlig anderem Keystream.
- In WP1 war die zentrale Maßnahme die Vereinheitlichung der Noise-Erzeugung (Rust → PyO3/WASM), um plattformübergreifend identische Felder zu erhalten.
- Token-Rotation wurde als Forward-Secrecy-Mechanismus umgesetzt: `Token_new = Hash(Token_old || Hash(message))`.
- Validierung erfolgte über Hash-Vergleiche (Noise-Feld / Keystream).
- Koordinaten waren (noch) fix/hardcoded, später dynamisch geplant.

Konsequenz für WP2:
- deterministische Tests + Golden Vectors sind Pflicht
- klare Versionierung im enc.json
- Rotation ist optionaler Modus (nicht MVP)
- Fokus auf Vergleichbarkeit statt GUI-Komplexität

---

## 11. MVP Scope Lock (verbindlich für WP2-Start)

Dieser Abschnitt definiert **explizit und verbindlich**, was zum **Minimal Viable Prototype (MVP)** von WP2 gehört – und was **nicht**. Er dient als Schutz vor Scope Creep, insbesondere bei Nutzung von Codex.

---

## 11.1 Ziel des MVP

Der MVP hat **ein einziges Ziel**:

> *Nachweis einer stabilen, deterministisch reproduzierbaren Pipeline von*  
> **Token → Memory → Seed → Chaossystem → Keystream → XOR → Decrypt = Original**

Alles, was diesem Ziel **nicht direkt dient**, ist **explizit ausgeschlossen**.

---

## 11.2 MVP-Komponenten (IN SCOPE)

### Memory Model
- **Typ:** OpenSimplex-ähnliches deterministisches 2D-Noise
- **Größe:** fix 128×128
- **Scale:** fix (z. B. 0.1)
- **Normalisierung:** float64, keine Quantisierung
- **Caching:** optional, aber standardmäßig **ausgeschaltet**

➡️ **Genau ein Memory Model** im MVP.

---

### Seed- / Initialwert-Ableitung
- **Strategie:** Neighborhood3
  - `(x0, y0, z0) = (field[x,y], field[x+1,y], field[x,y+1])`
- **Indexing:** modulo Feldgröße
- **Koordinaten:** ganzzahlig, öffentlich

➡️ **Genau eine Seed-Strategie** im MVP.

---

### Chaotisches System
- **System:** Lorenz-Attraktor
- **Parameter:**
  - σ = 10
  - ρ = 28
  - β = 8/3
- **Integrator:** expliziter Euler
- **Zeitschritt:** `dt = 0.01`
- **Warm-up:** fix 1 000 Iterationen

➡️ **Keine alternativen Chaossysteme im MVP.**

---

### Sampling / Keystream
- **Quelle:** x-Komponente
- **Strategie:** Quantisierung zu Byte
  - `byte = floor(abs(x) * 1e5) % 256`
- **Downsampling:** keiner

➡️ **Genau eine Sampling-Strategie** im MVP.

---

### Kryptographische Operation
- **Verfahren:** XOR
- **Daten:** byteweise
- **Encoding:** UTF-8 Input → Bytes → XOR → Base64 Output

---

### CLI Commands – Zielbild (final in WP2)

Dieser Abschnitt definiert die **angestrebten CLI-Kommandos am Ende von WP2**. Nicht alle Kommandos sind Teil des MVP, aber das Zielbild gibt die Richtung vor.

**Grundprinzip:**
- Kommandos sind **scriptbar** (stdin/stdout-fähig)
- Ausgaben sind wahlweise **menschenlesbar** oder **JSON** (`--json`)
- Fehler führen zu non-zero Exit Codes

#### A) Profil & Setup
- `init`
  - legt Profil an, speichert Memory-Parameter, berechnet `field_fingerprint`
- `profile show`
  - zeigt gespeicherte Profilparameter, Fingerprints, Version
- `profile list`
  - listet vorhandene Profile
- `profile delete`
  - löscht Profilartefakte (mit Sicherheitsabfrage)

#### B) Kernfunktionen (Crypto)
- `encrypt`
  - plaintext → `enc.json` (oder stdout)
- `decrypt`
  - `enc.json` → plaintext (oder stdout)
- `keystream`
  - generiert Keystream-Bytes für definierte Länge (Debug/Analyse)

#### C) Vergleichbarkeit & Messung
- `benchmark`
  - führt Variantenmatrix aus, exportiert CSV/JSON
- `analyze`
  - erzeugt Keystream und berechnet Basisstatistiken (bit-balance, autocorr, runs)

#### D) Hilfs-/Meta-Kommandos
- `version`
  - gibt Tool-Version + Build-Info aus
- `selftest`
  - führt Golden Test Vector(s) aus und meldet deterministische Integrität

---

### CLI-Kommandos (MVP)

| Command | Pflicht | Beschreibung |
|-------|--------|--------------|
| `init` | ✅ | Profil anlegen, Memory-Parameter speichern |
| `encrypt` | ✅ | Plaintext → enc.json |
| `decrypt` | ✅ | enc.json → Plaintext |

❌ Kein `benchmark`, kein `analyze` im MVP.

---

### Tests (MVP-Kriterium)

Der MVP gilt **nur dann als erreicht**, wenn alle folgenden Tests grün sind:

1. **Golden Test Vector**
   - token = `test-token`
   - coord = `(12,34)`
   - message = `"hello"`
   - `decrypt(encrypt(message)) == message`

2. **Determinismus-Test**
   - identischer Token + Parameter → identischer `field_fingerprint`

3. **Cross-Run-Test**
   - zwei getrennte Läufe → identisches Ergebnis

---

## 11.3 Explizit NICHT im MVP (OUT OF SCOPE)

Die folgenden Punkte dürfen **nicht** im MVP implementiert werden:

- weitere Memory-Modelle (Perlin, Value, …)
- alternative Seed-Strategien
- andere Chaossysteme (Rössler, Logistic, …)
- Token-Rotation
- Koordinaten-Tracking
- Performance-Benchmarks
- Randomness-Statistiken
- GUI oder TUI
- Parallelisierung oder Optimierung

➡️ **Alles hier gehört frühestens in Phase 2 nach MVP-Abnahme.**

---

## 11.4 Freigabekriterium für Phase 2

Phase 2 (Vergleichbarkeit) darf **erst beginnen**, wenn:

- alle MVP-Tests grün sind
- mindestens ein Golden Test Vector dokumentiert ist
- enc.json exakt dem definierten Output-Contract entspricht

---

## 12. Status, offene Punkte & nächste Schritte

Dieser Abschnitt wird laufend gepflegt und dient als gemeinsame Wahrheit über den Projektstand.

### 12.1 Status (erreicht)
- MVP-Pipeline steht: `init` / `encrypt` / `decrypt` über Typer-CLI, deterministisch (Profile + Feld-Fingerprint).
- Echtes OpenSimplex-Noise im Memory-Modul, Seed via Koordinate (Neighborhood3), Lorenz (Euler) mit festem `dt`/`warmup`/Parametern, Sampling via quantized x-Komponente, XOR.
- Artefakte/Fingerprints in `~/.chaoscrypto/wp2/<profile>/profile.json`; `enc.json` enthält alle Repro-Metadaten.
- Tests grün (`pytest -q`), Golden-Vector vorhanden; Requirements inkl. `opensimplex`.

### 12.2 Offen / Verbessern
- Parametrisierung: `dt`, `warmup`, `quant_k`, Lorenz-Parameter, Memory-Typ/Seed-/Sampling-Strategie sind aktuell hart codiert; sollen als CLI-Flags/Profil-Parameter konfigurierbar werden.
- Memory-Cache optional (derzeit wird Feld immer neu deterministisch erzeugt).
- Zusätzliche Commands aus Zielbild fehlen noch: `keystream`, `profile list/show/delete`, `benchmark`, `analyze`, `version`, `selftest`.
- Packaging: lokal via `pip install -e .`; kein Wheel/Release-Flow.
- Weitere Tests: mehr Golden Vectors, Edge-Cases (leere Datei, große Dateien, andere Koordinaten), Cross-run Determinism auf verschiedenen Maschinen.

### 12.3 Aktuelle Nutzung (WSL)

```bash
.venv/bin/python -m pip install --break-system-packages -r requirements.txt
.venv/bin/python -m pytest -q

.venv/bin/python -m chaoscrypto.cli.app init --profile alice --token "secret" --size 128 --scale 0.1
.venv/bin/python -m chaoscrypto.cli.app encrypt --profile alice --token "secret" --coord 12,34 --in msg.txt --out enc.json
.venv/bin/python -m chaoscrypto.cli.app decrypt --profile alice --token "secret" --in enc.json --out dec.txt
```

### 12.4 Empfohlene nächste Schritte (kontrolliert, ohne Scope-Creep)
1. **`selftest`** implementieren (führt Golden-Vector(s) aus, prüft Fingerprints, Exit-Code ≠ 0 bei Abweichung).
2. **`profile show/list`** (rein lesend) für schnelle Debugbarkeit.
3. **Parametrisierung minimal**: nur `dt`, `warmup`, `quant_k` als Flags + in `enc.json` (ohne Strategiewechsel).
4. **Edge-Case Tests**: leere Inputdatei, UTF-8/Bytes, sehr große Datei (Streaming optional später).
5. Danach erst: `keystream` → `benchmark` → `analyze`.

---

## 13. Erwartete Ergebnisse

- vollständig deterministischer Core
- minimaler, testbarer CLI-Workflow
- stabile Grundlage für Vergleichs- und Benchmark-Phase
- maximale Kontrolle über Systemkomplexität

---

*Motto für den MVP: „Erst korrekt, dann vergleichbar – nie umgekehrt.“*

