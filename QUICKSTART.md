# ChaosCrypto CLI Quickstart (Windows-first)

This is a short, copy-paste guide for a professor demo. Full details are in `README.md`.

Prereqs: Python 3.10+, Git, PowerShell.

## 1) Setup (Windows / PowerShell)

```powershell
git clone <REPO_URL>
cd chaoscrypto-wp2
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Verify:
```powershell
.\.venv\Scripts\python.exe -m chaoscrypto.cli.app --help
```

## 2) 5-minute demo

```powershell
.\.venv\Scripts\python.exe -m chaoscrypto.cli.app init --profile alice --token "secret" --memory-type opensimplex --size 128 --scale 0.1
"hello from ChaosCrypto" | Set-Content -Encoding utf8 msg.txt
.\.venv\Scripts\python.exe -m chaoscrypto.cli.app keystream --profile alice --token "secret" --coord 12,34 --nbytes 32
.\.venv\Scripts\python.exe -m chaoscrypto.cli.app encrypt --profile alice --token "secret" --coord 12,34 --in msg.txt --out enc.json
.\.venv\Scripts\python.exe -m chaoscrypto.cli.app decrypt --profile alice --token "secret" --in enc.json --out dec.txt
```

Outputs:
- `enc.json` and `dec.txt` in the repo folder.
- Profile metadata in `%USERPROFILE%\.chaoscrypto\wp2\alice\profile.json`.

## 3) BA1 pipeline (optional)

```powershell
.\.venv\Scripts\python.exe -m chaoscrypto.cli.app benchmark --config examples/ba1_benchmark.yaml --out out/ba1/bench/results.csv --out-json out/ba1/bench/results.json
.\.venv\Scripts\python.exe -m chaoscrypto.cli.app analyze --config examples/ba1_analyze.yaml --out out/ba1/analyze/analysis.csv --out-json out/ba1/analyze/analysis.json
.\.venv\Scripts\python.exe -m chaoscrypto.cli.app report --bench-csv out/ba1/bench/results.csv --analysis-csv out/ba1/analyze/analysis.csv --out out/ba1/report/report.md --plots-dir out/ba1/report/plots --no-timestamp
```

Clean output folder:
```powershell
Remove-Item -Recurse -Force .\out\ba1
```
