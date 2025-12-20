#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/out/ba1"

mkdir -p "$OUT_DIR/bench" "$OUT_DIR/analyze" "$OUT_DIR/report/plots"

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python --version
  uname -a
  pip freeze
} > "$OUT_DIR/run_meta.txt"

python -m chaoscrypto.cli.app benchmark \
  --config "$ROOT_DIR/examples/ba1_benchmark.yaml" \
  --out "$OUT_DIR/bench/results.csv" \
  --out-json "$OUT_DIR/bench/results.json"

python -m chaoscrypto.cli.app analyze \
  --config "$ROOT_DIR/examples/ba1_analyze.yaml" \
  --out "$OUT_DIR/analyze/analysis.csv" \
  --out-json "$OUT_DIR/analyze/analysis.json"

python -m chaoscrypto.cli.app report \
  --bench-csv "$OUT_DIR/bench/results.csv" \
  --analysis-csv "$OUT_DIR/analyze/analysis.csv" \
  --out "$OUT_DIR/report/report.md" \
  --plots-dir "$OUT_DIR/report/plots" \
  --no-timestamp
