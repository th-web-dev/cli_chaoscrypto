import csv
from pathlib import Path

import yaml
from typer.testing import CliRunner

from chaoscrypto.cli.app import app


def test_benchmark_memory_matrix(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    runner.invoke(
        app,
        [
            "init",
            "--profile",
            "alice",
            "--token",
            "secret",
            "--size",
            "128",
            "--scale",
            "0.1",
            "--memory-type",
            "opensimplex",
        ],
    )

    cfg = {
        "bench": {
            "profile": "alice",
            "token": "secret",
            "coord": [0, 0],
            "nbytes": 1024,
            "repeats": 1,
            "field_regen_each_repeat": True,
        },
        "matrix": {
            "dt": [0.01],
            "warmup": [100],
            "quant_k": [100000.0],
            "size": [128],
            "scale": [0.1],
            "seed_strategy": ["neighborhood3"],
            "memory_type": ["opensimplex", "perlin"],
        },
        "metrics": {"include_field_time": False, "include_xor_time": False, "include_decrypt_time": False, "keystream_hash": "sha256"},
        "output": {"include_timestamp_utc": False, "include_field_fingerprint": True, "include_keystream_preview": False, "keystream_preview_bytes": 8},
        "validate": {"assert_deterministic_within_run": True, "extra_determinism_check": False},
    }
    cfg_path = Path(tmp_path) / "bench.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out_csv = Path(tmp_path) / "out.csv"

    res = runner.invoke(app, ["benchmark", "--config", str(cfg_path), "--out", str(out_csv)])
    assert res.exit_code == 0, res.output

    with out_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    hashes = {r["keystream_sha256"] for r in rows}
    assert len(hashes) == 2  # different per memory type
