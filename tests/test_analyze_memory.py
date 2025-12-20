import csv
from pathlib import Path

import yaml
from typer.testing import CliRunner

from chaoscrypto.cli.app import app


def test_analyze_memory_matrix(tmp_path, monkeypatch):
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
        ],
    )

    cfg = {
        "analyze": {"profile": "alice", "token": "secret", "coord": [0, 0], "nbytes": 1024},
        "matrix": {
            "dt": [0.01],
            "warmup": [100],
            "quant_k": [100000.0],
            "size": [128],
            "scale": [0.1],
            "seed_strategy": ["neighborhood3"],
            "memory_type": ["opensimplex", "perlin"],
        },
        "metrics": {
            "bit_balance": True,
            "byte_histogram": True,
            "chi_square_bytes": True,
            "autocorr_bits": {"enabled": False, "max_lag": 0},
            "runs_test_bits": True,
            "hamming_weight_window": {"enabled": False, "window_bits": 0},
        },
        "output": {
            "include_timestamp_utc": False,
            "include_field_fingerprint": False,
            "include_keystream_sha256": True,
            "include_preview_base64": False,
            "preview_bytes": 8,
        },
        "validate": {"assert_deterministic_within_run": True},
    }
    cfg_path = Path(tmp_path) / "analyze.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out_csv = Path(tmp_path) / "out.csv"

    res = runner.invoke(app, ["analyze", "--config", str(cfg_path), "--out", str(out_csv)])
    assert res.exit_code == 0, res.output

    with out_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    hashes = {r["keystream_sha256"] for r in rows}
    assert len(hashes) == 2
