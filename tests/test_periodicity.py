import csv
import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from chaoscrypto.cli.app import app


def _periodicity_cfg() -> dict:
    return {
        "analyze": {
            "profile": "alice",
            "token": "secret",
            "coord": [1, 2],
            "nbytes": 8192,
        },
        "matrix": {
            "dt": [0.01],
            "warmup": [100],
            "quant_k": [100000.0],
            "size": [128],
            "scale": [0.1],
            "seed_strategy": ["neighborhood3"],
        },
        "metrics": {
            "bit_balance": False,
            "byte_histogram": False,
            "chi_square_bytes": False,
            "autocorr_bits": {"enabled": False, "max_lag": 0},
            "runs_test_bits": False,
            "hamming_weight_window": {"enabled": False, "window_bits": 0},
            "nist_suite": {"enabled": False, "alpha": 0.01},
        },
        "output": {
            "include_timestamp_utc": False,
            "include_field_fingerprint": True,
            "include_keystream_sha256": True,
            "include_preview_base64": False,
            "preview_bytes": 8,
        },
        "validate": {"assert_deterministic_within_run": True},
    }


def _init_profile(runner: CliRunner):
    init_res = runner.invoke(
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
    assert init_res.exit_code == 0, init_res.output


def test_periodicity_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    _init_profile(runner)

    cfg_path = Path(tmp_path) / "periodicity.yaml"
    cfg_path.write_text(yaml.safe_dump(_periodicity_cfg()), encoding="utf-8")
    csv_out = Path(tmp_path) / "periodicity.csv"
    json_out = Path(tmp_path) / "periodicity.json"

    res = runner.invoke(
        app,
        [
            "periodicity",
            "--config",
            str(cfg_path),
            "--out",
            str(csv_out),
            "--out-json",
            str(json_out),
            "--chunk-size-bytes",
            "1024",
            "--lag-step-bytes",
            "1024",
            "--max-detect-period-bytes",
            "2048",
        ],
    )
    assert res.exit_code == 0, res.output

    with csv_out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert int(row["chunk_count"]) > 0
    assert 0.0 <= float(row["lag_match_ratio"]) <= 1.0
    assert int(row["unique_chunk_hashes"]) > 0

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["summary"]["rows"] == 1
    assert "mean_lag_match_ratio" in payload["summary"]


def test_periodicity_rejects_invalid_parameters(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    _init_profile(runner)

    cfg_path = Path(tmp_path) / "periodicity.yaml"
    cfg_path.write_text(yaml.safe_dump(_periodicity_cfg()), encoding="utf-8")
    csv_out = Path(tmp_path) / "periodicity.csv"

    res = runner.invoke(
        app,
        [
            "periodicity",
            "--config",
            str(cfg_path),
            "--out",
            str(csv_out),
            "--chunk-size-bytes",
            "0",
        ],
    )
    assert res.exit_code != 0
