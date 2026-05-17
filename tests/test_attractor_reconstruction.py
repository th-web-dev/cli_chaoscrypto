import csv
import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from chaoscrypto.cli.app import app


def _cfg() -> dict:
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
            "chaos_engine": ["lorenz", "rossler"],
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


def test_attractor_reconstruct_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    _init_profile(runner)
    cfg_path = Path(tmp_path) / "recon.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg()), encoding="utf-8")
    csv_out = Path(tmp_path) / "recon.csv"
    json_out = Path(tmp_path) / "recon.json"

    res = runner.invoke(
        app,
        [
            "attractor-reconstruct",
            "--config",
            str(cfg_path),
            "--out",
            str(csv_out),
            "--out-json",
            str(json_out),
            "--embedding-dim",
            "3",
            "--delay-bytes",
            "1",
            "--max-samples",
            "2000",
        ],
    )
    assert res.exit_code == 0, res.output
    with csv_out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["chaos_engine"] in {"lorenz", "rossler"}
    assert rows[0]["recon_r2"] != ""
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["summary"]["rows"] == 2
    assert "mean_recon_r2" in payload["summary"]


def test_attractor_reconstruct_rejects_invalid_params(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    _init_profile(runner)
    cfg_path = Path(tmp_path) / "recon.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg()), encoding="utf-8")
    csv_out = Path(tmp_path) / "recon.csv"
    res = runner.invoke(
        app,
        [
            "attractor-reconstruct",
            "--config",
            str(cfg_path),
            "--out",
            str(csv_out),
            "--embedding-dim",
            "1",
        ],
    )
    assert res.exit_code != 0
