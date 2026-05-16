import csv
import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from chaoscrypto.cli.app import app


def _avalanche_cfg() -> dict:
    return {
        "analyze": {
            "profile": "alice",
            "token": "secret",
            "coord": [1, 2],
            "nbytes": 1024,
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


def test_avalanche_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    _init_profile(runner)

    cfg_path = Path(tmp_path) / "avalanche.yaml"
    cfg_path.write_text(yaml.safe_dump(_avalanche_cfg()), encoding="utf-8")
    csv_out = Path(tmp_path) / "avalanche.csv"
    json_out = Path(tmp_path) / "avalanche.json"

    res = runner.invoke(
        app,
        [
            "avalanche",
            "--config",
            str(cfg_path),
            "--out",
            str(csv_out),
            "--out-json",
            str(json_out),
            "--token-bit-flips",
            "4",
            "--coord-bit-flips",
            "3",
        ],
    )
    assert res.exit_code == 0, res.output

    with csv_out.open() as f:
        rows = list(csv.DictReader(f))
    # token flips 4 + coord x flips 3 + coord y flips 3 = 10 rows for one variant
    assert len(rows) == 10
    for row in rows:
        ratio = float(row["hamming_distance_ratio"])
        assert 0.0 <= ratio <= 1.0
        assert row["keystream_sha256_base"] != ""
        assert row["keystream_sha256_perturbed"] != ""

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["summary"]["rows"] == 10
    assert "mean_hamming_distance_ratio" in payload["summary"]


def test_avalanche_rejects_negative_flip_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    _init_profile(runner)

    cfg_path = Path(tmp_path) / "avalanche.yaml"
    cfg_path.write_text(yaml.safe_dump(_avalanche_cfg()), encoding="utf-8")
    csv_out = Path(tmp_path) / "avalanche.csv"

    res = runner.invoke(
        app,
        [
            "avalanche",
            "--config",
            str(cfg_path),
            "--out",
            str(csv_out),
            "--token-bit-flips",
            "-1",
        ],
    )
    assert res.exit_code != 0


def test_avalanche_skips_modulo_invariant_coord_flips(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    _init_profile(runner)

    cfg_path = Path(tmp_path) / "avalanche.yaml"
    cfg_path.write_text(yaml.safe_dump(_avalanche_cfg()), encoding="utf-8")
    csv_out = Path(tmp_path) / "avalanche.csv"
    json_out = Path(tmp_path) / "avalanche.json"

    res = runner.invoke(
        app,
        [
            "avalanche",
            "--config",
            str(cfg_path),
            "--out",
            str(csv_out),
            "--out-json",
            str(json_out),
            "--token-bit-flips",
            "0",
            "--coord-bit-flips",
            "8",
        ],
    )
    assert res.exit_code == 0, res.output

    with csv_out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 16
    skipped = [r for r in rows if r["perturbation_skipped"] == "True"]
    assert len(skipped) == 2
    assert all(r["perturbation_bit_index"] == "7" for r in skipped)
    assert all(r["hamming_distance_ratio"] == "" for r in skipped)

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["summary"]["rows"] == 16
    assert payload["summary"]["rows_skipped"] == 2
    assert payload["summary"]["rows_evaluated"] == 14
