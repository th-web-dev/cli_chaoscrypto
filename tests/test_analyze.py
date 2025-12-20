import csv
import pytest
import yaml
from pathlib import Path

from typer.testing import CliRunner

from chaoscrypto.cli.app import app
from chaoscrypto.analysis.runner import parse_config, ConfigError


def test_analyze_config_missing(tmp_path):
    cfg_path = Path(tmp_path) / "analyze.yaml"
    cfg_path.write_text("analyze: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        parse_config(cfg_path)


def test_analyze_smoke(tmp_path, monkeypatch):
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
        "analyze": {"profile": "alice", "token": "secret", "coord": [0, 0], "nbytes": 2048},
        "matrix": {"dt": [0.01], "warmup": [100, 1000], "quant_k": [100000.0], "size": [128], "scale": [0.1], "seed_strategy": ["neighborhood3", "window_mean_3x3"]},
        "metrics": {
            "bit_balance": True,
            "byte_histogram": True,
            "chi_square_bytes": True,
            "autocorr_bits": {"enabled": True, "max_lag": 2},
            "runs_test_bits": True,
            "hamming_weight_window": {"enabled": True, "window_bits": 256},
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
    cfg_path = Path(tmp_path) / "analyze.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    csv_out = Path(tmp_path) / "out.csv"
    json_out = Path(tmp_path) / "out.json"

    res = runner.invoke(
        app,
        [
            "analyze",
            "--config",
            str(cfg_path),
            "--out",
            str(csv_out),
            "--out-json",
            str(json_out),
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    assert csv_out.exists()
    lines = csv_out.read_text().splitlines()
    # header + rows (len(matrix product)= 2 warmup *2 seed =4)
    assert len(lines) == 1 + 4


def test_analyze_determinism_and_variation(tmp_path, monkeypatch):
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

    base_cfg = {
        "analyze": {"profile": "alice", "token": "secret", "coord": [1, 1], "nbytes": 1024},
        "matrix": {"dt": [0.01], "warmup": [100], "quant_k": [100000.0], "size": [128], "scale": [0.1], "seed_strategy": ["neighborhood3"]},
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

    def run_an(cfg, csv_path):
        cfg_path = Path(tmp_path) / f"{csv_path.stem}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        res = runner.invoke(app, ["analyze", "--config", str(cfg_path), "--out", str(csv_path)])
        assert res.exit_code == 0, res.output
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        return rows

    rows1 = run_an(base_cfg, Path(tmp_path) / "a1.csv")
    rows1b = run_an(base_cfg, Path(tmp_path) / "a1b.csv")
    assert rows1[0]["keystream_sha256"] == rows1b[0]["keystream_sha256"]

    base_cfg["matrix"]["warmup"] = [1000]
    rows2 = run_an(base_cfg, Path(tmp_path) / "a2.csv")
    assert rows1[0]["keystream_sha256"] != rows2[0]["keystream_sha256"]

    base_cfg["matrix"]["warmup"] = [100]
    base_cfg["matrix"]["seed_strategy"] = ["window_mean_3x3"]
    rows3 = run_an(base_cfg, Path(tmp_path) / "a3.csv")
    assert rows1[0]["keystream_sha256"] != rows3[0]["keystream_sha256"]

    # Sorting stability
    assert rows1 == sorted(rows1, key=lambda r: (int(r["coord_x"]), int(r["coord_y"]), float(r["dt"]), int(r["warmup"]), float(r["quant_k"]), int(r["size"]), float(r["scale"]), r.get("seed_strategy")))
