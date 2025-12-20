import json
import csv
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from chaoscrypto.cli.app import app
from chaoscrypto.bench.runner import parse_config, ConfigError


def test_parse_config_missing_key(tmp_path):
    cfg_path = Path(tmp_path) / "bench.yaml"
    cfg_path.write_text("bench: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        parse_config(cfg_path)


def test_benchmark_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    # init profile
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

    cfg = {
        "bench": {
            "profile": "alice",
            "token": "secret",
            "coord": [1, 2],
            "nbytes": 4096,
            "repeats": 2,
            "field_regen_each_repeat": True,
        },
        "matrix": {"dt": [0.01], "warmup": [100, 1000], "quant_k": [100000.0], "size": [128], "scale": [0.1]},
        "metrics": {
            "include_field_time": False,
            "include_xor_time": False,
            "include_decrypt_time": False,
            "keystream_hash": "sha256",
        },
        "output": {
            "include_timestamp_utc": False,
            "include_field_fingerprint": True,
            "include_keystream_preview": False,
            "keystream_preview_bytes": 8,
        },
        "validate": {"assert_deterministic_within_run": True, "extra_determinism_check": False},
    }
    cfg_path = Path(tmp_path) / "bench.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    csv_out = Path(tmp_path) / "out.csv"
    json_out = Path(tmp_path) / "out.json"
    res = runner.invoke(
        app,
        [
            "benchmark",
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

    lines = csv_out.read_text().strip().splitlines()
    # header + rows; rows = len(matrix product) * repeats = 2 * 2 = 4
    assert len(lines) == 1 + 4

    data = json.loads(json_out.read_text())
    assert len(data) == 4


def test_benchmark_determinism_and_variation(tmp_path, monkeypatch):
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

    cfg_base = {
        "bench": {
            "profile": "alice",
            "token": "secret",
            "coord": [0, 0],
            "nbytes": 1024,
            "repeats": 1,
            "field_regen_each_repeat": True,
        },
        "matrix": {"dt": [0.01], "warmup": [100], "quant_k": [100000.0], "size": [128], "scale": [0.1]},
        "metrics": {
            "include_field_time": False,
            "include_xor_time": False,
            "include_decrypt_time": False,
            "keystream_hash": "sha256",
        },
        "output": {
            "include_timestamp_utc": False,
            "include_field_fingerprint": False,
            "include_keystream_preview": False,
            "keystream_preview_bytes": 8,
        },
        "validate": {"assert_deterministic_within_run": True, "extra_determinism_check": False},
    }

    cfg_path1 = Path(tmp_path) / "bench1.yaml"
    cfg_path1.write_text(yaml.safe_dump(cfg_base), encoding="utf-8")
    csv1 = Path(tmp_path) / "out1.csv"
    res1 = runner.invoke(app, ["benchmark", "--config", str(cfg_path1), "--out", str(csv1)])
    assert res1.exit_code == 0, res1.output

    # rerun same config -> identical hash
    csv1b = Path(tmp_path) / "out1b.csv"
    res1b = runner.invoke(app, ["benchmark", "--config", str(cfg_path1), "--out", str(csv1b)])
    assert res1b.exit_code == 0, res1b.output
    import csv

    with csv1.open() as f:
        rows1 = list(csv.DictReader(f))
    with csv1b.open() as f:
        rows1b = list(csv.DictReader(f))
    assert rows1[0]["keystream_sha256"] == rows1b[0]["keystream_sha256"]

    # variation warmup -> hash should differ
    cfg_base["matrix"]["warmup"] = [1000]
    cfg_path2 = Path(tmp_path) / "bench2.yaml"
    cfg_path2.write_text(yaml.safe_dump(cfg_base), encoding="utf-8")
    csv2 = Path(tmp_path) / "out2.csv"
    res2 = runner.invoke(app, ["benchmark", "--config", str(cfg_path2), "--out", str(csv2)])
    assert res2.exit_code == 0, res2.output
    with csv2.open() as f:
        rows2 = list(csv.DictReader(f))
    assert rows1[0]["keystream_sha256"] != rows2[0]["keystream_sha256"]
