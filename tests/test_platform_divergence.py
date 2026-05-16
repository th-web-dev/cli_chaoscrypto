import csv
import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from chaoscrypto.cli.app import app


def _platform_cfg() -> dict:
    return {
        "analyze": {
            "profile": "alice",
            "token": "secret",
            "coord": [1, 2],
            "nbytes": 2048,
        },
        "matrix": {
            "dt": [0.01],
            "warmup": [100],
            "quant_k": [100000.0],
            "size": [128],
            "scale": [0.1],
            "seed_strategy": ["neighborhood3", "window_mean_3x3"],
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


def test_platform_check_and_compare_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    _init_profile(runner)

    cfg_path = Path(tmp_path) / "platform.yaml"
    cfg_path.write_text(yaml.safe_dump(_platform_cfg()), encoding="utf-8")
    csv_out = Path(tmp_path) / "platform.csv"
    json_out = Path(tmp_path) / "platform.json"

    res = runner.invoke(
        app,
        [
            "platform-check",
            "--config",
            str(cfg_path),
            "--out",
            str(csv_out),
            "--out-json",
            str(json_out),
            "--runtime-label",
            "wsl-test",
        ],
    )
    assert res.exit_code == 0, res.output

    with csv_out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["runtime_label"] == "wsl-test"
    assert rows[0]["platform_python_version"] != ""
    assert rows[0]["platform_numpy_version"] != ""
    assert rows[0]["field_fingerprint"] != ""
    assert rows[0]["keystream_sha256"] != ""

    compare_csv = Path(tmp_path) / "compare.csv"
    compare_json = Path(tmp_path) / "compare.json"
    cmp_res = runner.invoke(
        app,
        [
            "platform-compare",
            "--reference",
            str(json_out),
            "--candidate",
            str(json_out),
            "--out",
            str(compare_csv),
            "--out-json",
            str(compare_json),
        ],
    )
    assert cmp_res.exit_code == 0, cmp_res.output

    with compare_csv.open() as f:
        compare_rows = list(csv.DictReader(f))
    assert len(compare_rows) == 2
    assert all(row["status"] == "match" for row in compare_rows)
    assert all(row["field_fingerprint_match"] == "True" for row in compare_rows)
    assert all(row["keystream_sha256_match"] == "True" for row in compare_rows)

    compare_payload = json.loads(compare_json.read_text(encoding="utf-8"))
    assert compare_payload["summary"]["matches"] == 2
    assert compare_payload["summary"]["diverged"] == 0


def test_platform_compare_detects_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    _init_profile(runner)

    cfg_path = Path(tmp_path) / "platform.yaml"
    cfg_path.write_text(yaml.safe_dump(_platform_cfg()), encoding="utf-8")
    json_out = Path(tmp_path) / "platform.json"
    csv_out = Path(tmp_path) / "platform.csv"

    res = runner.invoke(
        app,
        [
            "platform-check",
            "--config",
            str(cfg_path),
            "--out",
            str(csv_out),
            "--out-json",
            str(json_out),
            "--runtime-label",
            "reference-env",
        ],
    )
    assert res.exit_code == 0, res.output

    candidate_path = Path(tmp_path) / "platform_candidate.json"
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    payload[0]["keystream_sha256"] = "deadbeef"
    candidate_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    compare_json = Path(tmp_path) / "compare.json"
    cmp_res = runner.invoke(
        app,
        [
            "platform-compare",
            "--reference",
            str(json_out),
            "--candidate",
            str(candidate_path),
            "--out",
            str(Path(tmp_path) / "compare.csv"),
            "--out-json",
            str(compare_json),
        ],
    )
    assert cmp_res.exit_code == 0, cmp_res.output

    compare_payload = json.loads(compare_json.read_text(encoding="utf-8"))
    assert compare_payload["summary"]["diverged"] == 1
    statuses = [row["status"] for row in compare_payload["records"]]
    assert "diverged" in statuses
