import csv
from pathlib import Path

import yaml
from typer.testing import CliRunner

from chaoscrypto.cli.app import app


def _init_profile(runner: CliRunner, name: str = "alice") -> None:
    result = runner.invoke(
        app,
        [
            "init",
            "--profile",
            name,
            "--token",
            "secret",
            "--size",
            "128",
            "--scale",
            "0.1",
        ],
    )
    assert result.exit_code == 0, result.output


def test_nist_batch_writes_runs_and_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    _init_profile(runner)

    cfg = {
        "analyze": {"profile": "alice", "token": "secret", "coord": [0, 0], "nbytes": 4096},
        "matrix": {
            "dt": [0.01],
            "warmup": [100],
            "quant_k": [100000.0],
            "size": [128],
            "scale": [0.1],
            "seed_strategy": ["neighborhood3"],
            "memory_type": ["opensimplex"],
        },
        "metrics": {
            "bit_balance": False,
            "byte_histogram": False,
            "chi_square_bytes": False,
            "autocorr_bits": {"enabled": False, "max_lag": 0},
            "runs_test_bits": False,
            "hamming_weight_window": {"enabled": False, "window_bits": 0},
            "nist_suite": {"enabled": True, "alpha": 0.01},
        },
        "output": {
            "include_timestamp_utc": False,
            "include_field_fingerprint": False,
            "include_keystream_sha256": True,
            "include_preview_base64": False,
            "preview_bytes": 0,
        },
        "validate": {"assert_deterministic_within_run": True},
    }
    cfg_path = Path(tmp_path) / "nist_batch.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out_runs = Path(tmp_path) / "nist_runs.csv"
    out_summary = Path(tmp_path) / "nist_summary.csv"

    result = runner.invoke(
        app,
        ["nist-batch", "--config", str(cfg_path), "--out-runs", str(out_runs), "--out-summary", str(out_summary)],
    )
    assert result.exit_code == 0, result.output

    with out_runs.open(encoding="utf-8") as handle:
        run_rows = list(csv.DictReader(handle))
    with out_summary.open(encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))

    assert len(run_rows) == 1
    assert len(summary_rows) == 15
    assert run_rows[0]["nist_frequency_monobit_status"] in {"pass", "fail", "skip"}
    assert summary_rows[0]["test_name"]


def test_nist_batch_marks_insufficient_bits_skip_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    _init_profile(runner)

    cfg = {
        "analyze": {"profile": "alice", "token": "secret", "coord": [0, 0], "nbytes": 16},
        "matrix": {
            "dt": [0.01],
            "warmup": [100],
            "quant_k": [100000.0],
            "size": [128],
            "scale": [0.1],
            "seed_strategy": ["neighborhood3"],
            "memory_type": ["opensimplex"],
        },
        "metrics": {
            "bit_balance": False,
            "byte_histogram": False,
            "chi_square_bytes": False,
            "autocorr_bits": {"enabled": False, "max_lag": 0},
            "runs_test_bits": False,
            "hamming_weight_window": {"enabled": False, "window_bits": 0},
            "nist_suite": {"enabled": True, "alpha": 0.01},
        },
        "output": {
            "include_timestamp_utc": False,
            "include_field_fingerprint": False,
            "include_keystream_sha256": True,
            "include_preview_base64": False,
            "preview_bytes": 0,
        },
        "validate": {"assert_deterministic_within_run": True},
    }
    cfg_path = Path(tmp_path) / "nist_batch_small.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out_runs = Path(tmp_path) / "nist_runs.csv"
    out_summary = Path(tmp_path) / "nist_summary.csv"

    result = runner.invoke(
        app,
        ["nist-batch", "--config", str(cfg_path), "--out-runs", str(out_runs), "--out-summary", str(out_summary)],
    )
    assert result.exit_code == 0, result.output

    with out_runs.open(encoding="utf-8") as handle:
        run_rows = list(csv.DictReader(handle))
    row = run_rows[0]
    assert row["nist_maurers_universal_status"] == "skip"
    assert row["nist_maurers_universal_skip_reason"] == "insufficient_bits"
