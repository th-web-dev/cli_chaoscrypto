import csv
from pathlib import Path

import numpy as np
import yaml
from typer.testing import CliRunner

from chaoscrypto.analysis.nist_validator import run_full_nist_suite
from chaoscrypto.cli.app import app


def test_run_full_nist_suite_structure():
    rng = np.random.default_rng(12345)
    keystream = rng.integers(0, 256, size=32_768, dtype=np.uint8).tobytes()

    results = run_full_nist_suite(keystream, alpha=0.01)

    assert results["alpha"] == 0.01
    assert results["n_bits"] == len(keystream) * 8
    assert "tests" in results
    assert "frequency_monobit" in results["tests"]
    assert "serial" in results["tests"]
    assert "random_excursions_variant" in results["tests"]
    assert {"passed", "failed", "skipped"} <= set(results["summary"].keys())

    monobit = results["tests"]["frequency_monobit"]
    assert monobit["status"] in {"pass", "fail"}
    assert isinstance(monobit["p_value"], float)
    assert isinstance(monobit["pass"], bool)

    random_excursions = results["tests"]["random_excursions"]
    assert random_excursions["status"] in {"pass", "fail", "skip"}


def test_analyze_with_nist_suite_writes_csv_columns(tmp_path, monkeypatch):
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
        "analyze": {"profile": "alice", "token": "secret", "coord": [0, 0], "nbytes": 4096},
        "matrix": {
            "dt": [0.01],
            "warmup": [100],
            "quant_k": [100000.0],
            "size": [128],
            "scale": [0.1],
            "seed_strategy": ["neighborhood3"],
        },
        "metrics": {
            "bit_balance": True,
            "byte_histogram": True,
            "chi_square_bytes": True,
            "autocorr_bits": {"enabled": False, "max_lag": 0},
            "runs_test_bits": True,
            "hamming_weight_window": {"enabled": False, "window_bits": 0},
            "nist_suite": {"enabled": True, "alpha": 0.01},
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
    cfg_path = Path(tmp_path) / "analyze_nist.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out_csv = Path(tmp_path) / "out.csv"

    result = runner.invoke(app, ["analyze", "--config", str(cfg_path), "--out", str(out_csv)])
    assert result.exit_code == 0, result.output

    with out_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert "nist_frequency_monobit_p_value" in row
    assert "nist_serial_status" in row
    assert "nist_random_excursions_variant_details_json" in row
    assert row["nist_frequency_monobit_status"] in {"pass", "fail"}
    assert row["nist_results_json"]
