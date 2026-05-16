import csv
from pathlib import Path

from typer.testing import CliRunner

from chaoscrypto.cli.app import app


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_ba2_eval_generates_outputs(tmp_path):
    nist_runs = tmp_path / "nist_runs.csv"
    _write_csv(
        nist_runs,
        [
            "dt",
            "warmup",
            "quant_k",
            "seed_strategy",
            "memory_type",
            "nist_passed_count",
            "nist_failed_count",
        ],
        [
            {"dt": "0.01", "warmup": "100", "quant_k": "1e5", "seed_strategy": "neighborhood3", "memory_type": "opensimplex", "nist_passed_count": "13", "nist_failed_count": "2"},
            {"dt": "0.01", "warmup": "1000", "quant_k": "1e7", "seed_strategy": "window_mean_3x3", "memory_type": "perlin", "nist_passed_count": "15", "nist_failed_count": "0"},
        ],
    )

    nist_summary = tmp_path / "nist_summary.csv"
    _write_csv(
        nist_summary,
        ["test_name", "n_fail", "pass_rate"],
        [
            {"test_name": "non_overlapping_template_matching", "n_fail": "2", "pass_rate": "0.0"},
            {"test_name": "frequency_monobit", "n_fail": "0", "pass_rate": "1.0"},
        ],
    )

    avalanche = tmp_path / "avalanche.csv"
    _write_csv(
        avalanche,
        [
            "dt",
            "warmup",
            "quant_k",
            "seed_strategy",
            "memory_type",
            "perturbation_skipped",
            "hamming_distance_ratio",
        ],
        [
            {"dt": "0.01", "warmup": "100", "quant_k": "1e5", "seed_strategy": "neighborhood3", "memory_type": "opensimplex", "perturbation_skipped": "false", "hamming_distance_ratio": "0.49"},
            {"dt": "0.01", "warmup": "100", "quant_k": "1e5", "seed_strategy": "neighborhood3", "memory_type": "opensimplex", "perturbation_skipped": "true", "hamming_distance_ratio": ""},
        ],
    )

    periodicity = tmp_path / "periodicity.csv"
    _write_csv(
        periodicity,
        ["lag_match_ratio", "repeated_chunk_hash_count", "detected_prefix_period_bytes"],
        [
            {"lag_match_ratio": "0.002", "repeated_chunk_hash_count": "0", "detected_prefix_period_bytes": ""},
            {"lag_match_ratio": "0.003", "repeated_chunk_hash_count": "1", "detected_prefix_period_bytes": "64"},
        ],
    )

    out_md = tmp_path / "ba2_eval_summary.md"
    out_csv = tmp_path / "ba2_eval_summary.csv"
    out_json = tmp_path / "ba2_eval_summary.json"
    usability_log = tmp_path / "usability_log.csv"
    _write_csv(
        usability_log,
        ["session_id", "command_name", "duration_s", "status", "repro_match_previous", "failed_attempts_before_success"],
        [
            {"session_id": "s1", "command_name": "analyze", "duration_s": "0.5", "status": "success", "repro_match_previous": "True", "failed_attempts_before_success": "0"},
            {"session_id": "s1", "command_name": "analyze", "duration_s": "0.8", "status": "fail", "repro_match_previous": "", "failed_attempts_before_success": ""},
        ],
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ba2-eval",
            "--nist-runs-csv",
            str(nist_runs),
            "--nist-summary-csv",
            str(nist_summary),
            "--avalanche-csv",
            str(avalanche),
            "--periodicity-csv",
            str(periodicity),
            "--out-md",
            str(out_md),
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
            "--usability-csv",
            str(usability_log),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_md.exists()
    assert out_csv.exists()
    assert out_json.exists()
    assert "BA2 Evaluation Summary" in out_md.read_text(encoding="utf-8")
    assert "## Usability" in out_md.read_text(encoding="utf-8")
