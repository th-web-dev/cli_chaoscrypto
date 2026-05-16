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


def test_usability_eval_generates_outputs(tmp_path):
    log_csv = tmp_path / "usability_log.csv"
    _write_csv(
        log_csv,
        ["session_id", "command_name", "duration_s", "status", "repro_match_previous", "failed_attempts_before_success"],
        [
            {"session_id": "s1", "command_name": "analyze", "duration_s": "0.45", "status": "success", "repro_match_previous": "True", "failed_attempts_before_success": "0"},
            {"session_id": "s1", "command_name": "analyze", "duration_s": "0.60", "status": "fail", "repro_match_previous": "", "failed_attempts_before_success": ""},
            {"session_id": "s2", "command_name": "nist-batch", "duration_s": "1.20", "status": "success", "repro_match_previous": "False", "failed_attempts_before_success": "1"},
        ],
    )
    out_md = tmp_path / "usability_summary.md"
    out_csv = tmp_path / "usability_summary.csv"
    out_json = tmp_path / "usability_summary.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usability-eval",
            "--log-csv",
            str(log_csv),
            "--out-md",
            str(out_md),
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_md.exists()
    assert out_csv.exists()
    assert out_json.exists()
    content = out_md.read_text(encoding="utf-8")
    assert "Usability Evaluation Summary" in content
    assert "Success rate" in content
