import csv
from pathlib import Path

from typer.testing import CliRunner

from chaoscrypto.cli.app import app


def _read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_study_run_logs_success_row(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    out_log = tmp_path / "usability.csv"

    result = runner.invoke(
        app,
        [
            "study-run",
            "--out-log",
            str(out_log),
            "--session-id",
            "s1",
            "profile",
            "list",
        ],
    )
    assert result.exit_code == 0, result.output
    rows = _read_rows(out_log)
    assert len(rows) == 1
    row = rows[0]
    assert row["session_id"] == "s1"
    assert row["status"] == "success"
    assert row["command_name"] == "profile"
    assert int(float(row["command_arg_count"])) >= 2


def test_study_run_marks_repro_match_on_same_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    out_log = tmp_path / "usability.csv"
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("stable-output", encoding="utf-8")

    args = [
        "study-run",
        "--out-log",
        str(out_log),
        "--session-id",
        "s2",
        "--repro-key",
        "same-run",
        "--artifact",
        str(artifact),
        "profile",
        "list",
    ]
    result1 = runner.invoke(app, args)
    assert result1.exit_code == 0, result1.output
    result2 = runner.invoke(app, args)
    assert result2.exit_code == 0, result2.output

    rows = _read_rows(out_log)
    assert len(rows) == 2
    assert rows[0]["repro_match_previous"] in {"", "None"}
    assert rows[1]["repro_match_previous"] == "True"
