from pathlib import Path

from typer.testing import CliRunner

from chaoscrypto.cli.app import app


def test_profile_list_and_show(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    init_result = runner.invoke(
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
    assert init_result.exit_code == 0, init_result.output

    list_result = runner.invoke(app, ["profile", "list"])
    assert list_result.exit_code == 0
    assert "alice" in list_result.output

    show_result = runner.invoke(app, ["profile", "show", "--profile", "alice"])
    assert show_result.exit_code == 0
    assert "field_fingerprint" in show_result.output


def test_selftest_command(tmp_path, monkeypatch):
    # HOME not required, but isolate environment for consistency
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(app, ["selftest"])
    assert result.exit_code == 0, result.output
    assert "Selftest" in result.output
