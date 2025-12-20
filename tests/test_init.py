import json
from pathlib import Path

from typer.testing import CliRunner

from chaoscrypto.cli.app import app


def test_init_creates_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "init",
            "--profile",
            "alice",
            "--token",
            "secret-token",
            "--size",
            "128",
            "--scale",
            "0.1",
        ],
    )
    assert result.exit_code == 0, result.output

    meta_path = Path(tmp_path) / ".chaoscrypto" / "wp2" / "alice" / "profile.json"
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text())
    assert meta["memory"]["size"] == 128
    assert meta["memory"]["scale"] == 0.1
    assert "field_fingerprint" in meta
    assert "token_fingerprint" in meta
