from pathlib import Path

from typer.testing import CliRunner

from chaoscrypto.cli.app import app


def test_encrypt_decrypt_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    plaintext_path = Path(tmp_path) / "msg.txt"
    plaintext_path.write_text("hello", encoding="utf-8")
    enc_path = Path(tmp_path) / "enc.json"
    dec_path = Path(tmp_path) / "dec.txt"

    init_result = runner.invoke(
        app,
        [
            "init",
            "--profile",
            "alice",
            "--token",
            "test-token",
            "--size",
            "128",
            "--scale",
            "0.1",
        ],
    )
    assert init_result.exit_code == 0, init_result.output

    enc_result = runner.invoke(
        app,
        [
            "encrypt",
            "--profile",
            "alice",
            "--token",
            "test-token",
            "--coord",
            "12,34",
            "--in",
            str(plaintext_path),
            "--out",
            str(enc_path),
        ],
    )
    assert enc_result.exit_code == 0, enc_result.output
    assert enc_path.exists()

    dec_result = runner.invoke(
        app,
        [
            "decrypt",
            "--profile",
            "alice",
            "--token",
            "test-token",
            "--in",
            str(enc_path),
            "--out",
            str(dec_path),
        ],
    )
    assert dec_result.exit_code == 0, dec_result.output
    assert dec_path.read_text(encoding="utf-8") == "hello"
