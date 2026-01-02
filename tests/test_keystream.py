import base64
import hashlib
import json
import re
from pathlib import Path

from typer.testing import CliRunner

from chaoscrypto.cli.app import app


def test_keystream_determinism(tmp_path, monkeypatch):
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

    res1 = runner.invoke(
        app,
        [
            "keystream",
            "--profile",
            "alice",
            "--token",
            "secret",
            "--coord",
            "10,20",
            "--nbytes",
            "16",
        ],
    )
    res2 = runner.invoke(
        app,
        [
            "keystream",
            "--profile",
            "alice",
            "--token",
            "secret",
            "--coord",
            "10,20",
            "--nbytes",
            "16",
        ],
    )

    assert res1.exit_code == 0, res1.output
    assert res2.exit_code == 0, res2.output
    assert extract_raw_hash(res1.output) == extract_raw_hash(res2.output)


def test_keystream_matches_encrypt_internal_keystream(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    plaintext_path = Path(tmp_path) / "msg.txt"
    plaintext_path.write_text("hello", encoding="utf-8")
    enc_path = Path(tmp_path) / "enc.json"
    ks_path = Path(tmp_path) / "ks.bin"

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

    enc_res = runner.invoke(
        app,
        [
            "encrypt",
            "--profile",
            "alice",
            "--token",
            "secret",
            "--coord",
            "12,34",
            "--in",
            str(plaintext_path),
            "--out",
            str(enc_path),
        ],
    )
    assert enc_res.exit_code == 0, enc_res.output

    enc_data = json.loads(enc_path.read_text())
    ciphertext = base64.b64decode(enc_data["ciphertext"])
    keystream_from_encrypt = bytes(
        p ^ c for p, c in zip(b"hello", ciphertext)
    )
    expected_hash = hashlib.sha256(keystream_from_encrypt).hexdigest()

    ks_res = runner.invoke(
        app,
        [
            "keystream",
            "--profile",
            "alice",
            "--token",
            "secret",
            "--coord",
            "12,34",
            "--nbytes",
            "5",
            "--out",
            str(ks_path),
        ],
    )
    assert ks_res.exit_code == 0, ks_res.output
    ks_bytes = ks_path.read_bytes()
    assert hashlib.sha256(ks_bytes).hexdigest() == expected_hash
    assert ks_bytes == keystream_from_encrypt


def test_keystream_parameter_variation_changes_hash(tmp_path, monkeypatch):
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

    res_default = runner.invoke(
        app,
        [
            "keystream",
            "--profile",
            "alice",
            "--token",
            "secret",
            "--coord",
            "1,1",
            "--nbytes",
            "8",
        ],
    )
    res_modified = runner.invoke(
        app,
        [
            "keystream",
            "--profile",
            "alice",
            "--token",
            "secret",
            "--coord",
            "1,1",
            "--nbytes",
            "8",
            "--warmup",
            "10",
        ],
    )

    assert res_default.exit_code == 0, res_default.output
    assert res_modified.exit_code == 0, res_modified.output
    assert extract_raw_hash(res_default.output) != extract_raw_hash(res_modified.output)


def test_keystream_smoke_hash_default(tmp_path, monkeypatch):
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

    res = runner.invoke(
        app,
        [
            "keystream",
            "--profile",
            "alice",
            "--token",
            "secret",
            "--coord",
            "0,0",
            "--nbytes",
            "0",
        ],
    )
    assert res.exit_code == 0, res.output
    assert extract_raw_hash(res.output) == hashlib.sha256(b"").hexdigest()


def extract_raw_hash(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    hex_lines = [line for line in lines if re.fullmatch(r"[0-9a-f]{64}", line)]
    assert hex_lines, f"No SHA-256 hash line found in output: {output}"
    return hex_lines[-1]
