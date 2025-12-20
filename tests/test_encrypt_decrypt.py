from pathlib import Path
import json

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

    # enc.json contains sampling metadata
    enc_data = json.loads(enc_path.read_text())
    assert enc_data["sampling"]["dt"] == 0.01
    assert enc_data["sampling"]["warmup"] == 1000
    assert enc_data["sampling"]["quant_k"] == 1e5
    assert enc_data["seed_strategy"]["name"] == "neighborhood3"

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


def test_encrypt_decrypt_new_seed_strategy(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    plaintext_path = Path(tmp_path) / "msg.txt"
    plaintext_path.write_text("hello", encoding="utf-8")
    enc_path = Path(tmp_path) / "enc.json"
    dec_path = Path(tmp_path) / "dec.txt"

    runner.invoke(
        app,
        [
            "init",
            "--profile",
            "carol",
            "--token",
            "seed-test",
            "--size",
            "128",
            "--scale",
            "0.1",
            "--memory-type",
            "perlin",
        ],
    )

    enc_result = runner.invoke(
        app,
        [
            "encrypt",
            "--profile",
            "carol",
            "--token",
            "seed-test",
            "--coord",
            "3,4",
            "--in",
            str(plaintext_path),
            "--out",
            str(enc_path),
            "--seed-strategy",
            "window_mean_3x3",
            "--memory-type",
            "perlin",
        ],
    )
    assert enc_result.exit_code == 0, enc_result.output

    dec_result = runner.invoke(
        app,
        [
            "decrypt",
            "--profile",
            "carol",
            "--token",
            "seed-test",
            "--in",
            str(enc_path),
            "--out",
            str(dec_path),
        ],
    )
    assert dec_result.exit_code == 0, dec_result.output
    assert dec_path.read_text(encoding="utf-8") == "hello"


def test_encrypt_decrypt_with_custom_params(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    plaintext_path = Path(tmp_path) / "msg.txt"
    plaintext_path.write_text("hello", encoding="utf-8")
    enc_path = Path(tmp_path) / "enc.json"
    dec_path = Path(tmp_path) / "dec.txt"

    runner.invoke(
        app,
        [
            "init",
            "--profile",
            "bob",
            "--token",
            "other-token",
            "--size",
            "128",
            "--scale",
            "0.1",
        ],
    )

    dt = 0.02
    warmup = 50
    quant_k = 12345

    enc_result = runner.invoke(
        app,
        [
            "encrypt",
            "--profile",
            "bob",
            "--token",
            "other-token",
            "--coord",
            "1,2",
            "--in",
            str(plaintext_path),
            "--out",
            str(enc_path),
            "--dt",
            str(dt),
            "--warmup",
            str(warmup),
            "--quant-k",
            str(quant_k),
            "--seed-strategy",
            "window_mean_3x3",
        ],
    )
    assert enc_result.exit_code == 0, enc_result.output

    enc_data = json.loads(enc_path.read_text())
    assert enc_data["sampling"]["dt"] == dt
    assert enc_data["sampling"]["warmup"] == warmup
    assert enc_data["sampling"]["quant_k"] == quant_k
    assert enc_data["seed_strategy"]["name"] == "window_mean_3x3"

    dec_result = runner.invoke(
        app,
        [
            "decrypt",
            "--profile",
            "bob",
            "--token",
            "other-token",
            "--in",
            str(enc_path),
            "--out",
            str(dec_path),
        ],
    )
    assert dec_result.exit_code == 0, dec_result.output
    assert dec_path.read_text(encoding="utf-8") == "hello"
