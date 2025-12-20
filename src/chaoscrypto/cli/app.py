from __future__ import annotations

from pathlib import Path
from typing import Tuple

import typer

from chaoscrypto.core import constants
from chaoscrypto.core.memory.base import MemoryParams
from chaoscrypto.io.formats import decode_ciphertext, encode_ciphertext, read_json, write_json
from chaoscrypto.io.profiles import (
    load_profile_meta,
    memory_params_from_meta,
    profile_exists,
    save_profile_meta,
    token_fingerprint,
)
from chaoscrypto.orchestrator.pipeline import (
    build_memory_field,
    decrypt_bytes,
    encrypt_bytes,
)

app = typer.Typer(help="ChaosCrypto WP2 CLI (MVP)")


def parse_coord(coord: str) -> Tuple[int, int]:
    try:
        if "," in coord:
            x_str, y_str = coord.split(",", 1)
        else:
            parts = coord.split()
            if len(parts) != 2:
                raise ValueError
            x_str, y_str = parts
        return int(x_str.strip()), int(y_str.strip())
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter("coord must be provided as 'x,y' or 'x y'") from exc


@app.command()
def init(
    profile: str = typer.Option(..., "--profile", "-p", help="Profile name"),
    token: str = typer.Option(..., "--token", "-t", help="Shared token (kept client-side)"),
    size: int = typer.Option(constants.DEFAULT_MEMORY_SIZE, help="Memory size (NxN)"),
    scale: float = typer.Option(constants.DEFAULT_MEMORY_SCALE, help="Memory scale factor"),
):
    """Create a profile and persist deterministic memory metadata."""
    if profile_exists(profile):
        typer.secho(f"Profile '{profile}' already exists.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    params = MemoryParams(type=constants.MEMORY_TYPE, size=size, scale=scale)
    token_bytes = token.encode(constants.ENCODING)

    field, field_fp = build_memory_field(token_bytes, params)
    meta = {
        "version": constants.VERSION,
        "memory": {"type": params.type, "size": params.size, "scale": params.scale},
        "field_fingerprint": field_fp,
        "token_fingerprint": token_fingerprint(token_bytes),
    }
    save_profile_meta(profile, meta)

    typer.secho(
        f"Profile '{profile}' ready. field_fingerprint={field_fp}",
        fg=typer.colors.GREEN,
    )


@app.command()
def encrypt(
    profile: str = typer.Option(..., "--profile", "-p", help="Profile name"),
    token: str = typer.Option(..., "--token", "-t", help="Token used for init"),
    coord: str = typer.Option(..., "--coord", "-c", help="Coordinate as 'x,y'"),
    input_path: Path = typer.Option(..., "--in", "-i", exists=True, readable=True, help="Plaintext file"),
    output_path: Path = typer.Option(..., "--out", "-o", help="Output enc.json path"),
):
    """Encrypt a plaintext file to enc.json with full reproduction metadata."""
    if not profile_exists(profile):
        typer.secho(f"Profile '{profile}' not found. Run init first.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    meta = load_profile_meta(profile)
    params = memory_params_from_meta(meta)
    token_bytes = token.encode(constants.ENCODING)
    coord_tuple = parse_coord(coord)

    field, field_fp = build_memory_field(token_bytes, params)
    if field_fp != meta["field_fingerprint"]:
        typer.secho("Token or parameters mismatch (field fingerprint differs).", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    plaintext = input_path.read_bytes()
    ciphertext, computed_fp = encrypt_bytes(
        plaintext,
        token_bytes=token_bytes,
        coord=coord_tuple,
        params=params,
        dt=constants.DEFAULT_DT,
        warmup=constants.DEFAULT_WARMUP,
        quant_k=constants.DEFAULT_QUANT_K,
    )
    assert computed_fp == field_fp  # sanity check

    enc_payload = {
        "version": constants.VERSION,
        "cipher": "lorenz",
        "memory": {"type": params.type, "size": params.size, "scale": params.scale},
        "seed_strategy": constants.SEED_STRATEGY,
        "sampling": {
            "type": constants.SAMPLING_TYPE,
            "warmup": constants.DEFAULT_WARMUP,
            "dt": constants.DEFAULT_DT,
            "quant_k": constants.DEFAULT_QUANT_K,
        },
        "coord": {"x": coord_tuple[0], "y": coord_tuple[1]},
        "field_fingerprint": field_fp,
        "ciphertext_encoding": "base64",
        "ciphertext": encode_ciphertext(ciphertext),
    }

    write_json(output_path, enc_payload)
    typer.secho(f"Encrypted → {output_path}", fg=typer.colors.GREEN)


@app.command()
def decrypt(
    profile: str = typer.Option(..., "--profile", "-p", help="Profile name"),
    token: str = typer.Option(..., "--token", "-t", help="Token used for init"),
    input_path: Path = typer.Option(..., "--in", "-i", exists=True, readable=True, help="enc.json input"),
    output_path: Path = typer.Option(..., "--out", "-o", help="Plaintext output path"),
):
    """Decrypt an enc.json produced by the encrypt command."""
    if not profile_exists(profile):
        typer.secho(f"Profile '{profile}' not found. Run init first.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    enc_payload = read_json(input_path)
    meta = load_profile_meta(profile)
    params = memory_params_from_meta(meta)
    token_bytes = token.encode(constants.ENCODING)

    # Basic metadata validation
    enc_memory = enc_payload.get("memory", {})
    if (
        int(enc_memory.get("size", params.size)) != params.size
        or float(enc_memory.get("scale", params.scale)) != params.scale
    ):
        typer.secho("enc.json memory parameters do not match the profile.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    coord_meta = enc_payload.get("coord")
    if not coord_meta or "x" not in coord_meta or "y" not in coord_meta:
        typer.secho("enc.json missing coordinate metadata.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    coord_tuple = (int(coord_meta["x"]), int(coord_meta["y"]))

    expected_fp = enc_payload.get("field_fingerprint")
    if not expected_fp:
        typer.secho("enc.json missing field_fingerprint.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    ciphertext = decode_ciphertext(enc_payload["ciphertext"])
    try:
        plaintext = decrypt_bytes(
            ciphertext,
            token_bytes=token_bytes,
            coord=coord_tuple,
            params=params,
            expected_fingerprint=expected_fp,
            dt=constants.DEFAULT_DT,
            warmup=constants.DEFAULT_WARMUP,
            quant_k=constants.DEFAULT_QUANT_K,
        )
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(plaintext)
    typer.secho(f"Decrypted → {output_path}", fg=typer.colors.GREEN)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
