from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

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
    profiles_root,
    profile_dir,
)
from chaoscrypto.bench.runner import (
    ConfigError,
    parse_config,
    run_benchmark,
    write_csv,
    write_json_output,
)
from chaoscrypto.orchestrator.pipeline import (
    build_memory_field,
    derive_initial_state,
    generate_keystream,
    decrypt_bytes,
    encrypt_bytes,
)

app = typer.Typer(help="ChaosCrypto WP2 CLI (MVP)")
profile_app = typer.Typer(help="Profile utilities (list/show)")


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
    dt: float = typer.Option(constants.DEFAULT_DT, help="Time step for Lorenz integration"),
    warmup: int = typer.Option(constants.DEFAULT_WARMUP, help="Warmup iterations before sampling"),
    quant_k: float = typer.Option(constants.DEFAULT_QUANT_K, help="Quantization factor for sampling"),
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
        dt=dt,
        warmup=warmup,
        quant_k=quant_k,
    )
    assert computed_fp == field_fp  # sanity check

    enc_payload = {
        "version": constants.VERSION,
        "cipher": "lorenz",
        "memory": {"type": params.type, "size": params.size, "scale": params.scale},
        "seed_strategy": constants.SEED_STRATEGY,
        "sampling": {
            "type": constants.SAMPLING_TYPE,
            "warmup": warmup,
            "dt": dt,
            "quant_k": quant_k,
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

    sampling_meta = enc_payload.get("sampling") or {}
    dt = float(sampling_meta.get("dt", constants.DEFAULT_DT))
    warmup = int(sampling_meta.get("warmup", constants.DEFAULT_WARMUP))
    quant_k = float(sampling_meta.get("quant_k", constants.DEFAULT_QUANT_K))

    ciphertext = decode_ciphertext(enc_payload["ciphertext"])
    try:
        plaintext = decrypt_bytes(
            ciphertext,
            token_bytes=token_bytes,
            coord=coord_tuple,
            params=params,
            expected_fingerprint=expected_fp,
            dt=dt,
            warmup=warmup,
            quant_k=quant_k,
        )
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(plaintext)
    typer.secho(f"Decrypted → {output_path}", fg=typer.colors.GREEN)


@app.command()
def keystream(
    profile: str = typer.Option(..., "--profile", "-p", help="Profile name"),
    token: str = typer.Option(..., "--token", "-t", help="Token used for init"),
    coord: str = typer.Option(..., "--coord", "-c", help="Coordinate as 'x,y'"),
    nbytes: int = typer.Option(..., "--nbytes", "-n", help="Number of keystream bytes to generate"),
    dt: float = typer.Option(constants.DEFAULT_DT, help="Time step for Lorenz integration"),
    warmup: int = typer.Option(constants.DEFAULT_WARMUP, help="Warmup iterations before sampling"),
    quant_k: float = typer.Option(constants.DEFAULT_QUANT_K, help="Quantization factor for sampling"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write raw keystream bytes to file"),
    hex_out: bool = typer.Option(False, "--hex", help="Write hex to stdout"),
    base64_out: bool = typer.Option(False, "--base64", help="Write base64 to stdout"),
    hash_out: bool = typer.Option(False, "--hash", help="Write SHA-256 hash (hex) to stdout"),
):
    """
    Generate a deterministic keystream (for analysis/benchmarks) without encrypting data.
    """
    if nbytes < 0:
        typer.secho("nbytes must be >= 0", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    output_flags = [
        out is not None,
        hex_out,
        base64_out,
        hash_out,
    ]
    selected = sum(1 for flag in output_flags if flag)
    if selected == 0:
        hash_out = True  # default
    elif selected > 1:
        typer.secho("Choose exactly one output option.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

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

    init_state = derive_initial_state(field, coord_tuple)
    ks = generate_keystream(nbytes, init_state, dt=dt, warmup=warmup, quant_k=quant_k)

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(ks)
        typer.secho(f"Wrote keystream bytes → {out}", fg=typer.colors.GREEN)
        return

    if hex_out:
        typer.echo(ks.hex())
        return

    if base64_out:
        import base64

        typer.echo(base64.b64encode(ks).decode("ascii"))
        return

    if hash_out:
        import hashlib

        typer.echo(hashlib.sha256(ks).hexdigest())
        return


@profile_app.command("list")
def profile_list():
    """List available profiles."""
    root = profiles_root()
    if not root.exists():
        typer.echo("No profiles found.")
        return
    profiles: List[str] = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not profiles:
        typer.echo("No profiles found.")
        return
    for name in profiles:
        typer.echo(name)


@profile_app.command("show")
def profile_show(profile: str = typer.Option(..., "--profile", "-p", help="Profile name")):
    """Show profile metadata."""
    if not profile_exists(profile):
        typer.secho(f"Profile '{profile}' not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    meta = load_profile_meta(profile)
    typer.echo(meta)


@app.command()
def selftest():
    """
    Run the built-in Golden Vector test (no filesystem writes).
    """
    token = b"test-token"
    params = MemoryParams(type=constants.MEMORY_TYPE, size=constants.DEFAULT_MEMORY_SIZE, scale=constants.DEFAULT_MEMORY_SCALE)
    coord = (12, 34)
    plaintext = b"hello"

    ciphertext, field_fp = encrypt_bytes(
        plaintext,
        token_bytes=token,
        coord=coord,
        params=params,
        dt=constants.DEFAULT_DT,
        warmup=constants.DEFAULT_WARMUP,
        quant_k=constants.DEFAULT_QUANT_K,
    )
    decrypted = decrypt_bytes(
        ciphertext,
        token_bytes=token,
        coord=coord,
        params=params,
        expected_fingerprint=field_fp,
        dt=constants.DEFAULT_DT,
        warmup=constants.DEFAULT_WARMUP,
        quant_k=constants.DEFAULT_QUANT_K,
    )

    if decrypted == plaintext:
        typer.secho("Selftest passed (Golden Vector).", fg=typer.colors.GREEN)
    else:
        typer.secho("Selftest FAILED.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


app.add_typer(profile_app, name="profile")


@app.command()
def benchmark(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True, help="YAML benchmark config"),
    out: Path = typer.Option(..., "--out", "-o", help="CSV output path"),
    out_json: Path | None = typer.Option(None, "--out-json", help="Optional JSON output path"),
    jobs: int = typer.Option(1, "--jobs", "-j", help="Parallel jobs (variants), default 1"),
    json_summary: bool = typer.Option(False, "--json", help="Print summary JSON to stdout"),
):
    """
    Run benchmark variants from YAML config and export CSV/JSON.
    """
    try:
        cfg = parse_config(config)
    except ConfigError as exc:
        typer.secho(f"Config error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Validate profile exists and matches memory params
    if not profile_exists(cfg.bench.profile):
        typer.secho(f"Profile '{cfg.bench.profile}' not found. Run init first.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    profile_meta = load_profile_meta(cfg.bench.profile)
    profile_params = memory_params_from_meta(profile_meta)
    if any(val != profile_params.size for val in cfg.matrix.size):
        typer.secho("Matrix size values must match the profile size.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if any(val != profile_params.scale for val in cfg.matrix.scale):
        typer.secho("Matrix scale values must match the profile scale.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        records = run_benchmark(cfg, jobs=jobs)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Benchmark failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        write_csv(out, records)
        if out_json:
            write_json_output(out_json, records)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Failed to write outputs: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.secho(f"Benchmark complete. CSV → {out}", fg=typer.colors.GREEN)
    if out_json:
        typer.secho(f"JSON → {out_json}", fg=typer.colors.GREEN)

    if json_summary:
        import json

        summary = {
            "runs": len(records),
            "csv": str(out),
            "json": str(out_json) if out_json else None,
            "variants": len(records),
        }
        typer.echo(json.dumps(summary))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
