from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, List

import typer

from chaoscrypto.core import constants
from chaoscrypto.core.crypto.xor import xor_bytes
from chaoscrypto.core.memory.base import MemoryParams
from chaoscrypto.io.formats import decode_ciphertext, encode_ciphertext, read_json, write_json
from chaoscrypto.io.profiles import (
    load_profile_meta,
    memory_params_from_meta,
    profile_exists,
    profile_meta_path,
    save_profile_meta,
    token_fingerprint,
    profiles_root,
    profile_dir,
)
from chaoscrypto.bench.runner import (
    ConfigError,
    MatrixConfig as BenchMatrixConfig,
    FullConfig as BenchFullConfig,
    parse_config,
    run_benchmark,
    write_csv,
    write_json_output,
)
from chaoscrypto.analysis.runner import (
    ConfigError as AnalyzeConfigError,
    MatrixConfig as AnalyzeMatrixConfig,
    FullConfig as AnalyzeFullConfig,
    parse_config as parse_analyze_config,
    run_analyze,
    write_csv as write_analyze_csv,
    write_json_output as write_analyze_json,
)
from chaoscrypto.report.runner import generate_report
from chaoscrypto.core.seed.base import list_seed_strategies
from chaoscrypto.orchestrator.pipeline import (
    build_memory_field,
    derive_initial_state,
    generate_keystream,
    encrypt_bytes,
)
from chaoscrypto.utils.logging import get_logger, resolve_log_level, set_command_context, setup_logging
from chaoscrypto.cli.ui import (
    print_run_header,
    print_io_read,
    print_io_write,
    print_noise_excerpt,
    print_keystream_preview,
    print_cipher_metadata,
    print_preview_bytes,
    print_variant_lines,
    print_done,
)

app = typer.Typer(help="ChaosCrypto WP2 CLI (MVP)")
profile_app = typer.Typer(help="Profile utilities (list/show)")
logger = get_logger(__name__)


@app.callback()
def cli_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable INFO-level logs"),
    debug: bool = typer.Option(False, "--debug", help="Enable DEBUG-level logs (implies verbose)"),
):
    """Global options applied to all subcommands."""
    if ctx.resilient_parsing:
        return
    level_name = resolve_log_level(verbose, debug)
    setup_logging(level_name)
    os.environ["CHAOSCRYPTO_LOG_LEVEL"] = level_name
    set_command_context("cli")
    ctx.obj = ctx.obj or {}
    ctx.obj["log_level"] = level_name


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
    memory_type: str = typer.Option(constants.MEMORY_TYPE, "--memory-type", help="Memory model (opensimplex|perlin)"),
    print_noise_preview: bool = typer.Option(False, "--print-noise-preview", help="Print a small noise field preview"),
    noise_preview_offset: str = typer.Option("0,0", "--noise-preview-offset", help="Preview top-left offset as 'x,y'"),
):
    """Create a profile and persist deterministic memory metadata."""
    set_command_context("init")
    if profile_exists(profile):
        typer.secho(f"Profile '{profile}' already exists.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    from chaoscrypto.core.memory.base import list_memory_models

    if memory_type not in list_memory_models():
        typer.secho(f"Unknown memory type '{memory_type}'. Options: {list_memory_models()}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    params = MemoryParams(type=memory_type, size=size, scale=scale)
    token_bytes = token.encode(constants.ENCODING)
    token_fp = token_fingerprint(token_bytes)
    profile_dir_path = profile_dir(profile)
    profile_meta = profile_meta_path(profile)
    print_run_header(
        "init",
        profile_name=profile,
        profile_dir_path=profile_dir_path,
        profile_files=[profile_meta],
        memory_params=params,
        token_fingerprint=token_fp,
    )

    field, field_fp = build_memory_field(token_bytes, params)
    print_noise_excerpt(field, coord=None, window=2)
    meta = {
        "version": constants.VERSION,
        "memory": {"type": params.type, "size": params.size, "scale": params.scale},
        "field_fingerprint": field_fp,
        "token_fingerprint": token_fp,
    }
    meta_path = save_profile_meta(profile, meta)
    print_io_write(meta_path)
    logger.info("Profile '%s' initialized with memory_type=%s size=%d scale=%s", profile, params.type, params.size, params.scale)
    logger.debug("Stored field fingerprint=%s", field_fp)

    typer.secho(
        f"Profile '{profile}' ready. field_fingerprint={field_fp}",
        fg=typer.colors.GREEN,
    )
    print_done("profile initialized")


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
    seed_strategy: str = typer.Option(constants.SEED_STRATEGY, "--seed-strategy", help="Seed strategy name"),
    memory_type: str | None = typer.Option(None, "--memory-type", help="Memory type (must match profile)"),
    print_noise_preview: bool = typer.Option(False, "--print-noise-preview", help="Print a small noise field preview"),
    noise_preview_offset: str = typer.Option("0,0", "--noise-preview-offset", help="Preview top-left offset as 'x,y'"),
):
    """Encrypt a plaintext file to enc.json with full reproduction metadata."""
    set_command_context("encrypt")
    if not profile_exists(profile):
        typer.secho(f"Profile '{profile}' not found. Run init first.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    meta = load_profile_meta(profile)
    params = memory_params_from_meta(meta)
    if memory_type and memory_type != params.type:
        typer.secho(f"Profile memory type is '{params.type}', but '{memory_type}' was requested.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    token_bytes = token.encode(constants.ENCODING)
    coord_tuple = parse_coord(coord)
    token_fp = token_fingerprint(token_bytes)
    print_run_header(
        "encrypt",
        profile_name=profile,
        profile_dir_path=profile_dir(profile),
        profile_files=[profile_meta_path(profile)],
        memory_params=params,
        token_fingerprint=token_fp,
        seed_strategy=seed_strategy,
        coord=coord_tuple,
        dt=dt,
        warmup=warmup,
        quant_k=quant_k,
    )
    logger.info("Using profile=%s memory_type=%s coord=(%d,%d)", profile, params.type, coord_tuple[0], coord_tuple[1])

    field, field_fp = build_memory_field(token_bytes, params)
    if field_fp != meta["field_fingerprint"]:
        typer.secho("Token or parameters mismatch (field fingerprint differs).", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    print_noise_excerpt(field, coord=coord_tuple, window=2)

    print_io_read(input_path)
    plaintext = input_path.read_bytes()
    print_preview_bytes("plaintext", plaintext, n=16, warning=True)
    if seed_strategy not in list_seed_strategies():
        typer.secho(f"Unknown seed strategy '{seed_strategy}'. Options: {list_seed_strategies()}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    logger.info("Seed strategy=%s dt=%s warmup=%d quant_k=%s", seed_strategy, dt, warmup, quant_k)
    logger.debug("Memory field fingerprint=%s", field_fp)

    ciphertext, computed_fp = encrypt_bytes(
        plaintext,
        token_bytes=token_bytes,
        coord=coord_tuple,
        params=params,
        seed_strategy=seed_strategy,
        dt=dt,
        warmup=warmup,
        quant_k=quant_k,
    )
    assert computed_fp == field_fp  # sanity check
    logger.info("Input size=%d bytes", len(plaintext))
    print_preview_bytes("ciphertext", ciphertext, n=16, warning=True)
    keystream = xor_bytes(plaintext, ciphertext)
    import hashlib

    keystream_sha256 = hashlib.sha256(keystream).hexdigest()
    print_keystream_preview(keystream, n=16, sha256_hex=keystream_sha256)

    enc_payload = {
        "version": constants.VERSION,
        "cipher": "lorenz",
        "memory": {"type": params.type, "size": params.size, "scale": params.scale},
        "seed_strategy": {"name": seed_strategy, "params": {}},
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
    print_io_write(output_path)
    print_cipher_metadata(
        coord=coord_tuple,
        seed_strategy=seed_strategy,
        params=params,
        dt=dt,
        warmup=warmup,
        quant_k=quant_k,
        keystream_sha256=keystream_sha256,
        label="enc.json metadata:",
    )
    logger.info("Encrypted payload written to %s", output_path)
    typer.secho(f"Encrypted → {output_path}", fg=typer.colors.GREEN)
    print_done("encryption complete")


@app.command()
def decrypt(
    profile: str = typer.Option(..., "--profile", "-p", help="Profile name"),
    token: str = typer.Option(..., "--token", "-t", help="Token used for init"),
    input_path: Path = typer.Option(..., "--in", "-i", exists=True, readable=True, help="enc.json input"),
    output_path: Path = typer.Option(..., "--out", "-o", help="Plaintext output path"),
):
    """Decrypt an enc.json produced by the encrypt command."""
    set_command_context("decrypt")
    if not profile_exists(profile):
        typer.secho(f"Profile '{profile}' not found. Run init first.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    enc_payload = read_json(input_path)
    meta = load_profile_meta(profile)
    params = memory_params_from_meta(meta)
    token_bytes = token.encode(constants.ENCODING)
    logger.info("Using profile=%s", profile)

    # Basic metadata validation
    enc_memory = enc_payload.get("memory", {})
    if (
        int(enc_memory.get("size", params.size)) != params.size
        or float(enc_memory.get("scale", params.scale)) != params.scale
        or str(enc_memory.get("type", params.type)) != params.type
    ):
        typer.secho("enc.json memory parameters do not match the profile.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    logger.info(
        "Validating enc.json against profile=%s memory_type=%s coord=(%s,%s)",
        profile,
        enc_memory.get("type", params.type),
        (enc_payload.get("coord") or {}).get("x"),
        (enc_payload.get("coord") or {}).get("y"),
    )

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

    seed_meta = enc_payload.get("seed_strategy")
    if isinstance(seed_meta, dict):
        seed_strategy = seed_meta.get("name", constants.SEED_STRATEGY)
    else:
        seed_strategy = enc_payload.get("seed_strategy", constants.SEED_STRATEGY)
    if seed_strategy not in list_seed_strategies():
        typer.secho(f"Unknown seed strategy '{seed_strategy}' in enc.json.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    logger.info(
        "Loaded enc.json metadata memory_type=%s seed_strategy=%s coord=(%d,%d)",
        enc_memory.get("type", params.type),
        seed_strategy,
        coord_tuple[0],
        coord_tuple[1],
    )
    logger.debug("Expected field fingerprint=%s", expected_fp)

    ciphertext = decode_ciphertext(enc_payload["ciphertext"])
    token_fp = token_fingerprint(token_bytes)
    print_run_header(
        "decrypt",
        profile_name=profile,
        profile_dir_path=profile_dir(profile),
        profile_files=[profile_meta_path(profile)],
        memory_params=params,
        token_fingerprint=token_fp,
        seed_strategy=seed_strategy,
        coord=coord_tuple,
        dt=dt,
        warmup=warmup,
        quant_k=quant_k,
    )
    print_io_read(input_path)

    field, field_fp = build_memory_field(token_bytes, params)
    if expected_fp and field_fp != expected_fp:
        typer.secho("Field fingerprint mismatch – token or parameters differ.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    print_noise_excerpt(field, coord=coord_tuple, window=2)
    init_state = derive_initial_state(field, coord_tuple, seed_strategy=seed_strategy)
    keystream = generate_keystream(len(ciphertext), init_state, dt=dt, warmup=warmup, quant_k=quant_k)
    plaintext = xor_bytes(ciphertext, keystream)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(plaintext)
    print_io_write(output_path)
    print_preview_bytes("recovered_plaintext", plaintext, n=16, warning=True)
    import hashlib

    keystream_sha256 = hashlib.sha256(keystream).hexdigest()
    print_keystream_preview(keystream, n=16, sha256_hex=keystream_sha256)
    print_cipher_metadata(
        coord=coord_tuple,
        seed_strategy=seed_strategy,
        params=params,
        dt=dt,
        warmup=warmup,
        quant_k=quant_k,
        keystream_sha256=keystream_sha256,
        label="enc.json metadata:",
    )
    logger.info("Plaintext written to %s", output_path)
    typer.secho(f"Decrypted → {output_path}", fg=typer.colors.GREEN)
    print_done("decryption complete")


@app.command()
def keystream(
    profile: str = typer.Option(..., "--profile", "-p", help="Profile name"),
    token: str = typer.Option(..., "--token", "-t", help="Token used for init"),
    coord: str = typer.Option(..., "--coord", "-c", help="Coordinate as 'x,y'"),
    nbytes: int = typer.Option(..., "--nbytes", "-n", help="Number of keystream bytes to generate"),
    dt: float = typer.Option(constants.DEFAULT_DT, help="Time step for Lorenz integration"),
    warmup: int = typer.Option(constants.DEFAULT_WARMUP, help="Warmup iterations before sampling"),
    quant_k: float = typer.Option(constants.DEFAULT_QUANT_K, help="Quantization factor for sampling"),
    seed_strategy: str = typer.Option(constants.SEED_STRATEGY, "--seed-strategy", help="Seed strategy name"),
    memory_type: str = typer.Option(None, "--memory-type", help="Memory type override (must match profile)"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write raw keystream bytes to file"),
    hex_out: bool = typer.Option(False, "--hex", help="Write hex to stdout"),
    base64_out: bool = typer.Option(False, "--base64", help="Write base64 to stdout"),
    hash_out: bool = typer.Option(False, "--hash", help="Write SHA-256 hash (hex) to stdout"),
    dump_trajectory: Path | None = typer.Option(None, "--dump-trajectory", help="Write Lorenz trajectory to CSV"),
    plot_trajectory: Path | None = typer.Option(None, "--plot-trajectory", help="Write Lorenz trajectory plot (PNG)"),
    print_noise_preview: bool = typer.Option(False, "--print-noise-preview", help="Print a small noise field preview"),
    noise_preview_offset: str = typer.Option("0,0", "--noise-preview-offset", help="Preview top-left offset as 'x,y'"),
):
    """
    Generate a deterministic keystream (for analysis/benchmarks) without encrypting data.
    """
    set_command_context("keystream")
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
    token_fp = token_fingerprint(token_bytes)

    if memory_type is None:
        memory_type = params.type
    elif memory_type != params.type:
        typer.secho(f"Profile memory type is '{params.type}', but '{memory_type}' was requested.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if seed_strategy not in list_seed_strategies():
        typer.secho(f"Unknown seed strategy '{seed_strategy}'. Options: {list_seed_strategies()}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    params = MemoryParams(type=memory_type, size=params.size, scale=params.scale)
    print_run_header(
        "keystream",
        profile_name=profile,
        profile_dir_path=profile_dir(profile),
        profile_files=[profile_meta_path(profile)],
        memory_params=params,
        token_fingerprint=token_fp,
        seed_strategy=seed_strategy,
        coord=coord_tuple,
        dt=dt,
        warmup=warmup,
        quant_k=quant_k,
        nbytes=nbytes,
    )
    field, field_fp = build_memory_field(token_bytes, params)
    if field_fp != meta["field_fingerprint"]:
        typer.secho("Token or parameters mismatch (field fingerprint differs).", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    print_noise_excerpt(field, coord=coord_tuple, window=2)
    logger.info("Using profile=%s memory_type=%s coord=(%d,%d)", profile, params.type, coord_tuple[0], coord_tuple[1])

    init_state = derive_initial_state(field, coord_tuple, seed_strategy=seed_strategy)
    if dump_trajectory or plot_trajectory:
        from chaoscrypto.core.chaos.lorenz import LorenzSystem
        from chaoscrypto.core.sampling.quantize_byte import QuantizeByteSampling
        import csv

        total_steps = warmup + nbytes
        system = LorenzSystem(
            x=init_state[0],
            y=init_state[1],
            z=init_state[2],
            dt=dt,
            sigma=constants.LCL_SIGMA,
            rho=constants.LCL_RHO,
            beta=constants.LCL_BETA,
        )
        sampler = QuantizeByteSampling(k=quant_k)
        trajectory = []
        ks_buf = bytearray()
        for step in range(total_steps):
            state = system.step()
            phase = "warmup" if step < warmup else "sample"
            trajectory.append((step, phase, state[0], state[1], state[2]))
            if step >= warmup:
                ks_buf.append(sampler.sample(state))
        ks = bytes(ks_buf)

        if dump_trajectory:
            dump_trajectory.parent.mkdir(parents=True, exist_ok=True)
            with dump_trajectory.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["step", "phase", "x", "y", "z"])
                for step, phase, x, y, z in trajectory:
                    writer.writerow([step, phase, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
            print_io_write(dump_trajectory)
            typer.secho(f"Wrote trajectory CSV → {dump_trajectory}", fg=typer.colors.GREEN)

        if plot_trajectory:
            try:
                import matplotlib.pyplot as plt
            except ImportError as exc:
                typer.secho("matplotlib is required for --plot-trajectory", fg=typer.colors.RED)
                raise typer.Exit(code=1) from exc

            steps = [entry[0] for entry in trajectory]
            xs = [entry[2] for entry in trajectory]
            fig, ax = plt.subplots()
            ax.plot(steps, xs, linewidth=1.0)
            ax.axvspan(0, warmup, color="gray", alpha=0.2, label="warmup")
            ax.axvline(warmup, color="red", linestyle="--", linewidth=1.0)
            ax.set_xlabel("step")
            ax.set_ylabel("x")
            ax.set_title(
                "Lorenz trajectory x over steps "
                f"(dt={dt}, warmup={warmup}, coord={coord_tuple[0]},{coord_tuple[1]}, seed_strategy={seed_strategy})"
            )
            ax.legend(loc="best")
            plot_trajectory.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(plot_trajectory)
            plt.close(fig)
            print_io_write(plot_trajectory)
            typer.secho(f"Wrote trajectory plot → {plot_trajectory}", fg=typer.colors.GREEN)
    else:
        ks = generate_keystream(nbytes, init_state, dt=dt, warmup=warmup, quant_k=quant_k)
    import hashlib

    ks_hash = hashlib.sha256(ks).hexdigest()
    logger.info(
        "Generated keystream bytes=%d sha256=%s seed_strategy=%s dt=%s warmup=%d quant_k=%s",
        nbytes,
        ks_hash,
        seed_strategy,
        dt,
        warmup,
        quant_k,
    )
    print_keystream_preview(ks, n=16, sha256_hex=ks_hash)

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(ks)
        print_io_write(out)
        typer.secho(f"Wrote keystream bytes → {out}", fg=typer.colors.GREEN)
        print_done("keystream generated")
        return

    if hex_out:
        typer.echo(ks.hex())
        print_done("keystream generated")
        return

    if base64_out:
        import base64

        typer.echo(base64.b64encode(ks).decode("ascii"))
        print_done("keystream generated")
        return

    if hash_out:
        typer.echo(ks_hash)
        print_done("keystream generated")
        return


@profile_app.command("list")
def profile_list():
    """List available profiles."""
    set_command_context("profile")
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
    set_command_context("profile")
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
    set_command_context("selftest")
    token = b"test-token"
    params = MemoryParams(type=constants.MEMORY_TYPE, size=constants.DEFAULT_MEMORY_SIZE, scale=constants.DEFAULT_MEMORY_SCALE)
    coord = (12, 34)
    plaintext = b"hello"

    ciphertext, field_fp = encrypt_bytes(
        plaintext,
        token_bytes=token,
        coord=coord,
        params=params,
        seed_strategy=constants.SEED_STRATEGY,
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
        seed_strategy=constants.SEED_STRATEGY,
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
    set_command_context("benchmark")
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
    token_fp = token_fingerprint(cfg.bench.token.encode(constants.ENCODING))
    if any(val != profile_params.size for val in cfg.matrix.size):
        typer.secho("Matrix size values must match the profile size.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if any(val != profile_params.scale for val in cfg.matrix.scale):
        typer.secho("Matrix scale values must match the profile scale.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not cfg.matrix.provided_memory_type:
        cfg = BenchFullConfig(
            bench=cfg.bench,
            matrix=BenchMatrixConfig(
                dt=cfg.matrix.dt,
                warmup=cfg.matrix.warmup,
                quant_k=cfg.matrix.quant_k,
                size=cfg.matrix.size,
                scale=cfg.matrix.scale,
                seed_strategy=cfg.matrix.seed_strategy,
                memory_type=[profile_params.type],
                provided_memory_type=False,
            ),
            metrics=cfg.metrics,
            output=cfg.output,
            validate=cfg.validate,
        )

    variant_count = (
        len(cfg.matrix.dt)
        * len(cfg.matrix.warmup)
        * len(cfg.matrix.quant_k)
        * len(cfg.matrix.size)
        * len(cfg.matrix.scale)
        * len(cfg.matrix.seed_strategy)
        * len(cfg.matrix.memory_type)
    )
    run_count = variant_count * cfg.bench.repeats
    seed_strategy_text = cfg.matrix.seed_strategy[0] if len(cfg.matrix.seed_strategy) == 1 else f"matrix[{len(cfg.matrix.seed_strategy)}]"
    print_run_header(
        "benchmark",
        profile_name=cfg.bench.profile,
        profile_dir_path=profile_dir(cfg.bench.profile),
        profile_files=[profile_meta_path(cfg.bench.profile)],
        memory_params=profile_params,
        token_fingerprint=token_fp,
        seed_strategy=seed_strategy_text,
        coord=cfg.bench.coord,
        dt=f"matrix[{len(cfg.matrix.dt)}]",
        warmup=f"matrix[{len(cfg.matrix.warmup)}]",
        quant_k=f"matrix[{len(cfg.matrix.quant_k)}]",
        nbytes=cfg.bench.nbytes,
    )
    print_io_read(config)
    token_bytes = cfg.bench.token.encode(constants.ENCODING)
    field, _field_fp = build_memory_field(token_bytes, profile_params)
    print_noise_excerpt(field, coord=cfg.bench.coord, window=2)
    logger.info(
        "Loaded benchmark config=%s profile=%s variants=%d runs=%d memory_types=%s seed_strategies=%s",
        config,
        cfg.bench.profile,
        variant_count,
        run_count,
        sorted(set(cfg.matrix.memory_type)),
        sorted(set(cfg.matrix.seed_strategy)),
    )

    try:
        records = run_benchmark(cfg, jobs=jobs)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Benchmark failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    print_variant_lines(
        "bench",
        [
            (
                "coord=({coord_x},{coord_y}) repeat={repeat_index} dt={dt} warmup={warmup} quant_k={quant_k} "
                "seed_strategy={seed_strategy} memory_type={memory_type} ks_sha256={keystream_sha256} "
                "throughput_bps={throughput_keystream_bps}"
            ).format(**rec)
            for rec in records
        ],
    )

    try:
        write_csv(out, records)
        if out_json:
            write_json_output(out_json, records)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Failed to write outputs: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    print_io_write(out)
    if out_json:
        print_io_write(out_json)
    logger.info("Benchmark CSV written to %s", out)
    if out_json:
        logger.info("Benchmark JSON written to %s", out_json)

    typer.secho(f"Benchmark complete. CSV → {out}", fg=typer.colors.GREEN)
    if out_json:
        typer.secho(f"JSON → {out_json}", fg=typer.colors.GREEN)
    print_done("benchmark complete")

    if json_summary:
        import json

        summary = {
            "runs": len(records),
            "csv": str(out),
            "json": str(out_json) if out_json else None,
            "variants": len(records),
        }
        typer.echo(json.dumps(summary))


@app.command()
def analyze(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True, help="YAML analyze config"),
    out: Path = typer.Option(..., "--out", "-o", help="CSV output path"),
    out_json: Path | None = typer.Option(None, "--out-json", help="Optional JSON output path"),
    jobs: int = typer.Option(1, "--jobs", "-j", help="Parallel jobs (variants), default 1"),
    json_summary: bool = typer.Option(False, "--json", help="Print summary JSON to stdout"),
):
    """
    Analyze keystream statistics based on a YAML config (matrix-driven).
    """
    set_command_context("analyze")
    try:
        cfg = parse_analyze_config(config)
    except AnalyzeConfigError as exc:
        typer.secho(f"Config error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if not profile_exists(cfg.analyze.profile):
        typer.secho(f"Profile '{cfg.analyze.profile}' not found. Run init first.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    profile_meta = load_profile_meta(cfg.analyze.profile)
    profile_params = memory_params_from_meta(profile_meta)
    token_fp = token_fingerprint(cfg.analyze.token.encode(constants.ENCODING))
    if any(val != profile_params.size for val in cfg.matrix.size):
        typer.secho("Matrix size values must match the profile size.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if any(val != profile_params.scale for val in cfg.matrix.scale):
        typer.secho("Matrix scale values must match the profile scale.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not cfg.matrix.provided_memory_type:
        cfg = AnalyzeFullConfig(
            analyze=cfg.analyze,
            matrix=AnalyzeMatrixConfig(
                dt=cfg.matrix.dt,
                warmup=cfg.matrix.warmup,
                quant_k=cfg.matrix.quant_k,
                size=cfg.matrix.size,
                scale=cfg.matrix.scale,
                seed_strategy=cfg.matrix.seed_strategy,
                memory_type=[profile_params.type],
                provided_memory_type=False,
            ),
            metrics=cfg.metrics,
            output=cfg.output,
            validate=cfg.validate,
        )

    variant_count = (
        len(cfg.analyze.coords)
        * len(cfg.matrix.dt)
        * len(cfg.matrix.warmup)
        * len(cfg.matrix.quant_k)
        * len(cfg.matrix.size)
        * len(cfg.matrix.scale)
        * len(cfg.matrix.seed_strategy)
        * len(cfg.matrix.memory_type)
    )
    seed_strategy_text = cfg.matrix.seed_strategy[0] if len(cfg.matrix.seed_strategy) == 1 else f"matrix[{len(cfg.matrix.seed_strategy)}]"
    coord_text = cfg.analyze.coords[0] if len(cfg.analyze.coords) == 1 else None
    print_run_header(
        "analyze",
        profile_name=cfg.analyze.profile,
        profile_dir_path=profile_dir(cfg.analyze.profile),
        profile_files=[profile_meta_path(cfg.analyze.profile)],
        memory_params=profile_params,
        token_fingerprint=token_fp,
        seed_strategy=seed_strategy_text,
        coord=coord_text,
        dt=f"matrix[{len(cfg.matrix.dt)}]",
        warmup=f"matrix[{len(cfg.matrix.warmup)}]",
        quant_k=f"matrix[{len(cfg.matrix.quant_k)}]",
        nbytes=cfg.analyze.nbytes,
    )
    print_io_read(config)
    token_bytes = cfg.analyze.token.encode(constants.ENCODING)
    field, _field_fp = build_memory_field(token_bytes, profile_params)
    preview_coord = cfg.analyze.coords[0] if cfg.analyze.coords else None
    print_noise_excerpt(field, coord=preview_coord, window=2)
    logger.info(
        "Loaded analyze config=%s profile=%s variants=%d memory_types=%s seed_strategies=%s",
        config,
        cfg.analyze.profile,
        variant_count,
        sorted(set(cfg.matrix.memory_type)),
        sorted(set(cfg.matrix.seed_strategy)),
    )

    try:
        records = run_analyze(cfg, jobs=jobs)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Analyze failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    print_variant_lines(
        "analyze",
        [
            (
                "coord=({coord_x},{coord_y}) dt={dt} warmup={warmup} quant_k={quant_k} seed_strategy={seed_strategy} "
                "memory_type={memory_type} ks_sha256={keystream_sha256} bit_ones_ratio={bit_ones_ratio}"
            ).format(**rec)
            for rec in records
        ],
    )

    max_autocorr = cfg.metrics.autocorr_bits_max_lag if cfg.metrics.autocorr_bits_enabled else 0
    try:
        write_analyze_csv(out, records, max_autocorr=max_autocorr)
        if out_json:
            write_analyze_json(out_json, records)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Failed to write outputs: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    print_io_write(out)
    if out_json:
        print_io_write(out_json)
    typer.secho(f"Analyze complete. CSV → {out}", fg=typer.colors.GREEN)
    if out_json:
        typer.secho(f"JSON → {out_json}", fg=typer.colors.GREEN)
    logger.info("Analyze CSV written to %s", out)
    if out_json:
        logger.info("Analyze JSON written to %s", out_json)
    print_done("analysis complete")
    if json_summary:
        import json

        summary = {
            "runs": len(records),
            "csv": str(out),
            "json": str(out_json) if out_json else None,
            "variants": len(records),
        }
        typer.echo(json.dumps(summary))


@app.command()
def report(
    bench_csv: Path = typer.Option(..., "--bench-csv", exists=True, readable=True, help="Benchmark CSV path"),
    analysis_csv: Path = typer.Option(..., "--analysis-csv", exists=True, readable=True, help="Analyze CSV path"),
    out: Path = typer.Option(..., "--out", "-o", help="Output markdown report path"),
    bench_json: Path | None = typer.Option(None, "--bench-json", help="Optional benchmark JSON path"),
    analysis_json: Path | None = typer.Option(None, "--analysis-json", help="Optional analysis JSON path"),
    plots_dir: Path | None = typer.Option(None, "--plots-dir", help="Optional directory for PNG plots"),
    json_summary: Path | None = typer.Option(None, "--json-summary", help="Write summary JSON to path"),
    no_timestamp: bool = typer.Option(False, "--no-timestamp", help="Omit timestamp for deterministic reports"),
    json_flag: bool = typer.Option(False, "--json", help="Print summary JSON to stdout"),
    plot_mode: str = typer.Option("condensed", "--plot-mode", help="Plot mode: condensed or matrix"),
):
    """Generate a markdown report (and optional plots) from benchmark/analyze outputs."""
    set_command_context("report")
    print_run_header(
        "report",
        profile_name=None,
        profile_dir_path=None,
        profile_files=None,
        memory_params=None,
        token_fingerprint=None,
    )
    print_io_read(bench_csv)
    print_io_read(analysis_csv)
    if bench_json:
        print_io_read(bench_json)
    if analysis_json:
        print_io_read(analysis_json)
    logger.info(
        "Generating report from bench_csv=%s analysis_csv=%s plots_dir=%s plot_mode=%s",
        bench_csv,
        analysis_csv,
        plots_dir,
        plot_mode,
    )
    try:
        summary = generate_report(
            bench_csv=bench_csv,
            analysis_csv=analysis_csv,
            bench_json=bench_json,
            analysis_json=analysis_json,
            out_md=out,
            plots_dir=plots_dir,
            include_timestamp=not no_timestamp,
            json_summary=json_summary,
            plot_mode=plot_mode,
        )
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Report failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    logger.info(
        "Report written to %s (bench_variants=%s, analyze_variants=%s)",
        out,
        summary.get("bench_variants"),
        summary.get("analyze_variants"),
    )

    print_io_write(out)
    typer.secho(f"Report written → {out}", fg=typer.colors.GREEN)
    if plots_dir:
        print_io_write(plots_dir)
        typer.secho(f"Plots in → {plots_dir}", fg=typer.colors.GREEN)
    if json_summary:
        print_io_write(json_summary)
        typer.secho(f"Summary JSON → {json_summary}", fg=typer.colors.GREEN)
    print_done("report complete")
    if json_flag:
        import json

        typer.echo(json.dumps(summary))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
