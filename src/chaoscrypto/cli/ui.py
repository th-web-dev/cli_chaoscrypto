from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import typer

from chaoscrypto.core.memory.base import MemoryParams


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _abs_path(path: Path | None) -> str:
    if path is None:
        return "n/a"
    try:
        return str(path.resolve())
    except Exception:  # noqa: BLE001
        return str(path)


def _truncate_hex(value: str, max_len: int = 12) -> str:
    if len(value) <= max_len:
        return value
    return f"{value[:max_len]}..."


def print_run_header(
    command: str,
    *,
    profile_name: str | None,
    profile_dir_path: Path | None,
    profile_files: Sequence[Path] | None,
    memory_params: MemoryParams | None,
    token_fingerprint: str | None,
    seed_strategy: str | None = None,
    coord: tuple[int, int] | None = None,
    dt: float | str | None = None,
    warmup: int | str | None = None,
    quant_k: float | str | None = None,
    nbytes: int | str | None = None,
) -> None:
    typer.echo(f"[run] command={command} ts_utc={_timestamp_utc()}")
    profile_name = profile_name or "n/a"
    profile_dir = _abs_path(profile_dir_path)
    files = "n/a"
    if profile_files:
        files = ", ".join(_abs_path(path) for path in profile_files)
    typer.echo(f"[profile] name={profile_name} dir={profile_dir} files={files}")
    if token_fingerprint:
        typer.echo(f"[profile] token_fingerprint={_truncate_hex(token_fingerprint)}")
    if memory_params:
        typer.echo(f"[memory] type={memory_params.type} size={memory_params.size} scale={memory_params.scale}")
    else:
        typer.echo("[memory] type=n/a size=n/a scale=n/a")
    if seed_strategy is not None or coord is not None:
        coord_text = f"({coord[0]},{coord[1]})" if coord else "n/a"
        seed_text = seed_strategy or "n/a"
        typer.echo(f"[seed] strategy={seed_text} coord={coord_text}")
    if dt is not None or warmup is not None or quant_k is not None:
        dt_text = dt if dt is not None else "n/a"
        warmup_text = warmup if warmup is not None else "n/a"
        quant_text = quant_k if quant_k is not None else "n/a"
        typer.echo(f"[lorenz] dt={dt_text} warmup={warmup_text} quant_k={quant_text}")
    if nbytes is not None:
        typer.echo(f"[keystream] nbytes={nbytes}")


def print_io_read(path: Path, size_bytes: int | None = None) -> None:
    if size_bytes is None:
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = None
    size_text = "unknown" if size_bytes is None else str(size_bytes)
    typer.echo(f"[io] Reading input: {_abs_path(path)} (bytes={size_text})")


def print_io_write(path: Path) -> None:
    typer.echo(f"[io] Writing output: {_abs_path(path)}")


def print_noise_excerpt(field, coord: tuple[int, int] | None, window: int = 2) -> None:
    size = field.shape[0]
    if coord:
        cx, cy = coord
    else:
        cx, cy = 0, 0

    span = window * 2 + 1
    if size <= span:
        x_start, x_end = 0, size - 1
    else:
        x_start = max(0, min(cx - window, size - span))
        x_end = x_start + span - 1

    if size <= span:
        y_start, y_end = 0, size - 1
    else:
        y_start = max(0, min(cy - window, size - span))
        y_end = y_start + span - 1

    typer.echo(
        f"[preview] noise center=({cx},{cy}) window={window} x={x_start}..{x_end} y={y_start}..{y_end}"
    )
    for x in range(x_start, x_end + 1):
        cells = []
        for y in range(y_start, y_end + 1):
            cells.append(f"({x},{y})={float(field[x, y]):.3f}")
        typer.echo(f"[preview] x={x}: " + " ".join(cells))


def print_keystream_preview(keystream: bytes, n: int = 16, sha256_hex: str | None = None) -> str:
    preview = keystream[:n]
    if sha256_hex is None:
        sha256_hex = hashlib.sha256(keystream).hexdigest()
    preview_hex = preview.hex()
    preview_b64 = base64.b64encode(preview).decode("ascii")
    typer.echo(
        f"[keystream] preview_hex={preview_hex} preview_b64={preview_b64} total_bytes={len(keystream)} sha256={sha256_hex}"
    )
    return sha256_hex


def print_cipher_metadata(
    *,
    coord: tuple[int, int],
    seed_strategy: str,
    params: MemoryParams,
    dt: float,
    warmup: int,
    quant_k: float,
    keystream_sha256: str,
    label: str,
) -> None:
    typer.echo(
        "[io] "
        f"{label} coord=({coord[0]},{coord[1]}) seed_strategy={seed_strategy} "
        f"memory_type={params.type} size={params.size} scale={params.scale} "
        f"dt={dt} warmup={warmup} quant_k={quant_k} keystream_sha256={keystream_sha256}"
    )


def print_preview_bytes(label: str, data: bytes, n: int = 16, warning: bool = False) -> None:
    preview_hex = data[:n].hex()
    suffix = " (preview only)" if warning else ""
    typer.echo(f"[preview] {label}_hex={preview_hex}{suffix}")


def print_variant_lines(tag: str, lines: Iterable[str]) -> None:
    for line in lines:
        typer.echo(f"[{tag}] {line}")


def print_done(summary: str) -> None:
    typer.echo(f"[done] {summary}")
