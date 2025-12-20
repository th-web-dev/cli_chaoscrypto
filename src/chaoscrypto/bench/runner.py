from __future__ import annotations

import base64
import csv
import hashlib
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from chaoscrypto.core import constants
from chaoscrypto.core.memory.base import MemoryParams, MemoryModel
from chaoscrypto.io.profiles import load_profile_meta, memory_params_from_meta, token_fingerprint
from chaoscrypto.orchestrator.pipeline import (
    build_memory_field,
    derive_initial_state,
    generate_keystream,
)
from chaoscrypto.core.seed.base import get_seed_strategy, list_seed_strategies


# -------------------------
# Config structures
# -------------------------


@dataclass(frozen=True)
class BenchConfig:
    profile: str
    token: str
    coord: Tuple[int, int]
    nbytes: int
    repeats: int
    field_regen_each_repeat: bool


@dataclass(frozen=True)
class MatrixConfig:
    dt: Sequence[float]
    warmup: Sequence[int]
    quant_k: Sequence[float]
    size: Sequence[int]
    scale: Sequence[float]
    seed_strategy: Sequence[str]


@dataclass(frozen=True)
class MetricsConfig:
    include_field_time: bool
    include_xor_time: bool
    include_decrypt_time: bool
    keystream_hash: str  # only sha256 supported


@dataclass(frozen=True)
class OutputConfig:
    include_timestamp_utc: bool
    include_field_fingerprint: bool
    include_keystream_preview: bool
    keystream_preview_bytes: int


@dataclass(frozen=True)
class ValidateConfig:
    assert_deterministic_within_run: bool
    extra_determinism_check: bool


@dataclass(frozen=True)
class FullConfig:
    bench: BenchConfig
    matrix: MatrixConfig
    metrics: MetricsConfig
    output: OutputConfig
    validate: ValidateConfig


# -------------------------
# Config parsing/validation
# -------------------------


class ConfigError(Exception):
    """Raised when the benchmark config is invalid."""


def _require(mapping: Dict[str, Any], key: str, expected_type: Tuple[type, ...]):
    if key not in mapping:
        raise ConfigError(f"Missing required key '{key}'")
    val = mapping[key]
    if not isinstance(val, expected_type):
        raise ConfigError(f"Key '{key}' must be of type {expected_type}, got {type(val)}")
    return val


def parse_config(path: Path) -> FullConfig:
    try:
        data = yaml.safe_load(path.read_text())
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Failed to read YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Top-level YAML must be a mapping.")

    bench = _require(data, "bench", (dict,))
    matrix = _require(data, "matrix", (dict,))
    metrics = _require(data, "metrics", (dict,))
    output = _require(data, "output", (dict,))
    validate = _require(data, "validate", (dict,))

    bench_cfg = BenchConfig(
        profile=_require(bench, "profile", (str,)),
        token=_require(bench, "token", (str,)),
        coord=tuple(_require(bench, "coord", (list, tuple))),
        nbytes=int(_require(bench, "nbytes", (int, float))),
        repeats=int(_require(bench, "repeats", (int, float))),
        field_regen_each_repeat=bool(bench.get("field_regen_each_repeat", True)),
    )
    if len(bench_cfg.coord) != 2:
        raise ConfigError("bench.coord must have exactly two entries (x,y)")
    bench_cfg = BenchConfig(
        profile=bench_cfg.profile,
        token=bench_cfg.token,
        coord=(int(bench_cfg.coord[0]), int(bench_cfg.coord[1])),
        nbytes=bench_cfg.nbytes,
        repeats=bench_cfg.repeats,
        field_regen_each_repeat=bench_cfg.field_regen_each_repeat,
    )

    matrix_cfg = MatrixConfig(
        dt=[float(x) for x in _require(matrix, "dt", (list, tuple))],
        warmup=[int(x) for x in _require(matrix, "warmup", (list, tuple))],
        quant_k=[float(x) for x in _require(matrix, "quant_k", (list, tuple))],
        size=[int(x) for x in matrix.get("size", [constants.DEFAULT_MEMORY_SIZE])],
        scale=[float(x) for x in matrix.get("scale", [constants.DEFAULT_MEMORY_SCALE])],
        seed_strategy=[str(x) for x in matrix.get("seed_strategy", [constants.SEED_STRATEGY])],
    )

    metrics_cfg = MetricsConfig(
        include_field_time=bool(metrics.get("include_field_time", True)),
        include_xor_time=bool(metrics.get("include_xor_time", True)),
        include_decrypt_time=bool(metrics.get("include_decrypt_time", False)),
        keystream_hash=str(metrics.get("keystream_hash", "sha256")),
    )
    if metrics_cfg.keystream_hash.lower() != "sha256":
        raise ConfigError("Only sha256 keystream_hash is supported.")

    output_cfg = OutputConfig(
        include_timestamp_utc=bool(output.get("include_timestamp_utc", True)),
        include_field_fingerprint=bool(output.get("include_field_fingerprint", True)),
        include_keystream_preview=bool(output.get("include_keystream_preview", False)),
        keystream_preview_bytes=int(output.get("keystream_preview_bytes", 64)),
    )

    validate_cfg = ValidateConfig(
        assert_deterministic_within_run=bool(validate.get("assert_deterministic_within_run", True)),
        extra_determinism_check=bool(validate.get("extra_determinism_check", False)),
    )

    for name in matrix_cfg.seed_strategy:
        if name not in list_seed_strategies():
            raise ConfigError(f"Unknown seed_strategy '{name}'. Available: {list_seed_strategies()}")

    return FullConfig(
        bench=bench_cfg,
        matrix=matrix_cfg,
        metrics=metrics_cfg,
        output=output_cfg,
        validate=validate_cfg,
    )


# -------------------------
# Benchmark internals
# -------------------------


def _deterministic_plaintext(nbytes: int) -> bytes:
    return bytes((i % 256 for i in range(nbytes)))


def _generate_field(token_bytes: bytes, params: MemoryParams) -> Tuple[np.ndarray, str]:
    field, field_fp = build_memory_field(token_bytes, params)
    return field, field_fp


def _measure_time(func):
    start = time.perf_counter()
    result = func()
    end = time.perf_counter()
    return result, end - start


def _keystream_and_metrics(
    params: MemoryParams,
    token_bytes: bytes,
    coord: Tuple[int, int],
    nbytes: int,
    dt: float,
    warmup: int,
    quant_k: float,
    include_field_time: bool,
    include_xor_time: bool,
    include_decrypt_time: bool,
    field_regen_each: bool,
    assert_deterministic: bool,
    include_field_fp: bool,
    include_preview: bool,
    preview_len: int,
    precomputed_field: Tuple[np.ndarray, str] | None = None,
    precomputed_field_time: float | None = None,
    seed_strategy: str = constants.SEED_STRATEGY,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    # Field
    if precomputed_field:
        field, field_fp = precomputed_field
        t_field = precomputed_field_time
    elif include_field_time or field_regen_each:
        (field, field_fp), t_field = _measure_time(lambda: _generate_field(token_bytes, params))
    else:
        field, field_fp = _generate_field(token_bytes, params)
        t_field = None
    result["field_fingerprint"] = field_fp if include_field_fp else None
    result["t_field_s"] = t_field

    # Seed / Keystream
    def gen_ks():
        init_state = derive_initial_state(field, coord, seed_strategy=seed_strategy)
        return generate_keystream(
            num_bytes=nbytes,
            init_state=init_state,
            dt=dt,
            warmup=warmup,
            quant_k=quant_k,
        )

    ks, t_keystream = _measure_time(gen_ks)

    if assert_deterministic:
        ks2 = generate_keystream(
            num_bytes=nbytes,
            init_state=derive_initial_state(field, coord, seed_strategy=seed_strategy),
            dt=dt,
            warmup=warmup,
            quant_k=quant_k,
        )
        if ks2 != ks:
            raise RuntimeError("Determinism check failed: keystream mismatch within run.")

    result["t_keystream_s"] = t_keystream

    # XOR encrypt
    plaintext = _deterministic_plaintext(nbytes)
    if include_xor_time:
        (ciphertext, _), t_xor = _measure_time(
            lambda: (bytes(p ^ k for p, k in zip(plaintext, ks)), None)
        )
        result["t_xor_s"] = t_xor
    else:
        ciphertext = bytes(p ^ k for p, k in zip(plaintext, ks))
        result["t_xor_s"] = None

    if include_decrypt_time:
        (_, _), t_dec = _measure_time(
            lambda: (bytes(c ^ k for c, k in zip(ciphertext, ks)), None)
        )
        result["t_decrypt_s"] = t_dec
    else:
        result["t_decrypt_s"] = None

    result["keystream_sha256"] = hashlib.sha256(ks).hexdigest()
    if include_preview:
        result["keystream_preview_base64"] = base64.b64encode(ks[:preview_len]).decode("ascii")
    else:
        result["keystream_preview_base64"] = None

    result["throughput_keystream_bps"] = nbytes / t_keystream if t_keystream else None
    if include_xor_time and result["t_xor_s"]:
        result["throughput_xor_bps"] = nbytes / result["t_xor_s"]
    else:
        result["throughput_xor_bps"] = None

    return result


def _variant_product(matrix: MatrixConfig) -> List[Dict[str, Any]]:
    combos = []
    for dt, warmup, quant_k, size, scale, seed_strategy in itertools.product(
        matrix.dt, matrix.warmup, matrix.quant_k, matrix.size, matrix.scale, matrix.seed_strategy
    ):
        combos.append(
            {
                "dt": float(dt),
                "warmup": int(warmup),
                "quant_k": float(quant_k),
                "size": int(size),
                "scale": float(scale),
                "seed_strategy": str(seed_strategy),
            }
        )
    return combos


def _run_single_variant(
    bench: BenchConfig,
    metrics: MetricsConfig,
    output: OutputConfig,
    validate: ValidateConfig,
    variant: Dict[str, Any],
    repeat_index: int,
) -> Dict[str, Any]:
    params = MemoryParams(
        type=constants.MEMORY_TYPE,
        size=variant["size"],
        scale=variant["scale"],
    )
    token_bytes = bench.token.encode(constants.ENCODING)
    token_fp = token_fingerprint(token_bytes)

    precomputed_field = None
    precomputed_time = None
    if bench.field_regen_each_repeat:
        field, field_fp = _generate_field(token_bytes, params)
    else:
        if metrics.include_field_time:
            (field, field_fp), measured = _measure_time(lambda: _generate_field(token_bytes, params))
        else:
            field, field_fp = _generate_field(token_bytes, params)
            measured = None
        precomputed_field = (field, field_fp)
        precomputed_time = measured

    # Validate seed strategy
    if variant["seed_strategy"] not in list_seed_strategies():
        raise ConfigError(f"Unknown seed_strategy '{variant['seed_strategy']}'. Available: {list_seed_strategies()}")

    ks_metrics = _keystream_and_metrics(
        params=params,
        token_bytes=token_bytes,
        coord=bench.coord,
        nbytes=bench.nbytes,
        dt=variant["dt"],
        warmup=variant["warmup"],
        quant_k=variant["quant_k"],
        include_field_time=metrics.include_field_time,
        include_xor_time=metrics.include_xor_time,
        include_decrypt_time=metrics.include_decrypt_time,
        field_regen_each=bench.field_regen_each_repeat,
        assert_deterministic=validate.assert_deterministic_within_run,
        include_field_fp=output.include_field_fingerprint,
        include_preview=output.include_keystream_preview,
        preview_len=output.keystream_preview_bytes,
        precomputed_field=precomputed_field,
        precomputed_field_time=precomputed_time,
        seed_strategy=variant["seed_strategy"],
    )

    record: Dict[str, Any] = {
        "profile": bench.profile,
        "coord_x": bench.coord[0],
        "coord_y": bench.coord[1],
        "nbytes": bench.nbytes,
        "repeats": bench.repeats,
        "repeat_index": repeat_index,
        "dt": variant["dt"],
        "warmup": variant["warmup"],
        "quant_k": variant["quant_k"],
        "size": variant["size"],
        "scale": variant["scale"],
        "seed_strategy": variant["seed_strategy"],
        "keystream_sha256": ks_metrics["keystream_sha256"],
        "token_fingerprint": token_fp,
        "field_fingerprint": field_fp if output.include_field_fingerprint else None,
        "t_field_s": ks_metrics["t_field_s"],
        "t_keystream_s": ks_metrics["t_keystream_s"],
        "t_xor_s": ks_metrics["t_xor_s"],
        "t_decrypt_s": ks_metrics["t_decrypt_s"],
        "throughput_keystream_bps": ks_metrics["throughput_keystream_bps"],
        "throughput_xor_bps": ks_metrics["throughput_xor_bps"],
        "keystream_preview_base64": ks_metrics["keystream_preview_base64"],
    }
    if output.include_timestamp_utc:
        record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    return record


def run_benchmark(config: FullConfig, jobs: int = 1) -> List[Dict[str, Any]]:
    variants = _variant_product(config.matrix)
    tasks: List[Tuple[Dict[str, Any], int]] = []
    for variant in variants:
        for repeat_index in range(config.bench.repeats):
            tasks.append((variant, repeat_index))

    def runner(task: Tuple[Dict[str, Any], int]) -> Dict[str, Any]:
        variant, repeat_index = task
        return _run_single_variant(
            bench=config.bench,
            metrics=config.metrics,
            output=config.output,
            validate=config.validate,
            variant=variant,
            repeat_index=repeat_index,
        )

    if jobs and jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(runner, tasks))
    else:
        results = [runner(task) for task in tasks]

    # Sort deterministically
    def sort_key(rec: Dict[str, Any]):
        return (
            rec["dt"],
            rec["warmup"],
            rec["quant_k"],
            rec["size"],
            rec["scale"],
            rec.get("seed_strategy"),
            rec["repeat_index"],
        )

    results_sorted = sorted(results, key=sort_key)
    return results_sorted


# -------------------------
# Output helpers
# -------------------------


CSV_FIELDS = [
    "timestamp_utc",
    "profile",
    "coord_x",
    "coord_y",
    "nbytes",
    "repeats",
    "repeat_index",
    "dt",
    "warmup",
    "quant_k",
    "size",
    "scale",
    "seed_strategy",
    "t_field_s",
    "t_keystream_s",
    "t_xor_s",
    "t_decrypt_s",
    "throughput_keystream_bps",
    "throughput_xor_bps",
    "keystream_sha256",
    "field_fingerprint",
    "token_fingerprint",
    "keystream_preview_base64",
]


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def write_json_output(path: Path, records: List[Dict[str, Any]]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
