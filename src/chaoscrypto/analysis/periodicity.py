from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from chaoscrypto.analysis.runner import FullConfig
from chaoscrypto.core import constants
from chaoscrypto.core.memory.base import MemoryParams
from chaoscrypto.core.seed.base import list_seed_strategies
from chaoscrypto.io.profiles import token_fingerprint
from chaoscrypto.orchestrator.pipeline import build_memory_field, derive_initial_state, generate_keystream
from chaoscrypto.utils.logging import get_logger, set_command_context, setup_logging

logger = get_logger(__name__)

_FIELD_CACHE: Dict[Tuple[bytes, str, int, float], Tuple[np.ndarray, str]] = {}

PERIODICITY_FIELDS = [
    "timestamp_utc",
    "profile",
    "coord_x",
    "coord_y",
    "nbytes",
    "dt",
    "warmup",
    "quant_k",
    "size",
    "scale",
    "seed_strategy",
    "memory_type",
    "chaos_engine",
    "token_fingerprint",
    "field_fingerprint",
    "keystream_sha256",
    "chunk_size_bytes",
    "chunk_count",
    "unique_chunk_hashes",
    "repeated_chunk_hash_count",
    "max_chunk_hash_frequency",
    "most_common_chunk_hash",
    "lag_step_bytes",
    "lag_match_ratio",
    "detected_prefix_period_bytes",
]


def _variant_product(config: FullConfig) -> List[Dict[str, Any]]:
    combos: List[Dict[str, Any]] = []
    for coord in config.analyze.coords:
        for dt in config.matrix.dt:
            for warmup in config.matrix.warmup:
                for quant_k in config.matrix.quant_k:
                    for size in config.matrix.size:
                        for scale in config.matrix.scale:
                            for seed_strategy in config.matrix.seed_strategy:
                                for memory_type in config.matrix.memory_type:
                                    for chaos_engine in config.matrix.chaos_engine:
                                        combos.append(
                                            {
                                                "coord": coord,
                                                "dt": float(dt),
                                                "warmup": int(warmup),
                                                "quant_k": float(quant_k),
                                                "size": int(size),
                                                "scale": float(scale),
                                                "seed_strategy": str(seed_strategy or constants.SEED_STRATEGY),
                                                "memory_type": str(memory_type or constants.MEMORY_TYPE),
                                                "chaos_engine": str(chaos_engine or constants.CHAOS_ENGINE),
                                            }
                                        )
    return combos


def _get_field(token_bytes: bytes, params: MemoryParams) -> Tuple[np.ndarray, str]:
    cache_key = (token_bytes, params.type, params.size, params.scale)
    cached = _FIELD_CACHE.get(cache_key)
    if cached is None:
        cached = build_memory_field(token_bytes, params)
        _FIELD_CACHE[cache_key] = cached
    return cached


def _chunk_hash_stats(data: bytes, chunk_size: int) -> Dict[str, Any]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size) if data[i : i + chunk_size]]
    hashes = [hashlib.sha256(ch).hexdigest() for ch in chunks]
    counts = Counter(hashes)
    unique = len(counts)
    repeated = sum(1 for v in counts.values() if v > 1)
    max_freq = max(counts.values()) if counts else 0
    most_common = counts.most_common(1)[0][0] if counts else None
    return {
        "chunk_count": len(chunks),
        "unique_chunk_hashes": unique,
        "repeated_chunk_hash_count": repeated,
        "max_chunk_hash_frequency": max_freq,
        "most_common_chunk_hash": most_common,
    }


def _lag_match_ratio(data: bytes, lag: int) -> float:
    if lag <= 0 or len(data) <= lag:
        return 0.0
    a = np.frombuffer(data[:-lag], dtype=np.uint8)
    b = np.frombuffer(data[lag:], dtype=np.uint8)
    return float((a == b).sum() / a.size) if a.size else 0.0


def _detect_exact_prefix_period(data: bytes, max_period: int) -> int | None:
    n = len(data)
    if n < 2:
        return None
    upper = min(max_period, n // 2)
    for p in range(1, upper + 1):
        prefix = data[:p]
        matches = True
        for i in range(p, n):
            if data[i] != prefix[i % p]:
                matches = False
                break
        if matches:
            return p
    return None


def _run_periodicity_one(
    config: FullConfig,
    variant: Dict[str, Any],
    token_bytes: bytes,
    token_fp: str,
    chunk_size_bytes: int,
    lag_step_bytes: int,
    max_detect_period_bytes: int,
) -> Dict[str, Any]:
    params = MemoryParams(type=variant["memory_type"], size=variant["size"], scale=variant["scale"])
    if variant["seed_strategy"] not in list_seed_strategies():
        raise ValueError(f"Unknown seed_strategy '{variant['seed_strategy']}'. Available: {list_seed_strategies()}")

    field, field_fp = _get_field(token_bytes, params)
    init_state = derive_initial_state(field, variant["coord"], seed_strategy=variant["seed_strategy"])
    ks = generate_keystream(
        config.analyze.nbytes,
        init_state,
        dt=variant["dt"],
        warmup=variant["warmup"],
        quant_k=variant["quant_k"],
        chaos_engine=variant["chaos_engine"],
    )

    if config.validate.assert_deterministic_within_run:
        init_state_2 = derive_initial_state(field, variant["coord"], seed_strategy=variant["seed_strategy"])
        ks2 = generate_keystream(
            config.analyze.nbytes,
            init_state_2,
            dt=variant["dt"],
            warmup=variant["warmup"],
            quant_k=variant["quant_k"],
            chaos_engine=variant["chaos_engine"],
        )
        if ks2 != ks:
            raise RuntimeError("Determinism check failed during periodicity run.")

    chunk_stats = _chunk_hash_stats(ks, chunk_size=chunk_size_bytes)
    lag_ratio = _lag_match_ratio(ks, lag=lag_step_bytes)
    detected_period = _detect_exact_prefix_period(ks, max_period=max_detect_period_bytes)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat() if config.output.include_timestamp_utc else None,
        "profile": config.analyze.profile,
        "coord_x": variant["coord"][0],
        "coord_y": variant["coord"][1],
        "nbytes": config.analyze.nbytes,
        "dt": variant["dt"],
        "warmup": variant["warmup"],
        "quant_k": variant["quant_k"],
        "size": variant["size"],
        "scale": variant["scale"],
        "seed_strategy": variant["seed_strategy"],
        "memory_type": variant["memory_type"],
        "chaos_engine": variant["chaos_engine"],
        "token_fingerprint": token_fp,
        "field_fingerprint": field_fp,
        "keystream_sha256": hashlib.sha256(ks).hexdigest(),
        "chunk_size_bytes": chunk_size_bytes,
        "chunk_count": chunk_stats["chunk_count"],
        "unique_chunk_hashes": chunk_stats["unique_chunk_hashes"],
        "repeated_chunk_hash_count": chunk_stats["repeated_chunk_hash_count"],
        "max_chunk_hash_frequency": chunk_stats["max_chunk_hash_frequency"],
        "most_common_chunk_hash": chunk_stats["most_common_chunk_hash"],
        "lag_step_bytes": lag_step_bytes,
        "lag_match_ratio": lag_ratio,
        "detected_prefix_period_bytes": detected_period,
    }


def _run_periodicity_task(
    args: Tuple[FullConfig, Dict[str, Any], bytes, str, int, int, int],
) -> Dict[str, Any]:
    config, variant, token_bytes, token_fp, chunk_size_bytes, lag_step_bytes, max_detect_period_bytes = args
    setup_logging(os.environ.get("CHAOSCRYPTO_LOG_LEVEL", "WARNING"))
    set_command_context("periodicity")
    return _run_periodicity_one(
        config=config,
        variant=variant,
        token_bytes=token_bytes,
        token_fp=token_fp,
        chunk_size_bytes=chunk_size_bytes,
        lag_step_bytes=lag_step_bytes,
        max_detect_period_bytes=max_detect_period_bytes,
    )


def run_periodicity(
    config: FullConfig,
    jobs: int = 1,
    chunk_size_bytes: int = 4096,
    lag_step_bytes: int = 4096,
    max_detect_period_bytes: int = 8192,
) -> List[Dict[str, Any]]:
    setup_logging(os.environ.get("CHAOSCRYPTO_LOG_LEVEL", "WARNING"))
    set_command_context("periodicity")
    variants = _variant_product(config)
    token_bytes = config.analyze.token.encode(constants.ENCODING)
    token_fp = token_fingerprint(token_bytes)

    if jobs and jobs > 1:
        task_args = [
            (config, variant, token_bytes, token_fp, chunk_size_bytes, lag_step_bytes, max_detect_period_bytes)
            for variant in variants
        ]
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            records = list(ex.map(_run_periodicity_task, task_args))
    else:
        records = [
            _run_periodicity_task((config, variant, token_bytes, token_fp, chunk_size_bytes, lag_step_bytes, max_detect_period_bytes))
            for variant in variants
        ]

    return sorted(
        records,
        key=lambda rec: (
            rec["coord_x"],
            rec["coord_y"],
            rec["dt"],
            rec["warmup"],
            rec["quant_k"],
            rec["size"],
            rec["scale"],
            rec["seed_strategy"],
            rec["memory_type"],
            rec["chaos_engine"],
        ),
    )


def write_periodicity_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PERIODICITY_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def write_periodicity_json(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    avg_lag = float(sum(float(r["lag_match_ratio"]) for r in records) / len(records)) if records else 0.0
    payload = {
        "summary": {
            "rows": len(records),
            "mean_lag_match_ratio": avg_lag,
            "rows_with_repeated_chunk_hashes": sum(1 for r in records if int(r["repeated_chunk_hash_count"]) > 0),
            "rows_with_detected_prefix_period": sum(1 for r in records if r.get("detected_prefix_period_bytes") is not None),
        },
        "records": list(records),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
