from __future__ import annotations

import csv
import hashlib
import json
import os
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

RECON_FIELDS = [
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
    "embedding_dim",
    "delay_bytes",
    "sample_count",
    "recon_r2",
    "recon_rmse",
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


def _reconstruction_metrics(data: bytes, embedding_dim: int, delay_bytes: int, max_samples: int) -> tuple[int, float | None, float | None]:
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
    if arr.size == 0:
        return 0, None, None
    series = (arr / 127.5) - 1.0
    m = embedding_dim
    tau = delay_bytes
    start = (m - 1) * tau
    target_idx = np.arange(start + 1, series.size)
    if target_idx.size <= m + 1:
        return int(target_idx.size), None, None
    feature_cols = []
    for i in range(m):
        feature_cols.append(series[target_idx - i * tau])
    X = np.stack(feature_cols, axis=1)
    y = series[target_idx]
    if max_samples > 0 and X.shape[0] > max_samples:
        X = X[:max_samples, :]
        y = y[:max_samples]
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    Xb = np.concatenate([ones, X], axis=1)
    coeffs, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    pred = Xb @ coeffs
    residual = y - pred
    mse = float(np.mean(residual * residual))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 0:
        r2 = None
    else:
        r2 = float(1.0 - float(np.sum(residual * residual)) / denom)
    return int(X.shape[0]), r2, rmse


def _run_one(
    config: FullConfig,
    variant: Dict[str, Any],
    token_bytes: bytes,
    token_fp: str,
    embedding_dim: int,
    delay_bytes: int,
    max_samples: int,
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
            raise RuntimeError("Determinism check failed during attractor reconstruction run.")

    sample_count, r2, rmse = _reconstruction_metrics(ks, embedding_dim=embedding_dim, delay_bytes=delay_bytes, max_samples=max_samples)
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
        "embedding_dim": embedding_dim,
        "delay_bytes": delay_bytes,
        "sample_count": sample_count,
        "recon_r2": r2,
        "recon_rmse": rmse,
    }


def _task(args: Tuple[FullConfig, Dict[str, Any], bytes, str, int, int, int]) -> Dict[str, Any]:
    cfg, variant, token_bytes, token_fp, embedding_dim, delay_bytes, max_samples = args
    setup_logging(os.environ.get("CHAOSCRYPTO_LOG_LEVEL", "WARNING"))
    set_command_context("attractor-reconstruct")
    return _run_one(cfg, variant, token_bytes, token_fp, embedding_dim, delay_bytes, max_samples)


def run_attractor_reconstruction(
    config: FullConfig,
    jobs: int = 1,
    embedding_dim: int = 3,
    delay_bytes: int = 1,
    max_samples: int = 200000,
) -> List[Dict[str, Any]]:
    setup_logging(os.environ.get("CHAOSCRYPTO_LOG_LEVEL", "WARNING"))
    set_command_context("attractor-reconstruct")
    variants = _variant_product(config)
    token_bytes = config.analyze.token.encode(constants.ENCODING)
    token_fp = token_fingerprint(token_bytes)
    if jobs and jobs > 1:
        task_args = [(config, v, token_bytes, token_fp, embedding_dim, delay_bytes, max_samples) for v in variants]
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            records = list(ex.map(_task, task_args))
    else:
        records = [_task((config, v, token_bytes, token_fp, embedding_dim, delay_bytes, max_samples)) for v in variants]
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


def write_reconstruction_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECON_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def write_reconstruction_json(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    r2_vals = [float(r["recon_r2"]) for r in records if r.get("recon_r2") is not None]
    rmse_vals = [float(r["recon_rmse"]) for r in records if r.get("recon_rmse") is not None]
    payload = {
        "summary": {
            "rows": len(records),
            "mean_recon_r2": (sum(r2_vals) / len(r2_vals)) if r2_vals else None,
            "mean_recon_rmse": (sum(rmse_vals) / len(rmse_vals)) if rmse_vals else None,
        },
        "rows": list(records),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
