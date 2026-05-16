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

AVALANCHE_FIELDS = [
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
    "perturbation_type",
    "perturbation_target",
    "perturbation_bit_index",
    "perturbation_skipped",
    "perturbation_skip_reason",
    "token_fingerprint_base",
    "token_fingerprint_perturbed",
    "field_fingerprint_base",
    "field_fingerprint_perturbed",
    "keystream_sha256_base",
    "keystream_sha256_perturbed",
    "hamming_distance_bits",
    "hamming_distance_ratio",
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
                                        }
                                    )
    return combos


def _flip_bit_bytes(data: bytes, bit_index: int) -> bytes:
    if bit_index < 0:
        raise ValueError("bit_index must be >= 0")
    arr = bytearray(data)
    byte_index = bit_index // 8
    bit_in_byte = bit_index % 8
    if byte_index >= len(arr):
        raise ValueError("bit_index outside byte array")
    arr[byte_index] ^= 1 << bit_in_byte
    return bytes(arr)


def _flip_bit_int(value: int, bit_index: int) -> int:
    if bit_index < 0:
        raise ValueError("bit_index must be >= 0")
    return int(value) ^ (1 << bit_index)


def _hamming_distance_bits(a: bytes, b: bytes) -> int:
    if len(a) != len(b):
        raise ValueError("Byte strings must have same length")
    ax = np.frombuffer(a, dtype=np.uint8)
    bx = np.frombuffer(b, dtype=np.uint8)
    x = np.bitwise_xor(ax, bx)
    return int(np.unpackbits(x).sum())


def _get_field(token_bytes: bytes, params: MemoryParams) -> Tuple[np.ndarray, str]:
    cache_key = (token_bytes, params.type, params.size, params.scale)
    cached = _FIELD_CACHE.get(cache_key)
    if cached is None:
        cached = build_memory_field(token_bytes, params)
        _FIELD_CACHE[cache_key] = cached
    return cached


def _generate_keystream_variant(
    token_bytes: bytes,
    coord: Tuple[int, int],
    nbytes: int,
    dt: float,
    warmup: int,
    quant_k: float,
    params: MemoryParams,
    seed_strategy: str,
) -> Tuple[bytes, str]:
    field, field_fp = _get_field(token_bytes, params)
    init_state = derive_initial_state(field, coord, seed_strategy=seed_strategy)
    ks = generate_keystream(nbytes, init_state, dt=dt, warmup=warmup, quant_k=quant_k)
    return ks, field_fp


def _build_perturbations(
    token_bytes: bytes,
    coord: Tuple[int, int],
    size: int,
    token_bit_flips: int,
    coord_bit_flips: int,
) -> List[Dict[str, Any]]:
    perturbations: List[Dict[str, Any]] = []
    max_token_bits = min(max(0, token_bit_flips), len(token_bytes) * 8)
    for bit_idx in range(max_token_bits):
        perturbations.append(
            {
                "type": "token_bit_flip",
                "target": "token",
                "bit_index": bit_idx,
                "token_bytes": _flip_bit_bytes(token_bytes, bit_idx),
                "coord": coord,
            }
        )

    max_coord_bits = max(0, coord_bit_flips)
    for bit_idx in range(max_coord_bits):
        flipped_x = _flip_bit_int(coord[0], bit_idx)
        x_invariant = (flipped_x % size) == (coord[0] % size)
        perturbations.append(
            {
                "type": "coord_x_bit_flip",
                "target": "coord_x",
                "bit_index": bit_idx,
                "token_bytes": token_bytes,
                "coord": (flipped_x, coord[1]),
                "skip": x_invariant,
                "skip_reason": "coord_x modulo memory size unchanged" if x_invariant else None,
            }
        )
        flipped_y = _flip_bit_int(coord[1], bit_idx)
        y_invariant = (flipped_y % size) == (coord[1] % size)
        perturbations.append(
            {
                "type": "coord_y_bit_flip",
                "target": "coord_y",
                "bit_index": bit_idx,
                "token_bytes": token_bytes,
                "coord": (coord[0], flipped_y),
                "skip": y_invariant,
                "skip_reason": "coord_y modulo memory size unchanged" if y_invariant else None,
            }
        )
    return perturbations


def _run_avalanche_one(
    config: FullConfig,
    variant: Dict[str, Any],
    token_bytes: bytes,
    token_fp: str,
    token_bit_flips: int,
    coord_bit_flips: int,
) -> List[Dict[str, Any]]:
    params = MemoryParams(type=variant["memory_type"], size=variant["size"], scale=variant["scale"])
    if variant["seed_strategy"] not in list_seed_strategies():
        raise ValueError(f"Unknown seed_strategy '{variant['seed_strategy']}'. Available: {list_seed_strategies()}")

    base_ks, base_field_fp = _generate_keystream_variant(
        token_bytes=token_bytes,
        coord=variant["coord"],
        nbytes=config.analyze.nbytes,
        dt=variant["dt"],
        warmup=variant["warmup"],
        quant_k=variant["quant_k"],
        params=params,
        seed_strategy=variant["seed_strategy"],
    )
    base_ks_sha = hashlib.sha256(base_ks).hexdigest()

    perturbations = _build_perturbations(
        token_bytes,
        variant["coord"],
        size=variant["size"],
        token_bit_flips=token_bit_flips,
        coord_bit_flips=coord_bit_flips,
    )
    rows: List[Dict[str, Any]] = []
    for p in perturbations:
        skipped = bool(p.get("skip", False))
        skip_reason = p.get("skip_reason")
        if skipped:
            record: Dict[str, Any] = {
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
                "perturbation_type": p["type"],
                "perturbation_target": p["target"],
                "perturbation_bit_index": p["bit_index"],
                "perturbation_skipped": True,
                "perturbation_skip_reason": skip_reason,
                "token_fingerprint_base": token_fp,
                "token_fingerprint_perturbed": token_fingerprint(p["token_bytes"]),
                "field_fingerprint_base": base_field_fp,
                "field_fingerprint_perturbed": base_field_fp,
                "keystream_sha256_base": base_ks_sha,
                "keystream_sha256_perturbed": base_ks_sha,
                "hamming_distance_bits": None,
                "hamming_distance_ratio": None,
            }
            rows.append(record)
            continue

        pert_ks, pert_field_fp = _generate_keystream_variant(
            token_bytes=p["token_bytes"],
            coord=p["coord"],
            nbytes=config.analyze.nbytes,
            dt=variant["dt"],
            warmup=variant["warmup"],
            quant_k=variant["quant_k"],
            params=params,
            seed_strategy=variant["seed_strategy"],
        )
        hd_bits = _hamming_distance_bits(base_ks, pert_ks)
        hd_ratio = hd_bits / (config.analyze.nbytes * 8) if config.analyze.nbytes > 0 else 0.0

        record: Dict[str, Any] = {
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
            "perturbation_type": p["type"],
            "perturbation_target": p["target"],
            "perturbation_bit_index": p["bit_index"],
            "perturbation_skipped": False,
            "perturbation_skip_reason": None,
            "token_fingerprint_base": token_fp,
            "token_fingerprint_perturbed": token_fingerprint(p["token_bytes"]),
            "field_fingerprint_base": base_field_fp,
            "field_fingerprint_perturbed": pert_field_fp,
            "keystream_sha256_base": base_ks_sha,
            "keystream_sha256_perturbed": hashlib.sha256(pert_ks).hexdigest(),
            "hamming_distance_bits": hd_bits,
            "hamming_distance_ratio": hd_ratio,
        }
        rows.append(record)
    return rows


def _run_avalanche_task(
    args: Tuple[FullConfig, Dict[str, Any], bytes, str, int, int],
) -> List[Dict[str, Any]]:
    config, variant, token_bytes, token_fp, token_bit_flips, coord_bit_flips = args
    setup_logging(os.environ.get("CHAOSCRYPTO_LOG_LEVEL", "WARNING"))
    set_command_context("avalanche")
    return _run_avalanche_one(config, variant, token_bytes, token_fp, token_bit_flips, coord_bit_flips)


def run_avalanche(
    config: FullConfig,
    jobs: int = 1,
    token_bit_flips: int = 8,
    coord_bit_flips: int = 8,
) -> List[Dict[str, Any]]:
    setup_logging(os.environ.get("CHAOSCRYPTO_LOG_LEVEL", "WARNING"))
    set_command_context("avalanche")
    variants = _variant_product(config)
    token_bytes = config.analyze.token.encode(constants.ENCODING)
    token_fp = token_fingerprint(token_bytes)

    if jobs and jobs > 1:
        task_args = [(config, variant, token_bytes, token_fp, token_bit_flips, coord_bit_flips) for variant in variants]
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            chunks = list(ex.map(_run_avalanche_task, task_args))
    else:
        chunks = [_run_avalanche_task((config, variant, token_bytes, token_fp, token_bit_flips, coord_bit_flips)) for variant in variants]

    rows = [row for chunk in chunks for row in chunk]
    return sorted(
        rows,
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
            rec["perturbation_type"],
            rec["perturbation_bit_index"],
        ),
    )


def write_avalanche_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AVALANCHE_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def write_avalanche_json(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    non_skipped = [r for r in records if not bool(r.get("perturbation_skipped"))]
    payload = {
        "summary": {
            "rows": len(records),
            "rows_skipped": len(records) - len(non_skipped),
            "rows_evaluated": len(non_skipped),
            "mean_hamming_distance_ratio": (
                float(sum(float(r["hamming_distance_ratio"]) for r in non_skipped) / len(non_skipped)) if non_skipped else 0.0
            ),
        },
        "records": list(records),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
