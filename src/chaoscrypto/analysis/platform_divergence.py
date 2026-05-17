from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
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

PLATFORM_CHECK_FIELDS = [
    "timestamp_utc",
    "runtime_label",
    "platform_system",
    "platform_release",
    "platform_machine",
    "platform_python_implementation",
    "platform_python_version",
    "platform_numpy_version",
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
]

PLATFORM_COMPARE_FIELDS = [
    "variant_key",
    "status",
    "field_fingerprint_match",
    "keystream_sha256_match",
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
    "reference_runtime_label",
    "reference_platform_system",
    "reference_platform_release",
    "reference_platform_machine",
    "reference_platform_python_version",
    "reference_platform_numpy_version",
    "reference_field_fingerprint",
    "reference_keystream_sha256",
    "candidate_runtime_label",
    "candidate_platform_system",
    "candidate_platform_release",
    "candidate_platform_machine",
    "candidate_platform_python_version",
    "candidate_platform_numpy_version",
    "candidate_field_fingerprint",
    "candidate_keystream_sha256",
]

_VARIANT_KEY_FIELDS = [
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


def _environment_info(runtime_label: str | None) -> Dict[str, Any]:
    return {
        "runtime_label": runtime_label or platform.platform(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "platform_python_implementation": platform.python_implementation(),
        "platform_python_version": platform.python_version(),
        "platform_numpy_version": np.__version__,
    }


def _get_field(token_bytes: bytes, params: MemoryParams) -> Tuple[np.ndarray, str]:
    cache_key = (token_bytes, params.type, params.size, params.scale)
    cached = _FIELD_CACHE.get(cache_key)
    if cached is None:
        cached = build_memory_field(token_bytes, params)
        _FIELD_CACHE[cache_key] = cached
    return cached


def _platform_check_one(
    config: FullConfig,
    variant: Dict[str, Any],
    token_bytes: bytes,
    token_fp: str,
    runtime_label: str | None,
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
        ks_2 = generate_keystream(
            config.analyze.nbytes,
            init_state_2,
            dt=variant["dt"],
            warmup=variant["warmup"],
            quant_k=variant["quant_k"],
            chaos_engine=variant["chaos_engine"],
        )
        if ks_2 != ks:
            raise RuntimeError("Determinism check failed during platform check.")

    record: Dict[str, Any] = {
        **_environment_info(runtime_label),
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
    }
    if config.output.include_timestamp_utc:
        record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    else:
        record["timestamp_utc"] = None
    return record


def _run_platform_check_task(args: Tuple[FullConfig, Dict[str, Any], bytes, str, str | None]) -> Dict[str, Any]:
    config, variant, token_bytes, token_fp, runtime_label = args
    setup_logging(os.environ.get("CHAOSCRYPTO_LOG_LEVEL", "WARNING"))
    set_command_context("platform-check")
    return _platform_check_one(config, variant, token_bytes, token_fp, runtime_label)


def run_platform_check(config: FullConfig, jobs: int = 1, runtime_label: str | None = None) -> List[Dict[str, Any]]:
    setup_logging(os.environ.get("CHAOSCRYPTO_LOG_LEVEL", "WARNING"))
    set_command_context("platform-check")
    variants = _variant_product(config)
    token_bytes = config.analyze.token.encode(constants.ENCODING)
    token_fp = token_fingerprint(token_bytes)

    if jobs and jobs > 1:
        task_args = [(config, variant, token_bytes, token_fp, runtime_label) for variant in variants]
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            records = list(ex.map(_run_platform_check_task, task_args))
    else:
        records = [_run_platform_check_task((config, variant, token_bytes, token_fp, runtime_label)) for variant in variants]

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


def write_platform_check_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLATFORM_CHECK_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def write_platform_check_json(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(records), handle, indent=2)


def _read_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _variant_key(record: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(str(record.get(field, "")) for field in _VARIANT_KEY_FIELDS)


def _variant_key_text(record: Dict[str, Any]) -> str:
    parts = [f"{field}={record.get(field, '')}" for field in _VARIANT_KEY_FIELDS]
    return "|".join(parts)


def _records_by_key(records: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    result: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for rec in records:
        key = _variant_key(rec)
        if key in result:
            raise ValueError(f"Duplicate platform-check variant detected for key '{_variant_key_text(rec)}'")
        result[key] = rec
    return result


def compare_platform_outputs(
    reference_records: Sequence[Dict[str, Any]],
    candidate_records: Sequence[Dict[str, Any]],
    reference_label: str,
    candidate_label: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ref_map = _records_by_key(reference_records)
    cand_map = _records_by_key(candidate_records)
    all_keys = sorted(set(ref_map) | set(cand_map))
    rows: List[Dict[str, Any]] = []

    match_count = 0
    diverged_count = 0
    missing_reference = 0
    missing_candidate = 0

    for key in all_keys:
        ref = ref_map.get(key)
        cand = cand_map.get(key)
        base = ref or cand or {}

        if ref is None:
            status = "missing_reference"
            field_match = False
            ks_match = False
            missing_reference += 1
        elif cand is None:
            status = "missing_candidate"
            field_match = False
            ks_match = False
            missing_candidate += 1
        else:
            field_match = str(ref.get("field_fingerprint", "")) == str(cand.get("field_fingerprint", ""))
            ks_match = str(ref.get("keystream_sha256", "")) == str(cand.get("keystream_sha256", ""))
            if field_match and ks_match:
                status = "match"
                match_count += 1
            else:
                status = "diverged"
                diverged_count += 1

        rows.append(
            {
                "variant_key": _variant_key_text(base),
                "status": status,
                "field_fingerprint_match": field_match,
                "keystream_sha256_match": ks_match,
                "profile": base.get("profile"),
                "coord_x": base.get("coord_x"),
                "coord_y": base.get("coord_y"),
                "nbytes": base.get("nbytes"),
                "dt": base.get("dt"),
                "warmup": base.get("warmup"),
                "quant_k": base.get("quant_k"),
                "size": base.get("size"),
                "scale": base.get("scale"),
                "seed_strategy": base.get("seed_strategy"),
                "memory_type": base.get("memory_type"),
                "reference_runtime_label": (ref or {}).get("runtime_label") or reference_label,
                "reference_platform_system": (ref or {}).get("platform_system"),
                "reference_platform_release": (ref or {}).get("platform_release"),
                "reference_platform_machine": (ref or {}).get("platform_machine"),
                "reference_platform_python_version": (ref or {}).get("platform_python_version"),
                "reference_platform_numpy_version": (ref or {}).get("platform_numpy_version"),
                "reference_field_fingerprint": (ref or {}).get("field_fingerprint"),
                "reference_keystream_sha256": (ref or {}).get("keystream_sha256"),
                "candidate_runtime_label": (cand or {}).get("runtime_label") or candidate_label,
                "candidate_platform_system": (cand or {}).get("platform_system"),
                "candidate_platform_release": (cand or {}).get("platform_release"),
                "candidate_platform_machine": (cand or {}).get("platform_machine"),
                "candidate_platform_python_version": (cand or {}).get("platform_python_version"),
                "candidate_platform_numpy_version": (cand or {}).get("platform_numpy_version"),
                "candidate_field_fingerprint": (cand or {}).get("field_fingerprint"),
                "candidate_keystream_sha256": (cand or {}).get("keystream_sha256"),
            }
        )

    summary = {
        "reference_label": reference_label,
        "candidate_label": candidate_label,
        "variants_total": len(rows),
        "matches": match_count,
        "diverged": diverged_count,
        "missing_reference": missing_reference,
        "missing_candidate": missing_candidate,
    }
    return rows, summary


def write_platform_compare_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLATFORM_COMPARE_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def write_platform_compare_json(path: Path, records: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "records": list(records)}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def compare_platform_files(
    reference_path: Path,
    candidate_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    reference_records = _read_records(reference_path)
    candidate_records = _read_records(candidate_path)
    return compare_platform_outputs(
        reference_records=reference_records,
        candidate_records=candidate_records,
        reference_label=reference_path.stem,
        candidate_label=candidate_path.stem,
    )
