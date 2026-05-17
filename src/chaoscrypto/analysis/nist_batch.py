from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Sequence, Tuple

from chaoscrypto.analysis.nist_validator import NIST_TEST_NAMES, flatten_nist_results, run_full_nist_suite
from chaoscrypto.analysis.runner import FullConfig
from chaoscrypto.core import constants
from chaoscrypto.core.memory.base import MemoryParams
from chaoscrypto.io.profiles import token_fingerprint
from chaoscrypto.orchestrator.pipeline import build_memory_field, derive_initial_state, generate_keystream
from chaoscrypto.utils.logging import set_command_context, setup_logging

_FIELD_CACHE: Dict[Tuple[bytes, str, int, float], Tuple[Any, str]] = {}

RUN_FIELDS: List[str] = [
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
    "keystream_sha256",
    "nist_alpha",
    "nist_passed_count",
    "nist_failed_count",
    "nist_skipped_count",
    "nist_total_runtime_s",
    "nist_failed_tests",
]
for _test_name in NIST_TEST_NAMES:
    _prefix = f"nist_{_test_name}"
    RUN_FIELDS.extend([f"{_prefix}_status", f"{_prefix}_p_value", f"{_prefix}_skip_reason"])

SUMMARY_FIELDS: List[str] = [
    "test_name",
    "alpha",
    "n_runs",
    "n_evaluated",
    "n_pass",
    "n_fail",
    "n_skip",
    "pass_rate",
    "pvalue_mean",
    "pvalue_std",
    "pvalue_min",
    "pvalue_max",
]


@dataclass(frozen=True)
class NistBatchSummary:
    runs: int
    rows_summary: int
    failed_any_runs: int


def _variant_product(cfg: FullConfig) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    for coord in cfg.analyze.coords:
        for dt in cfg.matrix.dt:
            for warmup in cfg.matrix.warmup:
                for quant_k in cfg.matrix.quant_k:
                    for size in cfg.matrix.size:
                        for scale in cfg.matrix.scale:
                            for seed_strategy in cfg.matrix.seed_strategy:
                                for memory_type in cfg.matrix.memory_type:
                                    for chaos_engine in cfg.matrix.chaos_engine:
                                        variants.append(
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
    return variants


def _generate_keystream(
    token_bytes: bytes,
    nbytes: int,
    coord: Tuple[int, int],
    dt: float,
    warmup: int,
    quant_k: float,
    size: int,
    scale: float,
    seed_strategy: str,
    memory_type: str,
    chaos_engine: str,
) -> bytes:
    params = MemoryParams(type=memory_type, size=size, scale=scale)
    cache_key = (token_bytes, params.type, params.size, params.scale)
    cached = _FIELD_CACHE.get(cache_key)
    if cached is None:
        cached = build_memory_field(token_bytes, params)
        _FIELD_CACHE[cache_key] = cached
    field, _field_fp = cached
    init_state = derive_initial_state(field, coord, seed_strategy=seed_strategy)
    return generate_keystream(nbytes, init_state, dt=dt, warmup=warmup, quant_k=quant_k, chaos_engine=chaos_engine)


def _run_one(cfg: FullConfig, variant: Dict[str, Any], token_bytes: bytes, token_fp: str) -> Dict[str, Any]:
    ks = _generate_keystream(
        token_bytes=token_bytes,
        nbytes=cfg.analyze.nbytes,
        coord=variant["coord"],
        dt=variant["dt"],
        warmup=variant["warmup"],
        quant_k=variant["quant_k"],
        size=variant["size"],
        scale=variant["scale"],
        seed_strategy=variant["seed_strategy"],
        memory_type=variant["memory_type"],
        chaos_engine=variant["chaos_engine"],
    )
    if cfg.validate.assert_deterministic_within_run:
        ks2 = _generate_keystream(
            token_bytes=token_bytes,
            nbytes=cfg.analyze.nbytes,
            coord=variant["coord"],
            dt=variant["dt"],
            warmup=variant["warmup"],
            quant_k=variant["quant_k"],
            size=variant["size"],
            scale=variant["scale"],
            seed_strategy=variant["seed_strategy"],
            memory_type=variant["memory_type"],
            chaos_engine=variant["chaos_engine"],
        )
        if ks2 != ks:
            raise RuntimeError("Determinism check failed for nist-batch run.")

    results = run_full_nist_suite(ks, alpha=cfg.metrics.nist_alpha)
    flat = flatten_nist_results(results)
    tests = results.get("tests", {})
    row: Dict[str, Any] = {
        "profile": cfg.analyze.profile,
        "coord_x": variant["coord"][0],
        "coord_y": variant["coord"][1],
        "nbytes": cfg.analyze.nbytes,
        "dt": variant["dt"],
        "warmup": variant["warmup"],
        "quant_k": variant["quant_k"],
        "size": variant["size"],
        "scale": variant["scale"],
        "seed_strategy": variant["seed_strategy"],
        "memory_type": variant["memory_type"],
        "chaos_engine": variant["chaos_engine"],
        "token_fingerprint": token_fp,
        "keystream_sha256": hashlib.sha256(ks).hexdigest(),
        **flat,
    }
    for test_name in NIST_TEST_NAMES:
        prefix = f"nist_{test_name}"
        details = (tests.get(test_name) or {}).get("details", {})
        row[f"{prefix}_skip_reason"] = details.get("reason")
    return row


def run_nist_batch(cfg: FullConfig, jobs: int = 1) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], NistBatchSummary]:
    setup_logging(os.environ.get("CHAOSCRYPTO_LOG_LEVEL", "WARNING"))
    set_command_context("nist-batch")
    if cfg.metrics.nist_suite_enabled is False:
        raise RuntimeError("metrics.nist_suite.enabled must be true for nist-batch.")

    token_bytes = cfg.analyze.token.encode(constants.ENCODING)
    token_fp = token_fingerprint(token_bytes)
    variants = _variant_product(cfg)

    if jobs and jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        args = [(cfg, variant, token_bytes, token_fp) for variant in variants]
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            run_rows = list(ex.map(_run_nist_batch_task, args))
    else:
        run_rows = [_run_nist_batch_task((cfg, variant, token_bytes, token_fp)) for variant in variants]

    run_rows = sorted(
        run_rows,
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
    summary_rows = _aggregate_by_test(run_rows, alpha=cfg.metrics.nist_alpha)
    failed_any_runs = sum(1 for row in run_rows if (row.get("nist_failed_count") or 0) > 0)
    return run_rows, summary_rows, NistBatchSummary(runs=len(run_rows), rows_summary=len(summary_rows), failed_any_runs=failed_any_runs)


def _run_nist_batch_task(args: Tuple[FullConfig, Dict[str, Any], bytes, str]) -> Dict[str, Any]:
    cfg, variant, token_bytes, token_fp = args
    setup_logging(os.environ.get("CHAOSCRYPTO_LOG_LEVEL", "WARNING"))
    set_command_context("nist-batch")
    return _run_one(cfg, variant, token_bytes, token_fp)


def _aggregate_by_test(run_rows: Sequence[Dict[str, Any]], alpha: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for test_name in NIST_TEST_NAMES:
        status_key = f"nist_{test_name}_status"
        pvalue_key = f"nist_{test_name}_p_value"
        statuses = [str(r.get(status_key) or "") for r in run_rows]
        pvals = [float(r[pvalue_key]) for r in run_rows if r.get(pvalue_key) is not None and str(r.get(pvalue_key)) != ""]
        n_pass = sum(1 for s in statuses if s == "pass")
        n_fail = sum(1 for s in statuses if s == "fail")
        n_skip = sum(1 for s in statuses if s == "skip")
        n_eval = n_pass + n_fail
        pass_rate = (n_pass / n_eval) if n_eval > 0 else None
        row: Dict[str, Any] = {
            "test_name": test_name,
            "alpha": alpha,
            "n_runs": len(run_rows),
            "n_evaluated": n_eval,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_skip": n_skip,
            "pass_rate": pass_rate,
            "pvalue_mean": mean(pvals) if pvals else None,
            "pvalue_std": pstdev(pvals) if len(pvals) > 1 else 0.0 if len(pvals) == 1 else None,
            "pvalue_min": min(pvals) if pvals else None,
            "pvalue_max": max(pvals) if pvals else None,
        }
        rows.append(row)
    return rows


def write_nist_runs_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_nist_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_nist_batch_json(path: Path, run_rows: Sequence[Dict[str, Any]], summary_rows: Sequence[Dict[str, Any]], summary: NistBatchSummary) -> None:
    payload = {
        "summary": {
            "runs": summary.runs,
            "rows_summary": summary.rows_summary,
            "failed_any_runs": summary.failed_any_runs,
        },
        "runs": list(run_rows),
        "tests": list(summary_rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
