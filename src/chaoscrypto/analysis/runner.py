from __future__ import annotations

import base64
import csv
import hashlib
import itertools
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml

from chaoscrypto.core import constants
from chaoscrypto.core.memory.base import MemoryParams
from chaoscrypto.io.profiles import token_fingerprint
from chaoscrypto.orchestrator.pipeline import (
    build_memory_field,
    derive_initial_state,
    generate_keystream,
)
from chaoscrypto.core.seed.base import list_seed_strategies
from chaoscrypto.core.memory.base import list_memory_models
from chaoscrypto.core.seed.base import list_seed_strategies


class ConfigError(Exception):
    """Raised when the analyze config is invalid."""


@dataclass(frozen=True)
class AnalyzeConfig:
    profile: str
    token: str
    coords: List[Tuple[int, int]]
    nbytes: int


@dataclass(frozen=True)
class MatrixConfig:
    dt: Sequence[float]
    warmup: Sequence[int]
    quant_k: Sequence[float]
    size: Sequence[int]
    scale: Sequence[float]
    seed_strategy: Sequence[str]
    memory_type: Sequence[str]
    provided_memory_type: bool


@dataclass(frozen=True)
class MetricsConfig:
    bit_balance: bool
    byte_histogram: bool
    chi_square_bytes: bool
    autocorr_bits_enabled: bool
    autocorr_bits_max_lag: int
    runs_test_bits: bool
    hamming_weight_enabled: bool
    hamming_weight_window_bits: int


@dataclass(frozen=True)
class OutputConfig:
    include_timestamp_utc: bool
    include_field_fingerprint: bool
    include_keystream_sha256: bool
    include_preview_base64: bool
    preview_bytes: int


@dataclass(frozen=True)
class ValidateConfig:
    assert_deterministic_within_run: bool


@dataclass(frozen=True)
class FullConfig:
    analyze: AnalyzeConfig
    matrix: MatrixConfig
    metrics: MetricsConfig
    output: OutputConfig
    validate: ValidateConfig


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

    analyze = _require(data, "analyze", (dict,))
    matrix = _require(data, "matrix", (dict,))
    metrics = _require(data, "metrics", (dict,))
    output = _require(data, "output", (dict,))
    validate = _require(data, "validate", (dict,))

    coords: List[Tuple[int, int]] = []
    if "coords" in analyze:
        coord_list = _require(analyze, "coords", (list, tuple))
        for c in coord_list:
            if not isinstance(c, (list, tuple)) or len(c) != 2:
                raise ConfigError("Each coords entry must be [x, y]")
            coords.append((int(c[0]), int(c[1])))
    else:
        coord_val = _require(analyze, "coord", (list, tuple))
        if len(coord_val) != 2:
            raise ConfigError("analyze.coord must have exactly two entries (x,y)")
        coords.append((int(coord_val[0]), int(coord_val[1])))

    analyze_cfg = AnalyzeConfig(
        profile=_require(analyze, "profile", (str,)),
        token=_require(analyze, "token", (str,)),
        coords=coords,
        nbytes=int(_require(analyze, "nbytes", (int, float))),
    )

    mem_provided = "memory_type" in matrix
    mem_val = matrix.get("memory_type", [constants.MEMORY_TYPE])
    if not isinstance(mem_val, (list, tuple)):
        mem_val = [mem_val]
    seed_val = matrix.get("seed_strategy", [constants.SEED_STRATEGY])
    if not isinstance(seed_val, (list, tuple)):
        seed_val = [seed_val]
    matrix_cfg = MatrixConfig(
        dt=[float(x) for x in _require(matrix, "dt", (list, tuple))],
        warmup=[int(x) for x in _require(matrix, "warmup", (list, tuple))],
        quant_k=[float(x) for x in _require(matrix, "quant_k", (list, tuple))],
        size=[int(x) for x in matrix.get("size", [constants.DEFAULT_MEMORY_SIZE])],
        scale=[float(x) for x in matrix.get("scale", [constants.DEFAULT_MEMORY_SCALE])],
        seed_strategy=[str(x) for x in seed_val],
        memory_type=[str(x) for x in mem_val],
        provided_memory_type=mem_provided,
    )

    autocorr = metrics.get("autocorr_bits", {})
    if autocorr and not isinstance(autocorr, dict):
        raise ConfigError("metrics.autocorr_bits must be a mapping if provided.")
    hw = metrics.get("hamming_weight_window", {})
    if hw and not isinstance(hw, dict):
        raise ConfigError("metrics.hamming_weight_window must be a mapping if provided.")

    metrics_cfg = MetricsConfig(
        bit_balance=bool(metrics.get("bit_balance", True)),
        byte_histogram=bool(metrics.get("byte_histogram", True)),
        chi_square_bytes=bool(metrics.get("chi_square_bytes", True)),
        autocorr_bits_enabled=bool((autocorr or {}).get("enabled", False)),
        autocorr_bits_max_lag=int((autocorr or {}).get("max_lag", 0)),
        runs_test_bits=bool(metrics.get("runs_test_bits", True)),
        hamming_weight_enabled=bool((hw or {}).get("enabled", False)),
        hamming_weight_window_bits=int((hw or {}).get("window_bits", 0)),
    )
    if metrics_cfg.autocorr_bits_enabled and metrics_cfg.autocorr_bits_max_lag <= 0:
        raise ConfigError("metrics.autocorr_bits.max_lag must be > 0 when enabled.")
    if metrics_cfg.hamming_weight_enabled and metrics_cfg.hamming_weight_window_bits <= 0:
        raise ConfigError("metrics.hamming_weight_window.window_bits must be > 0 when enabled.")

    output_cfg = OutputConfig(
        include_timestamp_utc=bool(output.get("include_timestamp_utc", True)),
        include_field_fingerprint=bool(output.get("include_field_fingerprint", True)),
        include_keystream_sha256=bool(output.get("include_keystream_sha256", True)),
        include_preview_base64=bool(output.get("include_preview_base64", False)),
        preview_bytes=int(output.get("preview_bytes", 64)),
    )

    validate_cfg = ValidateConfig(
        assert_deterministic_within_run=bool(validate.get("assert_deterministic_within_run", True))
    )

    for name in matrix_cfg.seed_strategy:
        if name not in list_seed_strategies():
            raise ConfigError(f"Unknown seed_strategy '{name}'. Available: {list_seed_strategies()}")
    for name in matrix_cfg.memory_type:
        if name not in list_memory_models():
            raise ConfigError(f"Unknown memory_type '{name}'. Available: {list_memory_models()}")

    return FullConfig(
        analyze=analyze_cfg,
        matrix=matrix_cfg,
        metrics=metrics_cfg,
        output=output_cfg,
        validate=validate_cfg,
    )


def _variant_product(matrix: MatrixConfig, coords: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    combos = []
    for coord in coords:
        for dt, warmup, quant_k, size, scale, seed_strategy, memory_type in itertools.product(
            matrix.dt, matrix.warmup, matrix.quant_k, matrix.size, matrix.scale, matrix.seed_strategy, matrix.memory_type
        ):
            seed_strategy = seed_strategy or constants.SEED_STRATEGY
            memory_type = memory_type or constants.MEMORY_TYPE
            combos.append(
                {
                    "coord": coord,
                    "dt": float(dt),
                    "warmup": int(warmup),
                    "quant_k": float(quant_k),
                    "size": int(size),
                    "scale": float(scale),
                    "seed_strategy": str(seed_strategy),
                    "memory_type": str(memory_type),
                }
            )
    return combos


def _bits_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits


def _bit_balance(bits: np.ndarray) -> Dict[str, Any]:
    ones = int(bits.sum())
    total = bits.size
    return {"bit_ones_ratio": ones / total if total else 0.0, "bit_ones_count": ones}


def _byte_histogram(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    return hist.astype(np.int64)


def _chi_square(hist: np.ndarray) -> Tuple[float, float]:
    total = hist.sum()
    if total == 0:
        return 0.0, 0.0
    expected = total / 256.0
    chi2 = float(((hist - expected) ** 2 / expected).sum())
    return chi2, chi2 / 255.0


def _byte_entropy(hist: np.ndarray) -> float:
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist / total
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _byte_top5(hist: np.ndarray) -> str:
    items = list(enumerate(hist.tolist()))
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    top5 = items[:5]
    return "|".join(f"{b:02x}:{cnt}" for b, cnt in top5)


def _autocorr_bits(bits: np.ndarray, max_lag: int) -> List[float]:
    if bits.size == 0:
        return [0.0 for _ in range(max_lag)]
    mean = bits.mean()
    var = bits.var()
    if var == 0:
        return [0.0 for _ in range(max_lag)]
    autocorr = []
    for lag in range(1, max_lag + 1):
        x = bits[:-lag]
        y = bits[lag:]
        cov = ((x - mean) * (y - mean)).mean()
        autocorr.append(float(cov / var))
    return autocorr


def _runs_test(bits: np.ndarray) -> Dict[str, Any]:
    if bits.size == 0:
        return {"runs_count": 0, "runs_expected": 0.0, "runs_norm_diff": 0.0}
    runs = 1 + int((bits[1:] != bits[:-1]).sum())
    n1 = int(bits.sum())
    n0 = int(bits.size - n1)
    if n0 == 0 or n1 == 0:
        expected = 1.0
    else:
        expected = 1 + 2 * n1 * n0 / bits.size
    norm_diff = (runs - expected) / bits.size
    return {"runs_count": runs, "runs_expected": expected, "runs_norm_diff": norm_diff}


def _hamming_weight_window(bits: np.ndarray, window_bits: int) -> Dict[str, Any]:
    if window_bits <= 0 or bits.size == 0:
        return {"hw_win_mean": None, "hw_win_std": None, "hw_win_min": None, "hw_win_max": None}
    n_windows = math.floor(bits.size / window_bits)
    if n_windows == 0:
        return {"hw_win_mean": None, "hw_win_std": None, "hw_win_min": None, "hw_win_max": None}
    reshaped = bits[: n_windows * window_bits].reshape(n_windows, window_bits)
    weights = reshaped.sum(axis=1)
    return {
        "hw_win_mean": float(weights.mean()),
        "hw_win_std": float(weights.std()),
        "hw_win_min": float(weights.min()),
        "hw_win_max": float(weights.max()),
    }


def _generate_keystream(params: MemoryParams, coord: Tuple[int, int], nbytes: int, dt: float, warmup: int, quant_k: float, token_bytes: bytes, seed_strategy: str):
    field, field_fp = build_memory_field(token_bytes, params)
    init_state = derive_initial_state(field, coord, seed_strategy=seed_strategy)
    ks = generate_keystream(nbytes, init_state, dt=dt, warmup=warmup, quant_k=quant_k)
    return ks, field_fp


def _analyze_one(
    cfg: FullConfig,
    variant: Dict[str, Any],
    token_bytes: bytes,
    token_fp: str,
) -> Dict[str, Any]:
    params = MemoryParams(type=constants.MEMORY_TYPE, size=variant["size"], scale=variant["scale"])
    params = MemoryParams(type=variant["memory_type"], size=variant["size"], scale=variant["scale"])
    if variant["seed_strategy"] not in list_seed_strategies():
        raise ConfigError(f"Unknown seed_strategy '{variant['seed_strategy']}'. Available: {list_seed_strategies()}")
    ks, field_fp = _generate_keystream(
        params=params,
        coord=variant["coord"],
        nbytes=cfg.analyze.nbytes,
        dt=variant["dt"],
        warmup=variant["warmup"],
        quant_k=variant["quant_k"],
        token_bytes=token_bytes,
        seed_strategy=variant["seed_strategy"],
    )

    if cfg.validate.assert_deterministic_within_run:
        ks2, _ = _generate_keystream(
            params=params,
            coord=variant["coord"],
            nbytes=cfg.analyze.nbytes,
            dt=variant["dt"],
            warmup=variant["warmup"],
            quant_k=variant["quant_k"],
            token_bytes=token_bytes,
            seed_strategy=variant["seed_strategy"],
        )
        if ks2 != ks:
            raise RuntimeError("Determinism check failed for analyze run.")

    bits = _bits_from_bytes(ks)
    metrics: Dict[str, Any] = {}

    if cfg.metrics.bit_balance:
        metrics.update(_bit_balance(bits))
    if cfg.metrics.byte_histogram or cfg.metrics.chi_square_bytes:
        hist = _byte_histogram(ks)
        if cfg.metrics.byte_histogram:
            metrics["byte_entropy_approx"] = _byte_entropy(hist)
            metrics["byte_top5"] = _byte_top5(hist)
        if cfg.metrics.chi_square_bytes:
            chi2, chi2_norm = _chi_square(hist)
            metrics["byte_chi2"] = chi2
            metrics["byte_chi2_norm"] = chi2_norm
        metrics["byte_histogram"] = hist.tolist()
    if cfg.metrics.autocorr_bits_enabled:
        autocorr = _autocorr_bits(bits, cfg.metrics.autocorr_bits_max_lag)
        for i, val in enumerate(autocorr, start=1):
            metrics[f"autocorr_lag_{i}"] = val
    if cfg.metrics.runs_test_bits:
        metrics.update(_runs_test(bits))
    if cfg.metrics.hamming_weight_enabled:
        metrics.update(_hamming_weight_window(bits, cfg.metrics.hamming_weight_window_bits))

    record: Dict[str, Any] = {
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
        "token_fingerprint": token_fp,
        "field_fingerprint": field_fp if cfg.output.include_field_fingerprint else None,
        "keystream_sha256": hashlib.sha256(ks).hexdigest() if cfg.output.include_keystream_sha256 else None,
    }
    if cfg.output.include_timestamp_utc:
        record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    if cfg.output.include_preview_base64:
        record["keystream_preview_base64"] = base64.b64encode(ks[: cfg.output.preview_bytes]).decode("ascii")
    else:
        record["keystream_preview_base64"] = None

    record.update(metrics)
    return record


def run_analyze(config: FullConfig, jobs: int = 1) -> List[Dict[str, Any]]:
    variants = _variant_product(config.matrix, config.analyze.coords)
    tasks = variants
    token_bytes = config.analyze.token.encode(constants.ENCODING)
    token_fp = token_fingerprint(token_bytes)

    def runner(variant: Dict[str, Any]) -> Dict[str, Any]:
        return _analyze_one(config, variant, token_bytes, token_fp)

    if jobs and jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=jobs) as ex:
            records = list(ex.map(runner, tasks))
    else:
        records = [runner(v) for v in tasks]

    def sort_key(rec: Dict[str, Any]):
        return (
            rec["coord_x"],
            rec["coord_y"],
            rec["dt"],
            rec["warmup"],
            rec["quant_k"],
            rec["size"],
            rec["scale"],
            rec.get("seed_strategy"),
            rec.get("memory_type"),
        )

    records_sorted = sorted(records, key=sort_key)
    return records_sorted


CSV_FIELDS_BASE = [
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
    "keystream_sha256",
    "field_fingerprint",
    "token_fingerprint",
    "bit_ones_ratio",
    "bit_ones_count",
    "byte_chi2",
    "byte_chi2_norm",
    "byte_entropy_approx",
    "byte_top5",
    "runs_count",
    "runs_expected",
    "runs_norm_diff",
    "hw_win_mean",
    "hw_win_std",
    "hw_win_min",
    "hw_win_max",
    "keystream_preview_base64",
]


def csv_fields(max_autocorr: int) -> List[str]:
    fields = list(CSV_FIELDS_BASE)
    for lag in range(1, max_autocorr + 1):
        fields.append(f"autocorr_lag_{lag}")
    return fields


def write_csv(path: Path, records: List[Dict[str, Any]], max_autocorr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields(max_autocorr), extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def write_json_output(path: Path, records: List[Dict[str, Any]]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
