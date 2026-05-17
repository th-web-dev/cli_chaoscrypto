from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

KNOWN_STRICT_TEMPLATE_TESTS = {"non_overlapping_template_matching", "overlapping_template_matching"}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    text = str(val).strip()
    if text == "":
        return None
    return float(text)


def _to_int(val: Any) -> int | None:
    if val is None:
        return None
    text = str(val).strip()
    if text == "":
        return None
    return int(float(text))


def _variant_key(row: Dict[str, str]) -> str:
    return (
        f"dt={row.get('dt')}|warmup={row.get('warmup')}|quant_k={row.get('quant_k')}|"
        f"seed_strategy={row.get('seed_strategy')}|memory_type={row.get('memory_type')}|chaos_engine={row.get('chaos_engine')}"
    )


def _summarize_nist(runs: List[Dict[str, str]], tests: List[Dict[str, str]]) -> Dict[str, Any]:
    total = len(runs)
    failed_any = sum(1 for r in runs if (_to_int(r.get("nist_failed_count")) or 0) > 0)
    pass_counts = [_to_int(r.get("nist_passed_count")) for r in runs]
    pass_counts = [x for x in pass_counts if x is not None]
    fail_counts = [_to_int(r.get("nist_failed_count")) for r in runs]
    fail_counts = [x for x in fail_counts if x is not None]
    worst_tests = sorted(
        tests,
        key=lambda r: (_to_int(r.get("n_fail")) or 0, -(_to_float(r.get("pass_rate")) or 0.0)),
        reverse=True,
    )[:5]
    core_tests = [r for r in tests if (r.get("test_name") or "") not in KNOWN_STRICT_TEMPLATE_TESTS]
    core_fail_tests = sum(1 for r in core_tests if (_to_int(r.get("n_fail")) or 0) > 0)
    by_memory_type: Dict[str, List[int]] = {}
    by_seed_strategy: Dict[str, List[int]] = {}
    by_quant_k: Dict[str, List[int]] = {}
    by_warmup: Dict[str, List[int]] = {}
    by_dt: Dict[str, List[int]] = {}
    for row in runs:
        failed = _to_int(row.get("nist_failed_count"))
        if failed is None:
            continue
        by_memory_type.setdefault(str(row.get("memory_type")), []).append(failed)
        by_seed_strategy.setdefault(str(row.get("seed_strategy")), []).append(failed)
        by_quant_k.setdefault(str(row.get("quant_k")), []).append(failed)
        by_warmup.setdefault(str(row.get("warmup")), []).append(failed)
        by_dt.setdefault(str(row.get("dt")), []).append(failed)

    def _rank_groups(group: Dict[str, List[int]], top_n: int = 3) -> List[Dict[str, Any]]:
        ranked = sorted(((k, mean(v)) for k, v in group.items() if v), key=lambda kv: kv[1], reverse=True)
        return [{"group": k, "mean_failed_tests": v} for k, v in ranked[:top_n]]

    return {
        "variants_total": total,
        "variants_with_any_fail": failed_any,
        "variants_fail_share": (failed_any / total) if total else None,
        "mean_passed_tests_per_variant": mean(pass_counts) if pass_counts else None,
        "mean_failed_tests_per_variant": mean(fail_counts) if fail_counts else None,
        "core_tests_count_excluding_template": len(core_tests),
        "core_tests_with_any_fail_excluding_template": core_fail_tests,
        "worst_tests": [
            {
                "test_name": row.get("test_name"),
                "n_fail": _to_int(row.get("n_fail")),
                "pass_rate": _to_float(row.get("pass_rate")),
            }
            for row in worst_tests
        ],
        "fail_hotspots": {
            "memory_type_top3": _rank_groups(by_memory_type),
            "seed_strategy_top3": _rank_groups(by_seed_strategy),
            "quant_k_top3": _rank_groups(by_quant_k),
            "warmup_top3": _rank_groups(by_warmup),
            "dt_top3": _rank_groups(by_dt),
        },
    }


def _summarize_avalanche(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if not rows:
        return {"rows": 0}
    ratios = [_to_float(r.get("hamming_distance_ratio")) for r in rows]
    ratios = [x for x in ratios if x is not None]
    skipped = sum(1 for r in rows if str(r.get("perturbation_skipped", "")).lower() in {"true", "1"})
    by_variant: Dict[str, List[float]] = {}
    for row in rows:
        ratio = _to_float(row.get("hamming_distance_ratio"))
        if ratio is None:
            continue
        by_variant.setdefault(_variant_key(row), []).append(ratio)
    ranked = sorted(
        ((k, mean(v)) for k, v in by_variant.items() if v),
        key=lambda kv: abs(kv[1] - 0.5),
    )
    return {
        "rows": len(rows),
        "rows_evaluated": len(ratios),
        "rows_skipped": skipped,
        "mean_hamming_ratio": mean(ratios) if ratios else None,
        "min_hamming_ratio": min(ratios) if ratios else None,
        "max_hamming_ratio": max(ratios) if ratios else None,
        "best_variants_top3": [{"variant": k, "mean_hamming_ratio": v} for k, v in ranked[:3]],
        "worst_variants_top3": [{"variant": k, "mean_hamming_ratio": v} for k, v in ranked[-3:]],
    }


def _summarize_periodicity(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if not rows:
        return {"variants_total": 0}
    lag_vals = [_to_float(r.get("lag_match_ratio")) for r in rows]
    lag_vals = [x for x in lag_vals if x is not None]
    repeats = [_to_int(r.get("repeated_chunk_hash_count")) for r in rows]
    repeats = [x for x in repeats if x is not None]
    periodic_hits = sum(1 for r in rows if _to_int(r.get("detected_prefix_period_bytes")) not in (None, 0))
    return {
        "variants_total": len(rows),
        "mean_lag_match_ratio": mean(lag_vals) if lag_vals else None,
        "max_lag_match_ratio": max(lag_vals) if lag_vals else None,
        "mean_repeated_chunk_hash_count": mean(repeats) if repeats else None,
        "max_repeated_chunk_hash_count": max(repeats) if repeats else None,
        "detected_prefix_period_variants": periodic_hits,
    }


def _build_overview_rows(nist: Dict[str, Any], avalanche: Dict[str, Any], periodicity: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [
        {"domain": "nist", "metric": "variants_total", "value": nist.get("variants_total")},
        {"domain": "nist", "metric": "variants_with_any_fail", "value": nist.get("variants_with_any_fail")},
        {"domain": "nist", "metric": "variants_fail_share", "value": nist.get("variants_fail_share")},
        {"domain": "nist", "metric": "mean_passed_tests_per_variant", "value": nist.get("mean_passed_tests_per_variant")},
        {"domain": "nist", "metric": "core_tests_count_excluding_template", "value": nist.get("core_tests_count_excluding_template")},
        {"domain": "nist", "metric": "core_tests_with_any_fail_excluding_template", "value": nist.get("core_tests_with_any_fail_excluding_template")},
        {"domain": "avalanche", "metric": "rows_evaluated", "value": avalanche.get("rows_evaluated")},
        {"domain": "avalanche", "metric": "rows_skipped", "value": avalanche.get("rows_skipped")},
        {"domain": "avalanche", "metric": "mean_hamming_ratio", "value": avalanche.get("mean_hamming_ratio")},
        {"domain": "periodicity", "metric": "variants_total", "value": periodicity.get("variants_total")},
        {"domain": "periodicity", "metric": "mean_lag_match_ratio", "value": periodicity.get("mean_lag_match_ratio")},
        {"domain": "periodicity", "metric": "detected_prefix_period_variants", "value": periodicity.get("detected_prefix_period_variants")},
    ]
    return rows


def _render_markdown(nist: Dict[str, Any], avalanche: Dict[str, Any], periodicity: Dict[str, Any], usability: Dict[str, Any] | None = None) -> str:
    lines: List[str] = []
    lines.append("# BA2 Evaluation Summary")
    lines.append("")
    lines.append("## NIST")
    lines.append(f"- Variants total: {nist.get('variants_total')}")
    lines.append(f"- Variants with any failed NIST test: {nist.get('variants_with_any_fail')} ({nist.get('variants_fail_share')})")
    lines.append(f"- Mean passed tests per variant: {nist.get('mean_passed_tests_per_variant')}")
    lines.append(f"- Mean failed tests per variant: {nist.get('mean_failed_tests_per_variant')}")
    lines.append(
        f"- Core tests (excluding template tests): {nist.get('core_tests_with_any_fail_excluding_template')} with fails out of {nist.get('core_tests_count_excluding_template')}"
    )
    lines.append("- Worst tests by fail count:")
    for row in nist.get("worst_tests", []):
        lines.append(f"- {row.get('test_name')}: n_fail={row.get('n_fail')}, pass_rate={row.get('pass_rate')}")
    hotspots = nist.get("fail_hotspots", {})
    lines.append("- Fail hotspots (mean failed tests per variant):")
    for entry in hotspots.get("memory_type_top3", []):
        lines.append(f"- memory_type={entry.get('group')}: {entry.get('mean_failed_tests')}")
    for entry in hotspots.get("seed_strategy_top3", []):
        lines.append(f"- seed_strategy={entry.get('group')}: {entry.get('mean_failed_tests')}")
    for entry in hotspots.get("quant_k_top3", []):
        lines.append(f"- quant_k={entry.get('group')}: {entry.get('mean_failed_tests')}")
    lines.append("")
    lines.append("## Avalanche")
    lines.append(f"- Rows evaluated: {avalanche.get('rows_evaluated')}")
    lines.append(f"- Rows skipped: {avalanche.get('rows_skipped')}")
    lines.append(f"- Mean hamming ratio: {avalanche.get('mean_hamming_ratio')}")
    lines.append(f"- Min/Max hamming ratio: {avalanche.get('min_hamming_ratio')} / {avalanche.get('max_hamming_ratio')}")
    lines.append("")
    lines.append("## Periodicity")
    lines.append(f"- Variants total: {periodicity.get('variants_total')}")
    lines.append(f"- Mean lag match ratio: {periodicity.get('mean_lag_match_ratio')}")
    lines.append(f"- Max lag match ratio: {periodicity.get('max_lag_match_ratio')}")
    lines.append(f"- Detected prefix period variants: {periodicity.get('detected_prefix_period_variants')}")
    if usability is not None:
        lines.append("")
        lines.append("## Usability")
        lines.append(f"- Runs total: {usability.get('runs_total')}")
        lines.append(f"- Success rate: {usability.get('success_rate')}")
        lines.append(f"- Duration median/p95 (s): {usability.get('duration_median_s')} / {usability.get('duration_p95_s')}")
        lines.append(f"- Reproducibility match rate: {usability.get('repro_match_rate')}")
        lines.append(f"- Mean failed attempts before success: {usability.get('mean_failed_attempts_before_success')}")
    lines.append("")
    lines.append("## BA2 Notes")
    lines.append("- Use NIST fail clusters to discuss parameter sensitivity (`dt`, `warmup`, `quant_k`, seed strategy, memory model).")
    lines.append("- Use avalanche deviation from 0.5 as diffusion indicator.")
    lines.append("- Use periodicity indicators as finite-precision risk signal.")
    return "\n".join(lines) + "\n"


def run_ba2_eval(
    nist_runs_csv: Path,
    nist_summary_csv: Path,
    avalanche_csv: Path,
    periodicity_csv: Path,
    usability_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    nist_runs = _read_csv(nist_runs_csv)
    nist_summary_rows = _read_csv(nist_summary_csv)
    avalanche_rows = _read_csv(avalanche_csv)
    periodicity_rows = _read_csv(periodicity_csv)
    nist = _summarize_nist(nist_runs, nist_summary_rows)
    avalanche = _summarize_avalanche(avalanche_rows)
    periodicity = _summarize_periodicity(periodicity_rows)
    overview = _build_overview_rows(nist, avalanche, periodicity)
    if usability_summary is not None:
        overview.extend(
            [
                {"domain": "usability", "metric": "runs_total", "value": usability_summary.get("runs_total")},
                {"domain": "usability", "metric": "success_rate", "value": usability_summary.get("success_rate")},
                {"domain": "usability", "metric": "duration_median_s", "value": usability_summary.get("duration_median_s")},
                {"domain": "usability", "metric": "repro_match_rate", "value": usability_summary.get("repro_match_rate")},
            ]
        )
    markdown = _render_markdown(nist, avalanche, periodicity, usability_summary)
    payload: Dict[str, Any] = {
        "nist": nist,
        "avalanche": avalanche,
        "periodicity": periodicity,
        "overview_rows": overview,
        "markdown": markdown,
    }
    if usability_summary is not None:
        payload["usability"] = usability_summary
    return payload


def write_ba2_eval_overview_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["domain", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_ba2_eval_markdown(path: Path, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def write_ba2_eval_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
