from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _parse_float(val: str | None) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _parse_int(val: str | None) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if "seed_strategy" not in row or not row.get("seed_strategy"):
                row["seed_strategy"] = "neighborhood3"
            rows.append(dict(row))
    return rows


@dataclass
class BenchAgg:
    key: Tuple
    count: int
    mean_t_keystream: float
    std_t_keystream: float
    mean_throughput: float
    std_throughput: float
    sample_hash: str | None
    params: Dict[str, Any]


@dataclass
class AnalyzeAgg:
    key: Tuple
    count: int
    metrics: Dict[str, float | None]
    params: Dict[str, Any]
    keystream_sha256: str | None


def _group_bench(rows: List[Dict[str, Any]]) -> List[BenchAgg]:
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (
            _parse_float(r.get("dt")),
            _parse_int(r.get("warmup")),
            _parse_float(r.get("quant_k")),
            _parse_int(r.get("size")),
            _parse_float(r.get("scale")),
            r.get("seed_strategy") or "neighborhood3",
            _parse_int(r.get("coord_x")),
            _parse_int(r.get("coord_y")),
        )
        groups[key].append(r)

    aggs: List[BenchAgg] = []
    for key, lst in groups.items():
        t_keystream_vals = [_parse_float(x.get("t_keystream_s")) for x in lst if _parse_float(x.get("t_keystream_s")) is not None]
        throughput_vals = [_parse_float(x.get("throughput_keystream_bps")) for x in lst if _parse_float(x.get("throughput_keystream_bps")) is not None]

        def mean_std(values: List[float]) -> Tuple[float, float]:
            if not values:
                return 0.0, 0.0
            m = sum(values) / len(values)
            var = sum((v - m) ** 2 for v in values) / len(values)
            return m, var**0.5

        mean_t, std_t = mean_std(t_keystream_vals)
        mean_tp, std_tp = mean_std(throughput_vals)

        sample = lst[0]
        params = {
            "dt": key[0],
            "warmup": key[1],
            "quant_k": key[2],
            "size": key[3],
            "scale": key[4],
            "seed_strategy": key[5],
            "coord_x": key[6],
            "coord_y": key[7],
            "profile": sample.get("profile"),
            "nbytes": _parse_int(sample.get("nbytes")),
        }
        aggs.append(
            BenchAgg(
                key=key,
                count=len(lst),
                mean_t_keystream=mean_t,
                std_t_keystream=std_t,
                mean_throughput=mean_tp,
                std_throughput=std_tp,
                sample_hash=sample.get("keystream_sha256"),
                params=params,
            )
        )
    return aggs


def _group_analyze(rows: List[Dict[str, Any]]) -> List[AnalyzeAgg]:
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (
            _parse_float(r.get("dt")),
            _parse_int(r.get("warmup")),
            _parse_float(r.get("quant_k")),
            _parse_int(r.get("size")),
            _parse_float(r.get("scale")),
            r.get("seed_strategy") or "neighborhood3",
            _parse_int(r.get("coord_x")),
            _parse_int(r.get("coord_y")),
        )
        groups[key].append(r)

    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    aggs: List[AnalyzeAgg] = []
    for key, lst in groups.items():
        def collect(field: str) -> List[float]:
            vals = []
            for x in lst:
                v = _parse_float(x.get(field))
                if v is not None:
                    vals.append(v)
            return vals

        metrics = {
            "bit_ones_ratio": mean(collect("bit_ones_ratio")),
            "runs_norm_diff": mean(collect("runs_norm_diff")),
            "byte_chi2_norm": mean(collect("byte_chi2_norm")),
            "autocorr_lag_1": mean(collect("autocorr_lag_1")),
        }
        sample = lst[0]
        # Hamming weights
        for field in ["hw_win_mean", "hw_win_std", "hw_win_min", "hw_win_max"]:
            vals = collect(field)
            metrics[field] = mean(vals) if vals else None

        params = {
            "dt": key[0],
            "warmup": key[1],
            "quant_k": key[2],
            "size": key[3],
            "scale": key[4],
            "seed_strategy": key[5],
            "coord_x": key[6],
            "coord_y": key[7],
            "profile": sample.get("profile"),
            "nbytes": _parse_int(sample.get("nbytes")),
        }
        aggs.append(
            AnalyzeAgg(
                key=key,
                count=len(lst),
                metrics=metrics,
                params=params,
                keystream_sha256=sample.get("keystream_sha256"),
            )
        )
    return aggs


def _render_markdown(
    bench_aggs: List[BenchAgg],
    analyze_aggs: List[AnalyzeAgg],
    bench_path: Path,
    analysis_path: Path,
    timestamp: Optional[str],
    plots: List[str],
) -> str:
    lines = []
    lines.append("# ChaosCrypto WP2 – Report")
    if timestamp:
        lines.append(f"_Generated: {timestamp} UTC_")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Benchmark CSV: `{bench_path}` ({len(bench_aggs)} variants aggregated)")
    lines.append(f"- Analyze CSV: `{analysis_path}` ({len(analyze_aggs)} variants aggregated)")
    lines.append("- Token: not stored; only fingerprints in source CSV")
    lines.append("")

    def params_summary(aggs):
        if not aggs:
            return {}
        sample = aggs[0].params
        varying = {}
        keys = ["dt", "warmup", "quant_k", "size", "scale"]
        for k in keys:
            vals = sorted({a.params.get(k) for a in aggs})
            varying[k] = vals
        return sample, varying

    lines.append("## Scope")
    if bench_aggs:
        sample, varying = params_summary(bench_aggs)
        lines.append(f"- Profile: {sample.get('profile')}")
        lines.append(f"- Coord: ({sample.get('coord_x')},{sample.get('coord_y')})")
        lines.append(f"- nbytes: {sample.get('nbytes')}")
        lines.append(f"- Seed strategies: {sorted({a.params.get('seed_strategy','neighborhood3') for a in bench_aggs})}")
        lines.append(f"- Varying parameters:")
        for k, vals in varying.items():
            lines.append(f"  - {k}: {vals}")
    lines.append("")

    # Benchmark summary
    lines.append("## Benchmark Summary")
    top_by_tp = sorted(bench_aggs, key=lambda a: a.mean_throughput or 0, reverse=True)[:5]
    lines.append("Top throughput overall (mean over repeats):")
    lines.append("")
    lines.append("| dt | warmup | quant_k | size | scale | seed_strategy | mean_t_keystream_s | mean_tp_bps | keystream_sha256 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for a in top_by_tp:
        lines.append(
            f"| {a.params['dt']} | {a.params['warmup']} | {a.params['quant_k']} | {a.params['size']} | {a.params['scale']} | {a.params.get('seed_strategy','')} | "
            f"{a.mean_t_keystream:.6g} | {a.mean_throughput:.3g} | {a.sample_hash or ''} |"
        )
    lines.append("")
    lines.append("Top throughput per seed_strategy:")
    lines.append("")
    lines.append("| seed_strategy | dt | warmup | quant_k | size | scale | mean_tp_bps | keystream_sha256 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    seed_groups: Dict[str, List[BenchAgg]] = defaultdict(list)
    for a in bench_aggs:
        seed_groups[a.params.get("seed_strategy", "neighborhood3")].append(a)
    for seed, aggs in seed_groups.items():
        best = max(aggs, key=lambda x: x.mean_throughput or 0)
        lines.append(
            f"| {seed} | {best.params['dt']} | {best.params['warmup']} | {best.params['quant_k']} | {best.params['size']} | {best.params['scale']} | "
            f"{best.mean_throughput:.3g} | {best.sample_hash or ''} |"
        )
    lines.append("")

    # Analyze summary
    lines.append("## Analyze Summary")
    if analyze_aggs:
        ratios = [a.metrics.get("bit_ones_ratio", 0.0) or 0.0 for a in analyze_aggs]
        runs = [a.metrics.get("runs_norm_diff", 0.0) or 0.0 for a in analyze_aggs]
        chi = [a.metrics.get("byte_chi2_norm", 0.0) or 0.0 for a in analyze_aggs]
        ac1 = [a.metrics.get("autocorr_lag_1", 0.0) or 0.0 for a in analyze_aggs]

        def stats(vals):
            return min(vals), sum(vals) / len(vals), max(vals)

        lines.append(f"- Bit ones ratio min/mean/max: {tuple(round(x,6) for x in stats(ratios))}")
        lines.append(f"- Runs norm diff min/mean/max: {tuple(round(x,6) for x in stats(runs))}")
        lines.append(f"- Byte chi2 norm min/mean/max: {tuple(round(x,6) for x in stats(chi))}")
        lines.append(f"- Autocorr lag1 min/mean/max: {tuple(round(x,6) for x in stats(ac1))}")
        lines.append("")
        lines.append("Per seed_strategy (mean values):")
        lines.append("| seed_strategy | bit_ones_ratio | byte_chi2_norm | runs_norm_diff | autocorr_lag_1 |")
        lines.append("|---|---|---|---|---|")
        analyze_seed_groups: Dict[str, List[AnalyzeAgg]] = defaultdict(list)
        for a in analyze_aggs:
            analyze_seed_groups[a.params.get("seed_strategy", "neighborhood3")].append(a)
        for seed, aggs in analyze_seed_groups.items():
            bs = [a.metrics.get("bit_ones_ratio") or 0.0 for a in aggs]
            cs = [a.metrics.get("byte_chi2_norm") or 0.0 for a in aggs]
            rs = [a.metrics.get("runs_norm_diff") or 0.0 for a in aggs]
            acs = [a.metrics.get("autocorr_lag_1") or 0.0 for a in aggs]
            lines.append(
                f"| {seed} | {sum(bs)/len(bs):.6f} | {sum(cs)/len(cs):.6f} | {sum(rs)/len(rs):.6f} | {sum(acs)/len(acs):.6f} |"
            )
        lines.append("")
        lines.append("| dt | warmup | quant_k | size | scale | seed_strategy | bit_ones_ratio | byte_chi2_norm | runs_norm_diff | autocorr_lag_1 | keystream_sha256 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for a in sorted(analyze_aggs, key=lambda x: (x.params["dt"], x.params["warmup"], x.params["quant_k"], x.params.get("seed_strategy"))):
            lines.append(
                f"| {a.params['dt']} | {a.params['warmup']} | {a.params['quant_k']} | {a.params['size']} | {a.params['scale']} | {a.params.get('seed_strategy','')} | "
                f"{(a.metrics.get('bit_ones_ratio') or 0):.6f} | {(a.metrics.get('byte_chi2_norm') or 0):.6f} | "
                f"{(a.metrics.get('runs_norm_diff') or 0):.6f} | {(a.metrics.get('autocorr_lag_1') or 0):.6f} | {a.keystream_sha256 or ''} |"
            )
    lines.append("")

    # Best candidates
    lines.append("## Best Candidates (heuristic score)")
    best_overall, best_per_seed = _score_candidates(bench_aggs, analyze_aggs)
    if best_overall:
        lines.append("Top 5 overall:")
        lines.append("| dt | warmup | quant_k | size | scale | seed_strategy | score | perf_score | rand_score | bit_ones_ratio | autocorr_lag_1 | runs_norm_diff | byte_chi2_norm |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for b in best_overall:
            lines.append(
                f"| {b['dt']} | {b['warmup']} | {b['quant_k']} | {b['size']} | {b['scale']} | {b.get('seed_strategy','')} | "
                f"{b['total_score']:.6f} | {b['perf_score']:.6f} | {b['rand_score']:.6f} | "
                f"{b['bit_ones_ratio']:.6f} | {b['autocorr_lag_1']:.6f} | {b['runs_norm_diff']:.6f} | {b['byte_chi2_norm']:.6f} |"
            )
        lines.append("")
    if best_per_seed:
        lines.append("Top 3 per seed_strategy:")
        lines.append("| seed_strategy | dt | warmup | quant_k | size | scale | score | bit_ones_ratio | autocorr_lag_1 | runs_norm_diff | byte_chi2_norm |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for seed, items in best_per_seed.items():
            for b in items:
                lines.append(
                    f"| {seed} | {b['dt']} | {b['warmup']} | {b['quant_k']} | {b['size']} | {b['scale']} | "
                    f"{b['total_score']:.6f} | {b['bit_ones_ratio']:.6f} | {b['autocorr_lag_1']:.6f} | {b['runs_norm_diff']:.6f} | {b['byte_chi2_norm']:.6f} |"
                )
    else:
        lines.append("- Not enough data to score.")
    lines.append("")

    if plots:
        lines.append("## Plots")
        for p in plots:
            lines.append(f"![]({p})")
        lines.append("")

    lines.append("## Appendix")
    lines.append("- CSV columns: benchmark includes timing/throughput; analyze includes keystream statistics.")
    lines.append("- Reproducibility: same config → identical hashes/metrics.")
    return "\n".join(lines)


def _score_candidates(bench_aggs: List[BenchAgg], analyze_aggs: List[AnalyzeAgg]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    bench_map = {a.key: a for a in bench_aggs}
    results = []

    # perf ranking
    perf_vals = [a.mean_throughput for a in bench_aggs if a.mean_throughput]
    max_perf = max(perf_vals) if perf_vals else 0.0

    for a in analyze_aggs:
        b = bench_map.get(a.key)
        if not b:
            continue
        perf_score = (b.mean_throughput / max_perf) if max_perf else 0.0
        ones_dev = abs((a.metrics.get("bit_ones_ratio") or 0) - 0.5)
        ac1 = abs(a.metrics.get("autocorr_lag_1") or 0)
        runs_dev = abs(a.metrics.get("runs_norm_diff") or 0)
        chi_norm = a.metrics.get("byte_chi2_norm")
        chi_dev = abs((chi_norm or 1.0) - 1.0)
        raw_rand = ones_dev + ac1 + runs_dev + chi_dev
        rand_score = 1.0 / (1.0 + raw_rand)
        total = 0.6 * rand_score + 0.4 * perf_score
        results.append(
            {
                "dt": a.params["dt"],
                "warmup": a.params["warmup"],
                "quant_k": a.params["quant_k"],
                "size": a.params["size"],
                "scale": a.params["scale"],
                "seed_strategy": a.params.get("seed_strategy"),
                "perf_score": perf_score,
                "rand_score": rand_score,
                "total_score": total,
                "bit_ones_ratio": a.metrics.get("bit_ones_ratio") or 0.0,
                "autocorr_lag_1": a.metrics.get("autocorr_lag_1") or 0.0,
                "runs_norm_diff": a.metrics.get("runs_norm_diff") or 0.0,
                "byte_chi2_norm": a.metrics.get("byte_chi2_norm") or 0.0,
            }
        )
    results_sorted = sorted(results, key=lambda r: r["total_score"], reverse=True)[:5]
    per_seed: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        seed = r.get("seed_strategy") or "neighborhood3"
        per_seed.setdefault(seed, []).append(r)
    for seed, lst in per_seed.items():
        per_seed[seed] = sorted(lst, key=lambda r: r["total_score"], reverse=True)[:3]
    return results_sorted, per_seed


def _plot_lines(
    data: Dict[Tuple[float, int, float, str | None], float],
    title: str,
    ylabel: str,
    out_path: Path,
):
    # data key: (dt, warmup, quant_k, seed_strategy) mapped to mean value
    # Heuristik: wenn <=3 seed strategies, je seed eigener Plot; sonst alles zusammen mit Legend.
    seed_values = sorted({k[3] or "neighborhood3" for k in data})
    plot_refs: List[str] = []

    def make_plot(filtered: Dict[Tuple[float, int, float, str | None], float], suffix: str = ""):
        grouped: Dict[Tuple[float, float], List[Tuple[int, float, str | None]]] = defaultdict(list)
        for (dt, warmup, quant_k, seed_strategy), val in filtered.items():
            grouped[(dt, quant_k)].append((warmup, val, seed_strategy))

        for (dt, quant_k), lst in grouped.items():
            lst_sorted = sorted(lst, key=lambda x: x[0])
            xs = [w for w, _, _ in lst_sorted]
            ys = [v for _, v, _ in lst_sorted]
            label = f"quant_k={quant_k}"
            plt.plot(xs, ys, label=label)
            plt.xlabel("warmup")
            plt.ylabel(ylabel)
            plt.title(f"{title} (dt={dt}{suffix})")
            if len(grouped) > 1:
                plt.legend()
            seed_name = suffix.replace(" seed=", "") or "all"
            out_file = out_path / f"{title.lower().replace(' ', '_')}_dt{str(dt).replace('.','p')}_q{str(quant_k).replace('.','p')}_{seed_name.replace(' ','_').replace('(','').replace(')','')}.png"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close()
            plot_refs.append(str(out_file))

    if len(seed_values) <= 3:
        for seed in seed_values:
            filtered = {k: v for k, v in data.items() if (k[3] or "neighborhood3") == seed}
            make_plot(filtered, suffix=f" seed={seed}")
    else:
        make_plot(data, suffix="")
    return plot_refs


def generate_report(
    bench_csv: Path,
    analysis_csv: Path,
    bench_json: Path | None,
    analysis_json: Path | None,
    out_md: Path,
    plots_dir: Path | None = None,
    include_timestamp: bool = True,
    json_summary: Path | None = None,
) -> Dict[str, Any]:
    bench_rows = _read_csv(bench_csv)
    analyze_rows = _read_csv(analysis_csv)

    bench_aggs = _group_bench(bench_rows)
    analyze_aggs = _group_analyze(analyze_rows)

    timestamp = datetime.now(timezone.utc).isoformat() if include_timestamp else None

    plot_refs: List[str] = []
    if plots_dir:
        # Throughput vs warmup
        tp_data: Dict[Tuple[float, int, float, str | None], float] = {}
        for a in bench_aggs:
            tp_data[(a.params["dt"], a.params["warmup"], a.params["quant_k"], a.params.get("seed_strategy"))] = a.mean_throughput
        plot_refs.extend(
            _plot_lines(tp_data, "Bench Throughput", "throughput_keystream_bps", plots_dir / "bench")
        )

        # Analyze plots
        def collect_metric(metric_name: str) -> Dict[Tuple[float, int, float, str | None], float]:
            data: Dict[Tuple[float, int, float, str | None], float] = {}
            for a in analyze_aggs:
                val = a.metrics.get(metric_name)
                if val is not None:
                    data[(a.params["dt"], a.params["warmup"], a.params["quant_k"], a.params.get("seed_strategy"))] = val
            return data

        bit_data = collect_metric("bit_ones_ratio")
        if bit_data:
            plot_refs.extend(_plot_lines(bit_data, "Analyze Bit Balance", "bit_ones_ratio", plots_dir / "analyze_bit"))
        ac_data = collect_metric("autocorr_lag_1")
        if ac_data:
            plot_refs.extend(_plot_lines(ac_data, "Analyze Autocorr Lag1", "autocorr_lag_1", plots_dir / "analyze_autocorr"))
        chi_data = collect_metric("byte_chi2_norm")
        if chi_data:
            plot_refs.extend(_plot_lines(chi_data, "Analyze Byte Chi2 Norm", "byte_chi2_norm", plots_dir / "analyze_chi2"))

    md = _render_markdown(bench_aggs, analyze_aggs, bench_csv, analysis_csv, timestamp, plot_refs)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")

    summary = {
        "bench_variants": len(bench_aggs),
        "analyze_variants": len(analyze_aggs),
        "bench_csv": str(bench_csv),
        "analysis_csv": str(analysis_csv),
        "timestamp": timestamp,
        "plots": plot_refs,
    }
    if json_summary:
        json_summary.parent.mkdir(parents=True, exist_ok=True)
        json_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
