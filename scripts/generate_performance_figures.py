from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v: str | None) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _variant_key(r: dict) -> str:
    return (
        f"{r['chaos_engine']}|{r['memory_type']}|{r['seed_strategy']}|"
        f"w{r['warmup']}|q{r['quant_k']}"
    )


def _variant_signature(r: dict) -> tuple:
    return (
        r.get("chaos_engine", ""),
        r.get("memory_type", ""),
        r.get("seed_strategy", ""),
        r.get("warmup", ""),
        r.get("quant_k", ""),
        r.get("dt", ""),
    )


def _build_variant_map(rows: list[dict]) -> tuple[dict[tuple, str], list[tuple[str, tuple]]]:
    unique = sorted({_variant_signature(r) for r in rows})
    mapping = {sig: f"V{idx:02d}" for idx, sig in enumerate(unique, start=1)}
    ordered = [(mapping[sig], sig) for sig in unique]
    return mapping, ordered


def write_variant_legend(rows: list[dict], out_csv: Path, out_tex: Path) -> None:
    _mapping, ordered = _build_variant_map(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant_id", "chaos_engine", "memory_type", "seed_strategy", "warmup", "quant_k", "dt"])
        for vid, sig in ordered:
            w.writerow([vid, sig[0], sig[1], sig[2], sig[3], sig[4], sig[5]])
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{llllll}")
    lines.append("    \\hline")
    lines.append("    ID & Engine & Memory & Seed & warmup & quant\\_k \\\\")
    lines.append("    \\hline")
    for vid, sig in ordered:
        lines.append(f"    {vid} & {sig[0]} & {sig[1]} & {sig[2]} & {sig[3]} & {sig[4]} \\\\")
    lines.append("    \\hline")
    lines.append("  \\end{tabular}")
    lines.append("  \\caption{Varianten-Legende für die Performance-Abbildungen.}")
    lines.append("  \\label{tab:perf:variant-legend}")
    lines.append("\\end{table}")
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def fig_phase_times(benchmark_csv: Path, out_path: Path) -> None:
    rows = _read_csv(benchmark_csv)
    field = [_to_float(r.get("t_field_s")) for r in rows]
    seed = [_to_float(r.get("t_seed_s")) for r in rows]
    ks = [_to_float(r.get("t_keystream_s")) for r in rows]
    xor = [_to_float(r.get("t_xor_s")) for r in rows]
    values = [
        mean([x for x in field if x is not None]),
        mean([x for x in seed if x is not None]),
        mean([x for x in ks if x is not None]),
        mean([x for x in xor if x is not None]),
    ]
    labels = ["t_field_s", "t_seed_s", "t_keystream_s", "t_xor_s"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color=["#475569", "#0ea5e9", "#22c55e", "#f59e0b"])
    ax.set_title("Mean Runtime per Phase")
    ax.set_ylabel("seconds")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def fig_throughput_by_variant(benchmark_csv: Path, out_path: Path) -> None:
    rows = _read_csv(benchmark_csv)
    variant_map, ordered = _build_variant_map(rows)
    grouped = defaultdict(list)
    for r in rows:
        tp = _to_float(r.get("throughput_keystream_bps"))
        if tp is not None:
            grouped[variant_map[_variant_signature(r)]].append(tp)
    labels = [vid for vid, _sig in ordered]
    vals = [mean(grouped[k]) if grouped.get(k) else 0.0 for k in labels]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(labels)), vals, color="#2563eb")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel("throughput_keystream_bps")
    ax.set_xlabel("variant ID")
    ax.set_title("Mean Keystream Throughput per Variant (V01..V16)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def fig_param_tradeoff_quantk_warmup(benchmark_csv: Path, out_path: Path) -> None:
    rows = _read_csv(benchmark_csv)
    grouped = defaultdict(list)
    for r in rows:
        key = (r.get("quant_k", "?"), r.get("warmup", "?"))
        tp = _to_float(r.get("throughput_keystream_bps"))
        if tp is not None:
            grouped[key].append(tp)
    keys = sorted(grouped.keys(), key=lambda x: (float(x[0]), int(x[1])))
    labels = [f"q={q}, w={w}" for q, w in keys]
    vals = [mean(grouped[k]) for k in keys]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, vals, color="#14b8a6")
    ax.set_ylabel("throughput_keystream_bps")
    ax.set_title("Throughput Trade-off by quant_k and warmup")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def fig_aes_hmac_overhead(benchmark_csv: Path, out_path: Path) -> None:
    rows = _read_csv(benchmark_csv)
    chaos = []
    aes = []
    hmac = []
    ratio_chaos_aes = []
    ratio_xor_hmac = []
    for r in rows:
        c = _to_float(r.get("throughput_keystream_bps"))
        a = _to_float(r.get("throughput_aes256ctr_bps"))
        h = _to_float(r.get("throughput_hmac_bps"))
        ra = _to_float(r.get("throughput_ratio_chaos_to_aes256ctr"))
        rh = _to_float(r.get("throughput_ratio_xor_to_hmac"))
        if c is not None:
            chaos.append(c)
        if a is not None:
            aes.append(a)
        if h is not None:
            hmac.append(h)
        if ra is not None:
            ratio_chaos_aes.append(ra)
        if rh is not None:
            ratio_xor_hmac.append(rh)

    labels = ["Chaos keystream", "AES-CTR", "HMAC"]
    vals = [mean(chaos), mean(aes), mean(hmac)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(labels, vals, color=["#2563eb", "#f97316", "#16a34a"])
    axes[0].set_title("Mean Throughput Comparison")
    axes[0].set_ylabel("bps")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(
        ["chaos/aes ratio", "xor/hmac ratio"],
        [mean(ratio_chaos_aes), mean(ratio_xor_hmac)],
        color=["#7c3aed", "#0ea5e9"],
    )
    axes[1].set_title("Mean Throughput Ratios")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    base = Path("out/ba2")
    out = base / "fig_performance"
    out.mkdir(parents=True, exist_ok=True)
    bench = base / "benchmark.csv"
    rows = _read_csv(bench)
    write_variant_legend(
        rows,
        out / "fig5_variant_legend.csv",
        out / "fig5_variant_legend_table.tex",
    )

    fig_phase_times(bench, out / "fig5_phase_times.png")
    fig_throughput_by_variant(bench, out / "fig5_throughput_by_variant.png")
    fig_param_tradeoff_quantk_warmup(bench, out / "fig5_tradeoff_quantk_warmup.png")
    fig_aes_hmac_overhead(bench, out / "fig5_aes_hmac_overhead.png")
    print(f"Generated performance figures in: {out}")


if __name__ == "__main__":
    main()
