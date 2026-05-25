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
    mapping, ordered = _build_variant_map(rows)
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
    lines.append("  \\caption{Varianten-Legende für die NIST-Heatmap.}")
    lines.append("  \\label{tab:sec:nist-variant-legend}")
    lines.append("\\end{table}")
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def plot_nist_fail_heatmap(nist_runs_csv: Path, out_path: Path) -> None:
    rows = _read_csv(nist_runs_csv)
    variant_map, _ordered = _build_variant_map(rows)
    tests = [
        "frequency_monobit",
        "block_frequency",
        "runs",
        "longest_run_of_ones",
        "binary_matrix_rank",
        "discrete_fourier_transform",
        "non_overlapping_template_matching",
        "overlapping_template_matching",
        "maurers_universal",
        "linear_complexity",
        "serial",
        "approximate_entropy",
        "cumulative_sums",
        "random_excursions",
        "random_excursions_variant",
    ]
    variants = []
    matrix = []
    for r in rows:
        label = variant_map[_variant_signature(r)]
        variants.append(label)
        vals = []
        for t in tests:
            st = (r.get(f"nist_{t}_status") or "").strip().lower()
            if st == "pass":
                vals.append(0.0)
            elif st == "fail":
                vals.append(1.0)
            else:
                vals.append(0.5)  # skip
        matrix.append(vals)

    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(tests)))
    ax.set_xticklabels(tests, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants, fontsize=8)
    ax.set_title("NIST Outcomes by Variant ID (pass=0, skip=0.5, fail=1)")
    ax.set_xlabel("NIST test")
    ax.set_ylabel("variant ID")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Outcome scale")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_nist_fail_by_quant_k(nist_runs_csv: Path, out_path: Path) -> None:
    rows = _read_csv(nist_runs_csv)
    grouped = defaultdict(list)
    for r in rows:
        q = r.get("quant_k", "unknown")
        fails = _to_float(r.get("nist_failed_count")) or 0.0
        grouped[q].append(fails)
    labels = sorted(grouped.keys(), key=lambda x: float(x))
    vals = [mean(grouped[k]) for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals, color="#3b82f6")
    ax.set_title("Mean NIST Fail Count by quant_k")
    ax.set_xlabel("quant_k")
    ax.set_ylabel("mean failed tests per variant")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_avalanche_hist(avalanche_csv: Path, out_path: Path) -> None:
    rows = _read_csv(avalanche_csv)
    vals = []
    for r in rows:
        if (r.get("perturbation_skipped") or "").lower() == "true":
            continue
        v = _to_float(r.get("hamming_distance_ratio"))
        if v is not None:
            vals.append(v)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vals, bins=30, color="#10b981", alpha=0.85, edgecolor="white")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.2, label="ideal 0.5")
    ax.set_title("Avalanche Distribution")
    ax.set_xlabel("hamming_distance_ratio")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_periodicity_scatter(periodicity_csv: Path, out_path: Path) -> None:
    rows = _read_csv(periodicity_csv)
    labels = []
    vals = []
    colors = []
    for r in rows:
        labels.append(f"{r['chaos_engine']}|{r['memory_type']}|w{r['warmup']}|{r['seed_strategy']}")
        vals.append(_to_float(r.get("lag_match_ratio")) or 0.0)
        colors.append("#2563eb" if r.get("chaos_engine") == "lorenz" else "#f59e0b")
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.scatter(range(len(vals)), vals, c=colors, s=30)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("lag_match_ratio")
    ax.set_title("Periodicity Indicator per Variant")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_reconstruction_by_engine(reconstruction_csv: Path, out_path: Path) -> None:
    rows = _read_csv(reconstruction_csv)
    grouped = defaultdict(list)
    for r in rows:
        engine = r.get("chaos_engine", "unknown")
        val = _to_float(r.get("recon_r2"))
        if val is not None:
            grouped[engine].append(val)
    labels = sorted(grouped.keys())
    data = [grouped[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, labels=labels)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_title("Reconstruction R² by Chaos Engine")
    ax.set_ylabel("recon_r2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    base = Path("out/ba2")
    out = base / "fig_security"
    out.mkdir(parents=True, exist_ok=True)
    nist_rows = _read_csv(base / "nist_runs.csv")
    write_variant_legend(
        nist_rows,
        out / "fig4_nist_variant_legend.csv",
        out / "fig4_nist_variant_legend_table.tex",
    )

    plot_nist_fail_heatmap(base / "nist_runs.csv", out / "fig4_nist_heatmap.png")
    plot_nist_fail_by_quant_k(base / "nist_runs.csv", out / "fig4_nist_fail_by_quant_k.png")
    plot_avalanche_hist(base / "avalanche_v2.csv", out / "fig4_avalanche_hist.png")
    plot_periodicity_scatter(base / "periodicity.csv", out / "fig4_periodicity_scatter.png")
    plot_reconstruction_by_engine(base / "reconstruction.csv", out / "fig4_reconstruction_r2_by_engine.png")

    print(f"Generated security figures in: {out}")


if __name__ == "__main__":
    main()
