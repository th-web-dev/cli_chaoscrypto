from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    base = Path("out/ba2")
    compare_csv = base / "platform_compare_wsl_vs_windows.csv"
    rows = _read_csv(compare_csv)

    status_counts = Counter((r.get("status") or "unknown") for r in rows)
    by_engine = defaultdict(lambda: Counter())
    for r in rows:
        engine = r.get("chaos_engine") or "unknown"
        by_engine[engine][r.get("status") or "unknown"] += 1

    out_dir = base / "fig_performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig5_platform_divergence.png"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    labels = ["match", "diverged", "missing_reference", "missing_candidate"]
    vals = [status_counts.get(k, 0) for k in labels]
    colors = ["#16a34a", "#dc2626", "#f59e0b", "#0ea5e9"]
    axes[0].bar(labels, vals, color=colors)
    axes[0].set_title("Platform Compare Status Counts")
    axes[0].set_ylabel("variants")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    engines = sorted(by_engine.keys())
    match_vals = [by_engine[e].get("match", 0) for e in engines]
    div_vals = [by_engine[e].get("diverged", 0) for e in engines]
    axes[1].bar(engines, match_vals, color="#22c55e", label="match")
    axes[1].bar(engines, div_vals, bottom=match_vals, color="#ef4444", label="diverged")
    axes[1].set_title("Status by Chaos Engine")
    axes[1].set_ylabel("variants")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    main()

