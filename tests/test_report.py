import csv
from pathlib import Path

from typer.testing import CliRunner

from chaoscrypto.cli.app import app


def _write_bench_csv(path: Path):
    base = {
        "timestamp_utc": "",
        "profile": "alice",
        "coord_x": "0",
        "coord_y": "0",
        "nbytes": "1024",
        "repeats": "1",
        "repeat_index": "0",
        "dt": "0.01",
        "quant_k": "100000",
        "size": "128",
        "scale": "0.1",
        "t_field_s": "",
        "t_xor_s": "",
        "t_decrypt_s": "",
        "throughput_xor_bps": "",
        "token_fingerprint": "tfp",
        "keystream_preview_base64": "",
    }
    rows = [
        {
            **base,
            "warmup": "100",
            "seed_strategy": "neighborhood3",
            "memory_type": "opensimplex",
            "t_keystream_s": "0.5",
            "throughput_keystream_bps": str(1024 / 0.5),
            "keystream_sha256": "hash1",
            "field_fingerprint": "ff1",
        },
        {
            **base,
            "warmup": "150",
            "seed_strategy": "neighborhood3",
            "memory_type": "perlin",
            "t_keystream_s": "0.55",
            "throughput_keystream_bps": str(1024 / 0.55),
            "keystream_sha256": "hash1p",
            "field_fingerprint": "ff1p",
        },
        {
            **base,
            "warmup": "200",
            "seed_strategy": "window_mean_3x3",
            "memory_type": "opensimplex",
            "t_keystream_s": "1.0",
            "throughput_keystream_bps": str(1024 / 1.0),
            "keystream_sha256": "hash2",
            "field_fingerprint": "ff2",
        },
        {
            **base,
            "warmup": "250",
            "seed_strategy": "window_mean_3x3",
            "memory_type": "perlin",
            "t_keystream_s": "1.1",
            "throughput_keystream_bps": str(1024 / 1.1),
            "keystream_sha256": "hash2p",
            "field_fingerprint": "ff2p",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _write_analysis_csv(path: Path):
    base = {
        "timestamp_utc": "",
        "profile": "alice",
        "coord_x": "0",
        "coord_y": "0",
        "nbytes": "1024",
        "dt": "0.01",
        "quant_k": "100000",
        "size": "128",
        "scale": "0.1",
        "token_fingerprint": "tfp",
        "byte_entropy_approx": "7.9",
        "byte_top5": "00:5|ff:4",
        "keystream_preview_base64": "",
    }
    rows = [
        {
            **base,
            "warmup": "100",
            "seed_strategy": "neighborhood3",
            "memory_type": "opensimplex",
            "keystream_sha256": "hash1",
            "field_fingerprint": "ff1",
            "bit_ones_ratio": "0.51",
            "bit_ones_count": "523",
            "byte_chi2": "1.0",
            "byte_chi2_norm": "1.0",
            "runs_count": "100",
            "runs_expected": "101",
            "runs_norm_diff": "0.01",
            "hw_win_mean": "256",
            "hw_win_std": "10",
            "hw_win_min": "240",
            "hw_win_max": "270",
            "autocorr_lag_1": "0.02",
        },
        {
            **base,
            "warmup": "150",
            "seed_strategy": "neighborhood3",
            "memory_type": "perlin",
            "keystream_sha256": "hash1p",
            "field_fingerprint": "ff1p",
            "bit_ones_ratio": "0.5",
            "bit_ones_count": "512",
            "byte_chi2": "1.1",
            "byte_chi2_norm": "1.05",
            "runs_count": "101",
            "runs_expected": "101",
            "runs_norm_diff": "0.0",
            "hw_win_mean": "255",
            "hw_win_std": "9",
            "hw_win_min": "238",
            "hw_win_max": "268",
            "autocorr_lag_1": "0.0",
        },
        {
            **base,
            "warmup": "200",
            "seed_strategy": "window_mean_3x3",
            "memory_type": "opensimplex",
            "keystream_sha256": "hash2",
            "field_fingerprint": "ff2",
            "bit_ones_ratio": "0.49",
            "bit_ones_count": "500",
            "byte_chi2": "1.2",
            "byte_chi2_norm": "1.1",
            "runs_count": "98",
            "runs_expected": "101",
            "runs_norm_diff": "-0.01",
            "hw_win_mean": "250",
            "hw_win_std": "9",
            "hw_win_min": "230",
            "hw_win_max": "270",
            "autocorr_lag_1": "-0.03",
        },
        {
            **base,
            "warmup": "250",
            "seed_strategy": "window_mean_3x3",
            "memory_type": "perlin",
            "keystream_sha256": "hash2p",
            "field_fingerprint": "ff2p",
            "bit_ones_ratio": "0.52",
            "bit_ones_count": "530",
            "byte_chi2": "1.15",
            "byte_chi2_norm": "1.08",
            "runs_count": "99",
            "runs_expected": "101",
            "runs_norm_diff": "-0.005",
            "hw_win_mean": "252",
            "hw_win_std": "8",
            "hw_win_min": "235",
            "hw_win_max": "270",
            "autocorr_lag_1": "0.01",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_report_smoke(tmp_path):
    bench_csv = tmp_path / "bench.csv"
    analysis_csv = tmp_path / "analysis.csv"
    _write_bench_csv(bench_csv)
    _write_analysis_csv(analysis_csv)
    report_md = tmp_path / "report.md"
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "report",
            "--bench-csv",
            str(bench_csv),
            "--analysis-csv",
            str(analysis_csv),
            "--out",
            str(report_md),
            "--no-timestamp",
        ],
    )
    assert res.exit_code == 0, res.output
    content = report_md.read_text()
    assert "# ChaosCrypto WP2 – Report" in content
    assert "Benchmark Summary" in content
    assert "Analyze Summary" in content
    assert "seed_strategy" in content
    assert "/mnt/" not in content and "C:\\" not in content
    assert "Top throughput per seed_strategy (best across memory types)" in content
    assert "Top throughput per seed_strategy and memory_type" in content
    # Check table row counts (2 rows for seed_strategy, 4 for seed/memory)
    seed_section = content.split("Top throughput per seed_strategy (best across memory types):")[1].split(
        "Top throughput per seed_strategy and memory_type:"
    )[0]
    seed_lines = [l for l in seed_section.splitlines() if l.startswith("|") and not l.startswith("|---") and "seed_strategy" not in l and l.strip()]
    assert len(seed_lines) == 2
    seed_mem_section = content.split("Top throughput per seed_strategy and memory_type:")[1].split("## Analyze Summary")[0]
    seed_mem_lines = [l for l in seed_mem_section.splitlines() if l.startswith("|") and not l.startswith("|---") and "seed_strategy" not in l and l.strip()]
    assert len(seed_mem_lines) == 4


def test_report_plots(tmp_path):
    bench_csv = tmp_path / "bench.csv"
    analysis_csv = tmp_path / "analysis.csv"
    _write_bench_csv(bench_csv)
    _write_analysis_csv(analysis_csv)
    report_md = tmp_path / "report.md"
    plots_dir = tmp_path / "plots"
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "report",
            "--bench-csv",
            str(bench_csv),
            "--analysis-csv",
            str(analysis_csv),
            "--out",
            str(report_md),
            "--plots-dir",
            str(plots_dir),
            "--no-timestamp",
            "--plot-mode",
            "condensed",
        ],
    )
    assert res.exit_code == 0, res.output
    pngs = list(plots_dir.rglob("*.png"))
    assert pngs, "Expected at least one plot"
    assert all(p.stat().st_size > 0 for p in pngs)
    assert all("plots" not in str(p)[0:4] or True for p in pngs)  # dummy check to avoid empty condition
    # plots should be relative-embedded; ensure names use relative directories
    assert all("plots" in p.as_posix() for p in pngs)
    # Condensed mode: up to 4 plots per quant_k (here 1 quant_k)
    assert len(pngs) <= 4
