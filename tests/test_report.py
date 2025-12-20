import csv
from pathlib import Path

from typer.testing import CliRunner

from chaoscrypto.cli.app import app


def _write_bench_csv(path: Path):
    rows = [
        {
            "timestamp_utc": "",
            "profile": "alice",
            "coord_x": "0",
            "coord_y": "0",
            "nbytes": "1024",
            "repeats": "1",
            "repeat_index": "0",
            "dt": "0.01",
            "warmup": "100",
            "quant_k": "100000",
            "size": "128",
            "scale": "0.1",
            "t_field_s": "",
            "t_keystream_s": "0.5",
            "t_xor_s": "",
            "t_decrypt_s": "",
            "throughput_keystream_bps": str(1024 / 0.5),
            "throughput_xor_bps": "",
            "keystream_sha256": "hash1",
            "field_fingerprint": "ff1",
            "token_fingerprint": "tfp",
            "keystream_preview_base64": "",
        },
        {
            "timestamp_utc": "",
            "profile": "alice",
            "coord_x": "0",
            "coord_y": "0",
            "nbytes": "1024",
            "repeats": "1",
            "repeat_index": "0",
            "dt": "0.01",
            "warmup": "200",
            "quant_k": "100000",
            "size": "128",
            "scale": "0.1",
            "t_field_s": "",
            "t_keystream_s": "1.0",
            "t_xor_s": "",
            "t_decrypt_s": "",
            "throughput_keystream_bps": str(1024 / 1.0),
            "throughput_xor_bps": "",
            "keystream_sha256": "hash2",
            "field_fingerprint": "ff2",
            "token_fingerprint": "tfp",
            "keystream_preview_base64": "",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _write_analysis_csv(path: Path):
    rows = [
        {
            "timestamp_utc": "",
            "profile": "alice",
            "coord_x": "0",
            "coord_y": "0",
            "nbytes": "1024",
            "dt": "0.01",
            "warmup": "100",
            "quant_k": "100000",
            "size": "128",
            "scale": "0.1",
            "keystream_sha256": "hash1",
            "field_fingerprint": "ff1",
            "token_fingerprint": "tfp",
            "bit_ones_ratio": "0.51",
            "bit_ones_count": "523",
            "byte_chi2": "1.0",
            "byte_chi2_norm": "1.0",
            "byte_entropy_approx": "7.9",
            "byte_top5": "00:5|ff:4",
            "runs_count": "100",
            "runs_expected": "101",
            "runs_norm_diff": "0.01",
            "hw_win_mean": "256",
            "hw_win_std": "10",
            "hw_win_min": "240",
            "hw_win_max": "270",
            "keystream_preview_base64": "",
            "autocorr_lag_1": "0.02",
        },
        {
            "timestamp_utc": "",
            "profile": "alice",
            "coord_x": "0",
            "coord_y": "0",
            "nbytes": "1024",
            "dt": "0.01",
            "warmup": "200",
            "quant_k": "100000",
            "size": "128",
            "scale": "0.1",
            "keystream_sha256": "hash2",
            "field_fingerprint": "ff2",
            "token_fingerprint": "tfp",
            "bit_ones_ratio": "0.49",
            "bit_ones_count": "500",
            "byte_chi2": "1.2",
            "byte_chi2_norm": "1.1",
            "byte_entropy_approx": "7.8",
            "byte_top5": "00:6|ff:3",
            "runs_count": "98",
            "runs_expected": "101",
            "runs_norm_diff": "-0.01",
            "hw_win_mean": "250",
            "hw_win_std": "9",
            "hw_win_min": "230",
            "hw_win_max": "270",
            "keystream_preview_base64": "",
            "autocorr_lag_1": "-0.03",
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
        ],
    )
    assert res.exit_code == 0, res.output
    pngs = list(plots_dir.rglob("*.png"))
    assert pngs, "Expected at least one plot"
    assert all(p.stat().st_size > 0 for p in pngs)
