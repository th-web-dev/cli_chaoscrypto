from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Sequence

USABILITY_LOG_FIELDS = [
    "timestamp_utc",
    "session_id",
    "command_name",
    "command_argv_json",
    "command_arg_count",
    "command_flag_count",
    "duration_s",
    "status",
    "exit_code",
    "error_message",
    "artifact_paths_json",
    "artifact_hashes_json",
    "repro_key",
    "repro_match_previous",
    "failed_attempts_before_success",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_artifact_hashes(paths: Sequence[Path]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for path in paths:
        if path.exists() and path.is_file():
            hashes[str(path)] = _sha256_file(path)
    return hashes


def read_usability_log(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_usability_log(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=USABILITY_LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    text = str(val).strip()
    if text == "":
        return None
    return float(text)


def _percentile(values: List[float], q: float) -> float | None:
    if not values:
        return None
    arr = sorted(values)
    if len(arr) == 1:
        return arr[0]
    idx = (len(arr) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(arr) - 1)
    frac = idx - lo
    return arr[lo] * (1.0 - frac) + arr[hi] * frac


def build_usability_row(
    *,
    session_id: str,
    command_argv: Sequence[str],
    duration_s: float,
    status: str,
    exit_code: int,
    artifact_paths: Sequence[Path],
    repro_key: str | None,
    error_message: str | None,
    existing_rows: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    command_name = command_argv[0] if command_argv else ""
    command_signature = json.dumps(list(command_argv), ensure_ascii=True)
    artifact_hashes = compute_artifact_hashes(artifact_paths)
    failed_before_success = 0
    repro_match_previous: bool | None = None
    previous_artifact_hashes: Dict[str, str] | None = None

    for old in reversed(existing_rows):
        if old.get("command_argv_json") != command_signature:
            continue
        if repro_key and old.get("repro_key") != repro_key:
            continue
        if old.get("status") == "success":
            try:
                previous_artifact_hashes = json.loads(old.get("artifact_hashes_json") or "{}")
            except Exception:  # noqa: BLE001
                previous_artifact_hashes = None
            break
        failed_before_success += 1

    if status == "success" and previous_artifact_hashes is not None and artifact_hashes:
        repro_match_previous = artifact_hashes == previous_artifact_hashes

    return {
        "timestamp_utc": now_utc_iso(),
        "session_id": session_id,
        "command_name": command_name,
        "command_argv_json": command_signature,
        "command_arg_count": len(command_argv),
        "command_flag_count": sum(1 for arg in command_argv if str(arg).startswith("-")),
        "duration_s": duration_s,
        "status": status,
        "exit_code": exit_code,
        "error_message": error_message,
        "artifact_paths_json": json.dumps([str(p) for p in artifact_paths], ensure_ascii=True),
        "artifact_hashes_json": json.dumps(artifact_hashes, sort_keys=True),
        "repro_key": repro_key,
        "repro_match_previous": repro_match_previous,
        "failed_attempts_before_success": failed_before_success if status == "success" else None,
    }


def run_usability_eval(log_csv: Path) -> Dict[str, Any]:
    rows = read_usability_log(log_csv)
    total = len(rows)
    success_rows = [r for r in rows if str(r.get("status")) == "success"]
    fail_rows = [r for r in rows if str(r.get("status")) == "fail"]
    durations = [_to_float(r.get("duration_s")) for r in rows]
    durations = [x for x in durations if x is not None]
    repro_vals = [str(r.get("repro_match_previous")) for r in success_rows if str(r.get("repro_match_previous")) not in {"", "None", "none"}]
    repro_true = sum(1 for v in repro_vals if v.lower() == "true")
    repro_rate = (repro_true / len(repro_vals)) if repro_vals else None
    failed_before = [_to_float(r.get("failed_attempts_before_success")) for r in success_rows]
    failed_before = [x for x in failed_before if x is not None]
    by_command: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        cmd = str(row.get("command_name") or "")
        by_command.setdefault(cmd, {"durations": [], "total": 0, "success": 0, "fail": 0})
        by_command[cmd]["total"] += 1
        if str(row.get("status")) == "success":
            by_command[cmd]["success"] += 1
        else:
            by_command[cmd]["fail"] += 1
        d = _to_float(row.get("duration_s"))
        if d is not None:
            by_command[cmd]["durations"].append(d)
    command_rows: List[Dict[str, Any]] = []
    for cmd, data in sorted(by_command.items()):
        dvals = data["durations"]
        command_rows.append(
            {
                "command_name": cmd,
                "runs": data["total"],
                "success_rate": (data["success"] / data["total"]) if data["total"] else None,
                "duration_median_s": median(dvals) if dvals else None,
                "duration_p95_s": _percentile(dvals, 0.95) if dvals else None,
                "mean_failed_attempts_before_success": None if data["success"] == 0 else mean(
                    [
                        _to_float(r.get("failed_attempts_before_success")) or 0.0
                        for r in rows
                        if str(r.get("command_name") or "") == cmd and str(r.get("status")) == "success"
                    ]
                ),
            }
        )
    summary = {
        "runs_total": total,
        "runs_success": len(success_rows),
        "runs_fail": len(fail_rows),
        "success_rate": (len(success_rows) / total) if total else None,
        "duration_median_s": median(durations) if durations else None,
        "duration_p95_s": _percentile(durations, 0.95) if durations else None,
        "repro_checks": len(repro_vals),
        "repro_match_rate": repro_rate,
        "mean_failed_attempts_before_success": mean(failed_before) if failed_before else None,
    }
    return {"summary": summary, "by_command": command_rows}


def write_usability_summary_csv(path: Path, summary: Dict[str, Any], by_command: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "metric", "value"])
        writer.writeheader()
        for k, v in summary.items():
            writer.writerow({"scope": "overall", "metric": k, "value": v})
        for row in by_command:
            cmd = row.get("command_name")
            for metric in ("runs", "success_rate", "duration_median_s", "duration_p95_s", "mean_failed_attempts_before_success"):
                writer.writerow({"scope": f"command:{cmd}", "metric": metric, "value": row.get(metric)})


def render_usability_markdown(summary: Dict[str, Any], by_command: List[Dict[str, Any]]) -> str:
    lines = ["# Usability Evaluation Summary", "", "## Overall"]
    lines.append(f"- Runs total: {summary.get('runs_total')}")
    lines.append(f"- Success rate: {summary.get('success_rate')}")
    lines.append(f"- Duration median/p95 (s): {summary.get('duration_median_s')} / {summary.get('duration_p95_s')}")
    lines.append(f"- Reproducibility match rate: {summary.get('repro_match_rate')} ({summary.get('repro_checks')} checks)")
    lines.append(f"- Mean failed attempts before success: {summary.get('mean_failed_attempts_before_success')}")
    lines.append("")
    lines.append("## By Command")
    for row in by_command:
        lines.append(
            f"- {row.get('command_name')}: runs={row.get('runs')} success_rate={row.get('success_rate')} "
            f"median_s={row.get('duration_median_s')} p95_s={row.get('duration_p95_s')}"
        )
    return "\n".join(lines) + "\n"


def write_usability_markdown(path: Path, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def write_usability_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
