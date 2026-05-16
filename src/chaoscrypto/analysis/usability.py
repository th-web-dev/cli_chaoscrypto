from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
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
