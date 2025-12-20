from __future__ import annotations

def normalize_record_memory_type(record: dict) -> dict:
    """Ensure a record has memory_type set; defaults to opensimplex."""
    if not record.get("memory_type"):
        record["memory_type"] = "opensimplex"
    return record
