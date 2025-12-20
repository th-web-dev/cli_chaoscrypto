from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict


def encode_ciphertext(ciphertext: bytes) -> str:
    return base64.b64encode(ciphertext).decode("ascii")


def decode_ciphertext(encoded: str) -> bytes:
    return base64.b64decode(encoded.encode("ascii"))


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
