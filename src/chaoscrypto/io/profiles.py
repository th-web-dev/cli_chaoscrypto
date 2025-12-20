from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict

from chaoscrypto.core.constants import MEMORY_TYPE
from chaoscrypto.core.memory.base import MemoryParams


def profiles_root(home: Path | None = None) -> Path:
    base = home or Path.home()
    return base / ".chaoscrypto" / "wp2"


def profile_dir(profile: str, home: Path | None = None) -> Path:
    return profiles_root(home) / profile


def profile_meta_path(profile: str, home: Path | None = None) -> Path:
    return profile_dir(profile, home) / "profile.json"


def profile_exists(profile: str, home: Path | None = None) -> bool:
    return profile_meta_path(profile, home).exists()


def save_profile_meta(profile: str, meta: Dict[str, Any], home: Path | None = None) -> Path:
    meta_path = profile_meta_path(profile, home)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def load_profile_meta(profile: str, home: Path | None = None) -> Dict[str, Any]:
    meta_path = profile_meta_path(profile, home)
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def token_fingerprint(token_bytes: bytes) -> str:
    return hashlib.sha256(token_bytes).hexdigest()


def memory_params_from_meta(meta: Dict[str, Any]) -> MemoryParams:
    memory_meta = meta.get("memory") or meta
    return MemoryParams(
        type=memory_meta.get("type", MEMORY_TYPE),
        size=int(memory_meta["size"]),
        scale=float(memory_meta["scale"]),
    )
