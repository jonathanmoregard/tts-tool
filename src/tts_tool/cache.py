"""Content-hashed MP3 cache. Atomic writes, no eviction."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from platformdirs import user_cache_dir


def default_cache_dir() -> Path:
    return Path(user_cache_dir("tts-tool"))


def chunks_dir(override: str | os.PathLike[str] | None = None) -> Path:
    base = Path(override) if override else default_cache_dir()
    out = base / "chunks"
    out.mkdir(parents=True, exist_ok=True)
    return out


def key_for(text: str, model: str, voice_id: str | None, speed: float) -> str:
    payload = f"{text}|{model}|{voice_id or ''}|{speed:.4f}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def path_for(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.mp3"


def get(cache_dir: Path, key: str) -> bytes | None:
    p = path_for(cache_dir, key)
    return p.read_bytes() if p.exists() else None


def put(cache_dir: Path, key: str, data: bytes) -> None:
    p = path_for(cache_dir, key)
    tmp = p.with_suffix(".mp3.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)
