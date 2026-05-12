"""Fish Audio TTS client wrapper. Per-chunk synth with retries."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from fishaudio import FishAudio, RateLimitError, ServerError


class MissingAPIKey(RuntimeError):
    pass


def read_api_key() -> str:
    key_file = os.environ.get("FISH_AUDIO_API_KEY_FILE", "").strip()
    if key_file:
        return Path(key_file).read_text().strip()
    key = os.environ.get("FISH_AUDIO_API_KEY", "").strip()
    if not key:
        raise MissingAPIKey(
            "set FISH_AUDIO_API_KEY or FISH_AUDIO_API_KEY_FILE "
            "(see .env.example or, on dellan, agenix secret fish-audio-api-key)"
        )
    return key


def make_client(api_key: str, *, timeout: float = 240.0) -> FishAudio:
    return FishAudio(api_key=api_key, timeout=timeout)


def synthesize_chunk(
    client: Any,
    text: str,
    *,
    model: str = "s2-pro",
    voice_id: str | None = None,
    speed: float = 1.0,
    max_retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """Convert one chunk of text to MP3 bytes. Retries transient errors."""
    backoffs = [2.0, 4.0, 8.0][:max_retries]
    last_err: Exception | None = None
    for delay in [0.0, *backoffs]:
        if delay > 0:
            sleep(delay)
        try:
            return client.tts.convert(
                text=text,
                reference_id=voice_id,
                format="mp3",
                speed=speed,
                model=model,
            )
        except (RateLimitError, ServerError) as e:
            last_err = e
            continue
    assert last_err is not None
    raise last_err
