"""Fish Audio voice cloning. Wraps client.voices.create()."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class CloneError(RuntimeError):
    pass


def clone_voice(
    client: Any,
    *,
    title: str,
    sample_paths: list[Path],
    description: str | None = None,
    texts: list[str] | None = None,
    visibility: str = "private",
) -> str:
    """Upload sample audio to Fish Audio, return the new voice reference_id.

    `texts`, when given, must align 1-to-1 with `sample_paths`: each entry
    is the transcript of the corresponding sample. Fish uses transcripts to
    improve cloning fidelity. Omit entirely to let Fish auto-transcribe.
    """
    if not sample_paths:
        raise CloneError("at least one sample is required")
    if texts is not None and len(texts) != len(sample_paths):
        raise CloneError(
            f"text count ({len(texts)}) must match sample count "
            f"({len(sample_paths)}), or omit text entirely"
        )
    audio_bytes: list[bytes] = []
    for p in sample_paths:
        if not p.is_file():
            raise CloneError(f"sample not found: {p}")
        audio_bytes.append(p.read_bytes())
    voice = client.voices.create(
        title=title,
        voices=audio_bytes,
        description=description,
        texts=texts,
        visibility=visibility,
    )
    return voice.id
