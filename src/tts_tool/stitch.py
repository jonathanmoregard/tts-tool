"""Concatenate MP3 chunks via ffmpeg concat demuxer (-c copy, no re-encode)."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class StitchError(RuntimeError):
    pass


def stitch_mp3s(parts: list[Path], output: Path) -> None:
    if not parts:
        raise StitchError("no inputs to stitch")
    if shutil.which("ffmpeg") is None:
        raise StitchError("ffmpeg not on PATH")

    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        list_path = Path(f.name)
        for p in parts:
            abs_p = str(p.resolve()).replace("'", r"'\''")
            f.write(f"file '{abs_p}'\n")

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(list_path),
                "-c", "copy",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise StitchError(f"ffmpeg failed: {result.stderr.strip() or 'unknown'}")
    finally:
        list_path.unlink(missing_ok=True)
