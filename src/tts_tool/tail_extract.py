"""Extract the trailing N seconds of an MP3 byte blob via ffmpeg.

Used by the --prime-tail synth mode: we feed the tail of chunk N as a
ReferenceAudio to chunk N+1 so the Fish embedding sees the actual pitch
the prior chunk ended on, reducing inter-chunk pitch jumps.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


class TailExtractError(RuntimeError):
    pass


def tail_seconds(mp3: bytes, seconds: float) -> bytes:
    """Return the last `seconds` of the input MP3 as MP3 bytes.

    Uses ffmpeg with stream copy (`-c copy`) to avoid re-encoding. If the
    input is shorter than `seconds`, returns the whole input unchanged —
    pointless to slice and re-pay the encode cost.
    """
    if seconds <= 0:
        raise TailExtractError("tail seconds must be positive")
    if not mp3:
        raise TailExtractError("empty mp3 input")

    with tempfile.TemporaryDirectory(prefix="tts-tail-") as td:
        td_path = Path(td)
        src = td_path / "in.mp3"
        dst = td_path / "out.mp3"
        src.write_bytes(mp3)

        # Probe duration first; cheaper than running the slice and
        # discovering the file was 1s long.
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of",
                 "default=noprint_wrappers=1:nokey=1", str(src)],
                check=True, capture_output=True, text=True,
            )
            duration = float(probe.stdout.strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            raise TailExtractError(f"ffprobe failed: {e}") from e

        if duration <= seconds:
            return mp3

        try:
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-sseof", f"-{seconds}", "-i", str(src),
                 "-c", "copy", str(dst)],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise TailExtractError(f"ffmpeg tail extract failed: {e}") from e

        return dst.read_bytes()
