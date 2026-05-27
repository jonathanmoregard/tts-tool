"""Tests for ffmpeg-backed tail extraction used by --prime-tail mode."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tts_tool.tail_extract import TailExtractError, tail_seconds

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _make_silent_mp3(path: Path, seconds: float) -> bytes:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
         "-t", str(seconds), "-b:a", "64k", str(path)],
        check=True,
    )
    return path.read_bytes()


def _mp3_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(out)


def test_tail_seconds_rejects_zero():
    with pytest.raises(TailExtractError, match="positive"):
        tail_seconds(b"fake", 0)


def test_tail_seconds_rejects_empty():
    with pytest.raises(TailExtractError, match="empty"):
        tail_seconds(b"", 2.0)


@pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg + ffprobe")
def test_tail_seconds_short_input_returned_as_is(tmp_path: Path):
    """1s input, 3s tail requested — return whole thing."""
    src = tmp_path / "short.mp3"
    src_bytes = _make_silent_mp3(src, 1.0)
    out = tail_seconds(src_bytes, 3.0)
    assert out == src_bytes


@pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg + ffprobe")
def test_tail_seconds_long_input_sliced(tmp_path: Path):
    """6s input, 2s tail requested — output should be ~2s long, not the whole 6s."""
    src = tmp_path / "long.mp3"
    src_bytes = _make_silent_mp3(src, 6.0)
    out = tail_seconds(src_bytes, 2.0)

    assert len(out) < len(src_bytes)
    out_path = tmp_path / "out.mp3"
    out_path.write_bytes(out)
    dur = _mp3_duration(out_path)
    assert 1.5 <= dur <= 2.5, f"tail duration {dur}s outside expected window"
