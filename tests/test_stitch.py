import shutil
import subprocess
from pathlib import Path

import pytest

from tts_tool.stitch import StitchError, stitch_mp3s

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _silence_mp3(path: Path, seconds: float = 0.2) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
            "-t", str(seconds), "-b:a", "128k",
            str(path),
        ],
        check=True,
    )


def _duration(path: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(probe.stdout.strip())


@pytest.mark.skipif(not HAS_FFMPEG, reason="requires ffmpeg")
def test_stitch_two_chunks_bare_paths(tmp_path: Path):
    """Back-compat: list[Path] still works (silence_after defaults to 0)."""
    a, b = tmp_path / "a.mp3", tmp_path / "b.mp3"
    _silence_mp3(a, 0.2)
    _silence_mp3(b, 0.3)
    out = tmp_path / "out.mp3"
    stitch_mp3s([a, b], out)
    assert out.exists() and out.stat().st_size > 0
    assert 0.4 < _duration(out) < 0.7


def test_stitch_empty_raises():
    with pytest.raises(StitchError, match="no inputs"):
        stitch_mp3s([], Path("/tmp/nope.mp3"))


@pytest.mark.skipif(not HAS_FFMPEG, reason="requires ffmpeg")
def test_stitch_single_input_works(tmp_path: Path):
    a = tmp_path / "a.mp3"
    _silence_mp3(a, 0.1)
    out = tmp_path / "out.mp3"
    stitch_mp3s([a], out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.skipif(not HAS_FFMPEG, reason="requires ffmpeg")
def test_stitch_per_chunk_silence_after(tmp_path: Path):
    """Each tuple specifies its own silence_after duration. Total =
    0.2 + 1.0 (after a) + 0.2 = 1.4s. Silence after the LAST chunk is
    suppressed even if specified."""
    a, b = tmp_path / "a.mp3", tmp_path / "b.mp3"
    _silence_mp3(a, 0.2)
    _silence_mp3(b, 0.2)
    out = tmp_path / "out.mp3"
    stitch_mp3s([(a, 1.0), (b, 5.0)], out)
    duration = _duration(out)
    assert 1.3 < duration < 1.6, f"expected ~1.4s, got {duration}s"


@pytest.mark.skipif(not HAS_FFMPEG, reason="requires ffmpeg")
def test_stitch_per_chunk_varied_silences(tmp_path: Path):
    """Three chunks with varied silence_after: 0.2 + 0.5 + 0.2 + 1.0 +
    0.2 = 2.1s (last silence suppressed)."""
    a, b, c = tmp_path / "a.mp3", tmp_path / "b.mp3", tmp_path / "c.mp3"
    _silence_mp3(a, 0.2)
    _silence_mp3(b, 0.2)
    _silence_mp3(c, 0.2)
    out = tmp_path / "out.mp3"
    stitch_mp3s([(a, 0.5), (b, 1.0), (c, 9.9)], out)
    duration = _duration(out)
    # ~2.1s expected; allow up to 2.5 for mp3-frame rounding overhead
    assert 2.0 < duration < 2.5, f"expected ~2.1s, got {duration}s"


@pytest.mark.skipif(not HAS_FFMPEG, reason="requires ffmpeg")
def test_stitch_per_chunk_zero_silence_no_op(tmp_path: Path):
    a, b = tmp_path / "a.mp3", tmp_path / "b.mp3"
    _silence_mp3(a, 0.2)
    _silence_mp3(b, 0.2)
    out = tmp_path / "out.mp3"
    stitch_mp3s([(a, 0.0), (b, 0.0)], out)
    duration = _duration(out)
    assert 0.3 < duration < 0.6, f"expected ~0.4s, got {duration}s"
