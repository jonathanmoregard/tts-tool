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


@pytest.mark.skipif(not HAS_FFMPEG, reason="requires ffmpeg")
def test_stitch_two_silences(tmp_path: Path):
    a, b = tmp_path / "a.mp3", tmp_path / "b.mp3"
    _silence_mp3(a, 0.2)
    _silence_mp3(b, 0.3)
    out = tmp_path / "out.mp3"

    stitch_mp3s([a, b], out)

    assert out.exists() and out.stat().st_size > 0

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
        capture_output=True, text=True, check=True,
    )
    duration = float(probe.stdout.strip())
    assert 0.4 < duration < 0.7


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
