"""Concatenate MP3 chunks via ffmpeg concat demuxer (-c copy, no re-encode).

Optional inter-chunk silence: tts-tool chunks at paragraph boundaries
(`chunk.py` greedy-packs sentences but prefers paragraph soft-close).
Fish s2-pro empirically drops pause tags at chunk edges (probed
2026-05-22: `[very long pause]` at end-of-text adds 0ms; mid-text
adds ~1s). So inter-chunk paragraph pauses are lost. Inserting a
silent MP3 BETWEEN every chunk at stitch time is a deterministic,
ffmpeg-level paragraph beat that's immune to Fish's tag-edge quirks.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class StitchError(RuntimeError):
    pass


_SILENCE_BITRATE = "128k"
_SILENCE_SAMPLE_RATE = "22050"


def _generate_silence(tmpdir: Path, seconds: float) -> Path:
    """Render `seconds` of silent MP3 at the same bitrate / sample rate
    Fish s2-pro emits, so the concat demuxer doesn't have to re-encode."""
    out = tmpdir / f"silence-{seconds:.3f}s.mp3"
    if out.exists():
        return out
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"anullsrc=r={_SILENCE_SAMPLE_RATE}:cl=mono",
            "-t", f"{seconds}",
            "-b:a", _SILENCE_BITRATE,
            str(out),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise StitchError(
            f"ffmpeg silence generation failed: {result.stderr.strip()}"
        )
    return out


def stitch_mp3s(
    parts: list[Path],
    output: Path,
    *,
    inter_chunk_silence_seconds: float = 0.0,
) -> None:
    """Concatenate `parts` into `output`. If `inter_chunk_silence_seconds`
    > 0, a silent MP3 of that duration is interleaved between every
    pair of chunks (no leading/trailing silence)."""
    if not parts:
        raise StitchError("no inputs to stitch")
    if shutil.which("ffmpeg") is None:
        raise StitchError("ffmpeg not on PATH")

    with tempfile.TemporaryDirectory(prefix="tts-tool-stitch-") as td:
        td_path = Path(td)
        silence_path: Path | None = None
        if inter_chunk_silence_seconds > 0 and len(parts) > 1:
            silence_path = _generate_silence(td_path, inter_chunk_silence_seconds)

        list_path = td_path / "concat.txt"
        with list_path.open("w", encoding="utf-8") as f:
            for i, p in enumerate(parts):
                if i > 0 and silence_path is not None:
                    abs_sil = str(silence_path.resolve()).replace("'", r"'\''")
                    f.write(f"file '{abs_sil}'\n")
                abs_p = str(p.resolve()).replace("'", r"'\''")
                f.write(f"file '{abs_p}'\n")

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
