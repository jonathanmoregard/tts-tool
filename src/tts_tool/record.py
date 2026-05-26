"""Inline voice capture for the clone subcommand.

Prefers pw-record (PipeWire, default on modern NixOS desktop), falls back
to arecord (ALSA) or ffmpeg (PulseAudio). Output is always mono 44.1 kHz
signed-16 WAV, which Fish Audio accepts cleanly for voice cloning.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class RecordError(RuntimeError):
    pass


_RECORDERS = ("pw-record", "arecord", "ffmpeg")


def find_recorder() -> str:
    for cmd in _RECORDERS:
        if shutil.which(cmd):
            return cmd
    raise RecordError(
        "no audio capture tool on PATH. Install one of: "
        + ", ".join(_RECORDERS)
    )


def record_wav(
    out_path: Path,
    seconds: float,
    *,
    recorder: str | None = None,
) -> None:
    """Capture `seconds` of mono 44.1 kHz s16 WAV to `out_path`."""
    if seconds <= 0:
        raise RecordError("record duration must be positive")
    rec = recorder or find_recorder()
    if rec == "pw-record":
        # pw-record records until SIGINT; we stop it with a subprocess
        # timeout, which sends SIGKILL — pw-record finalises the WAV
        # header on each flush, so the truncated file is still valid.
        try:
            subprocess.run(
                ["pw-record", "--rate=44100", "--channels=1", "--format=s16",
                 str(out_path)],
                timeout=seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            pass
    elif rec == "arecord":
        subprocess.run(
            ["arecord", "-q", "-f", "S16_LE", "-r", "44100", "-c", "1",
             "-d", str(int(seconds)), str(out_path)],
            check=True,
        )
    elif rec == "ffmpeg":
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "pulse", "-i", "default",
             "-ac", "1", "-ar", "44100", "-t", str(seconds), str(out_path)],
            check=True,
        )
    else:
        raise RecordError(f"unknown recorder: {rec}")

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RecordError(f"recorder produced no output at {out_path}")
