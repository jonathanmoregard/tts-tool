"""Optional voice sample cleanup via Auphonic.

Wraps the `auphonic` CLI (https://auphonic.com/cli). When the
`--enhance` path is taken, per-segment recorded WAVs are sent to
Auphonic's `process` endpoint with breath/mouth-noise removal +
denoise enabled, then the cleaned WAV is downloaded and played back
to the user for accept/redo. Adobe Podcast Enhance is general
denoise; Auphonic has a dedicated `--debreath-amount` flag and a
"cough cutter" — the right tool for the breath-onset issue at chunk
boundaries in long-form synth from cloned voices.

Requires:
  - `auphonic` binary on PATH (https://auphonic.com/cli)
  - `AUPHONIC_API_KEY` in env (the binary also reads ~/.auphonic
    config but env wins per `auphonic --help`)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class EnhanceError(RuntimeError):
    pass


# Conservative defaults — strong enough to remove sharp inhales without
# over-processing the voice (which destroys the timbre Fish needs).
DEFAULT_DEBREATH_DB = 12
DEFAULT_DENOISE = True
DEFAULT_TIMEOUT_S = 300.0


def auphonic_available() -> bool:
    """True if `auphonic` is on PATH and AUPHONIC_API_KEY is set."""
    if shutil.which("auphonic") is None:
        return False
    return bool(os.environ.get("AUPHONIC_API_KEY", "").strip())


def enhance_with_auphonic(
    in_path: Path,
    out_path: Path,
    *,
    debreath_db: int = DEFAULT_DEBREATH_DB,
    denoise: bool = DEFAULT_DENOISE,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> None:
    """Upload `in_path` to Auphonic, download the cleaned WAV to `out_path`.

    `debreath_db` controls breath reduction in dB (Auphonic accepts positive
    values; default 12 dB is a moderate reduction that survives loud breaths
    without stripping the speaker's normal exhale character).

    Raises EnhanceError if the binary is missing, the API key isn't set, the
    CLI exits non-zero, or the output file fails to materialise.
    """
    if shutil.which("auphonic") is None:
        raise EnhanceError(
            "auphonic CLI not on PATH (https://auphonic.com/cli)"
        )
    if not os.environ.get("AUPHONIC_API_KEY", "").strip():
        raise EnhanceError(
            "AUPHONIC_API_KEY not set (or empty); auphonic --help for auth)"
        )
    if not in_path.is_file():
        raise EnhanceError(f"input not found: {in_path}")

    out_name = out_path.name
    with tempfile.TemporaryDirectory(prefix="tts-enhance-") as td:
        td_path = Path(td)
        cmd = [
            "auphonic", "process", str(in_path),
            "--debreath-amount", str(debreath_db),
            "--denoise" if denoise else "--denoise=false",
            "--output", f"format=wav,filename={out_name}",
            "--wait", "--download",
            "--download-dir", str(td_path),
        ]
        try:
            subprocess.run(
                cmd, check=True, timeout=timeout,
                capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip()
            raise EnhanceError(
                f"auphonic process exited {e.returncode}: {stderr or '(no stderr)'}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise EnhanceError(
                f"auphonic process timed out after {timeout}s"
            ) from e

        produced = td_path / out_name
        if not produced.exists() or produced.stat().st_size == 0:
            raise EnhanceError(
                f"auphonic produced no output at {produced}"
            )
        out_path.write_bytes(produced.read_bytes())
