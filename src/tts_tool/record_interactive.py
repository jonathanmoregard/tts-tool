"""Interactive multi-segment voice capture for cloning.

Drives the user through a scripted set of register-varied segments,
recording each, playing it back, and prompting for accept / redo. The
accepted segments are bundled into a single `voices.create` upload so
the embedding sees varied timbre + phonetic coverage in one model.

Why this exists: Fish (and every neural cloner) benefits from samples
that span the register range you want the clone to reproduce. Reading
4 minutes straight is unforgiving — one slurred consonant taints the
whole take. Segmenting + per-segment retry lets a speaker with a slight
impediment iterate without redoing the entire script.
"""
from __future__ import annotations

import select
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import record
from .record import RecordError


@dataclass(frozen=True)
class Segment:
    """One scripted register slot in the recording flow."""
    label: str            # short tag shown in prompts, e.g. "warmup"
    instructions: str     # how to read it (volume, pace, register)
    transcript: str       # the text to read aloud
    max_seconds: float    # cap on recording length


# Default 6-segment script. Per research:
#   - segment 1: phonetically balanced warm-up (timbre + phoneme baseline)
#   - segments 2-4: register variety (tender / playful / reflective)
#   - segment 5: number + list stability anchor
#   - segment 6: intimate close (the "for someone you love" register)
# Each segment caps at ~45s so total stays in the 3-4 min sweet spot
# Fish recommends for a stable, full-bodied embedding.
DEFAULT_SCRIPT: tuple[Segment, ...] = (
    Segment(
        label="warmup",
        instructions="Neutral, conversational. Establishes baseline timbre.",
        transcript=(
            "Hello. My name is Jonathan, and I'm recording a short sample "
            "so a model can learn the sound of my voice. The rainbow is "
            "a division of white light into many beautiful colors. These "
            "take the shape of a long round arch, with its path high "
            "above, and its two ends apparently beyond the horizon."
        ),
        max_seconds=45.0,
    ),
    Segment(
        label="tender",
        instructions="Soft, slow, lower volume. Space between words.",
        transcript=(
            "When I think about the people I love, I notice how my voice "
            "changes. It gets lower. It slows down. I leave more room "
            "between words, because I want each one to actually land."
        ),
        max_seconds=40.0,
    ),
    Segment(
        label="playful",
        instructions="Lighter, slightly faster, a small grin in the voice.",
        transcript=(
            "I'm not very good at being formal. Most of the time I'd "
            "rather make someone laugh than impress them — a bad pun, "
            "a story that goes nowhere, the kind of joke where the "
            "build-up is the point."
        ),
        max_seconds=35.0,
    ),
    Segment(
        label="reflective",
        instructions="Quiet, almost breathy, very slow. Like thinking out loud.",
        transcript=(
            "Sometimes, walking by water in the evening, the light goes "
            "pink and gold at the same time, and the city quiets down. "
            "I notice my breathing slow. I notice my shoulders drop."
        ),
        max_seconds=40.0,
    ),
    Segment(
        label="stability",
        instructions="Neutral register. Clear articulation. Numbers + lists.",
        transcript=(
            "A few anchors for the model. In two thousand twenty-six we "
            "live in Sweden. The weather today is around fifteen degrees. "
            "Three apples, seven minutes, eleven o'clock, twenty-four "
            "hours. Bread, coffee, a long evening, a quiet morning."
        ),
        max_seconds=40.0,
    ),
    Segment(
        label="intimate",
        instructions="Warm, close to the mic, almost whispered at the end.",
        transcript=(
            "If you hear this voice saying something kind to you, then "
            "you already know — it means it. Take your time. Sleep "
            "well. I love you."
        ),
        max_seconds=30.0,
    ),
)


def _wait_enter_or_timeout(timeout: float) -> bool:
    """Block on stdin until ENTER or timeout. Return True if ENTER pressed.

    Uses select() on stdin so we can also be interrupted by the recorder
    subprocess exiting (its fd isn't watched here, but the 0.1s poll loop
    is fine-grained enough that the caller can check process state too).
    """
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        rlist, _, _ = select.select([sys.stdin], [], [], min(0.1, remaining))
        if rlist:
            sys.stdin.readline()
            return True


def record_until_enter(
    out_path: Path,
    max_seconds: float,
    *,
    recorder: str | None = None,
) -> None:
    """Record audio until the user presses ENTER, or max_seconds elapses.

    Uses pw-record as a subprocess (the only one of the recorder backends
    that supports SIGTERM-on-demand cleanly — arecord/ffmpeg with -d would
    need a separate code path; pw-record is the NixOS desktop default
    anyway). Falls back to fixed-duration record_wav if the backend is
    ALSA/ffmpeg, because those tools don't speak interactive stop.
    """
    rec = recorder or record.find_recorder()
    if rec != "pw-record":
        # arecord / ffmpeg with -d: caller asked for stop-on-ENTER but
        # we can't deliver. Run the fixed-duration recorder and emit a
        # warning so the caller knows we silently degraded UX.
        print(
            f"warning: {rec} doesn't support stop-on-ENTER; "
            f"recording fixed {max_seconds:.0f}s.",
            file=sys.stderr,
        )
        record.record_wav(out_path, max_seconds, recorder=rec)
        return

    proc = subprocess.Popen(
        ["pw-record", "--rate=44100", "--channels=1", "--format=s16",
         str(out_path)],
    )
    try:
        _wait_enter_or_timeout(max_seconds)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RecordError(f"pw-record produced no output at {out_path}")


def play_back(path: Path) -> None:
    """Play a WAV/MP3 file via mpv if available; else no-op with notice."""
    if shutil.which("mpv") is None:
        print("(no mpv on PATH; skipping playback)", file=sys.stderr)
        return
    subprocess.run(
        ["mpv", "--no-video", "--really-quiet", str(path)],
        check=False,
    )


def prompt_verdict(prompt_input=None) -> str:
    """Ask user to accept (y), redo (r), skip (s), or quit (q). Returns letter."""
    if prompt_input is None:
        prompt_input = input
    while True:
        raw = prompt_input("accept (y) / redo (r) / skip (s) / quit (q)? [y]: ")
        choice = (raw or "y").strip().lower()[:1]
        if choice in {"y", "r", "s", "q"}:
            return choice
        print("invalid choice, try again", file=sys.stderr)


def run_interactive_capture(
    segments: tuple[Segment, ...],
    work_dir: Path,
    *,
    recorder: str | None = None,
    prompt_input=None,
    record_fn=None,
    play_fn=None,
) -> list[tuple[Path, str]]:
    """Walk segments, recording each with user accept/redo prompt.

    Returns a list of (wav_path, transcript) for every accepted segment,
    in script order, skipping any the user dropped with (s). Exits early
    on (q) and raises KeyboardInterrupt to bubble up to the CLI.
    """
    if prompt_input is None:
        prompt_input = input
    if record_fn is None:
        record_fn = record_until_enter
    if play_fn is None:
        play_fn = play_back
    accepted: list[tuple[Path, str]] = []
    work_dir.mkdir(parents=True, exist_ok=True)

    for idx, seg in enumerate(segments, start=1):
        print(f"\n=== [{idx}/{len(segments)}] {seg.label} ===", file=sys.stderr)
        print(f"register: {seg.instructions}", file=sys.stderr)
        print(f"text:\n  {seg.transcript}", file=sys.stderr)

        while True:
            prompt_input(
                f"press ENTER to start recording (max {seg.max_seconds:.0f}s, "
                "ENTER again to stop): "
            )
            wav = work_dir / f"{idx:02d}-{seg.label}.wav"
            record_fn(wav, seg.max_seconds, recorder=recorder)
            print("playback:", file=sys.stderr)
            play_fn(wav)

            v = prompt_verdict(prompt_input=prompt_input)
            if v == "y":
                accepted.append((wav, seg.transcript))
                break
            if v == "r":
                continue
            if v == "s":
                print(f"skipped {seg.label}", file=sys.stderr)
                break
            if v == "q":
                raise KeyboardInterrupt("user quit recording flow")

    return accepted
