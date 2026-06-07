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
    """One scripted register slot in the recording flow.

    `transcript` is a template that may contain the literal `{name}`
    placeholder; render via `render_transcript`. Other text is read
    verbatim — the speaker doesn't need to identify with the prose;
    Fish only uses the transcript to improve fidelity, not as belief.
    """
    label: str            # short tag shown in prompts, e.g. "warmup"
    instructions: str     # how to read it (volume, pace, register)
    transcript: str       # template; may contain "{name}"
    max_seconds: float    # cap on recording length


def render_transcript(template: str, name: str = "") -> str:
    """Substitute {name} or strip the name clause if name is empty."""
    if name:
        return template.replace("{name}", name)
    # No name supplied: drop the introduction clause cleanly.
    cleaned = template.replace("My name is {name}, and ", "")
    cleaned = cleaned.replace(", {name},", "")
    cleaned = cleaned.replace("{name}", "the speaker")
    return cleaned


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
            "Hello. My name is {name}, and I'm recording a short sample "
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
            "A few anchors for the model. The year is two thousand "
            "twenty-six. The weather today is around fifteen degrees. "
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


def prompt_verdict(
    prompt_input=None,
    *,
    is_enhanced: bool = False,
    enhance_available: bool = False,
) -> str:
    """Ask user to (p)lay / (y) accept / (r) redo / (s) skip / (q) quit.

    Default is (r)edo — empty input re-records, since the speaker typically
    knows mid-recording whether they slipped up and shouldn't have to
    type anything to retry. (p) plays the take back through mpv.

    When `enhance_available` is True (auphonic CLI + key reachable), the
    prompt shows EITHER `enhance (e)` (when currently raw) OR
    `unenhance (u)` (when currently enhanced). The toggle is free after
    the first (e) — the enhanced file is cached on disk and (u) just
    flips which version is the "active" one for (p)/(y).
    """
    if prompt_input is None:
        prompt_input = input
    while True:
        toggle_segment = ""
        if enhance_available:
            toggle_segment = (
                " / unenhance (u)" if is_enhanced else " / enhance (e)"
            )
        raw = prompt_input(
            f"play (p) / accept (y) / redo (r){toggle_segment} / "
            f"skip (s) / quit (q)? [r]: "
        )
        choice = (raw or "r").strip().lower()[:1]
        valid = {"p", "y", "r", "s", "q"}
        if enhance_available:
            valid.add("u" if is_enhanced else "e")
        if choice in valid:
            return choice
        print("invalid choice, try again", file=sys.stderr)


def run_interactive_capture(
    segments: tuple[Segment, ...],
    work_dir: Path,
    *,
    recorder: str | None = None,
    speaker_name: str = "",
    prompt_input=None,
    record_fn=None,
    play_fn=None,
    enhance_fn=None,
) -> list[tuple[Path, str]]:
    """Walk segments, recording each with user accept/redo prompt.

    Enhancement is per-take, opt-in via the (e) verdict choice. When
    `enhance_fn` is provided AND the user hits (e), enhance_fn is called
    ONCE on the raw recording — its output written to a sibling
    `<basename>.enhanced.wav`. (p) / (y) then operate on the enhanced
    version. (u) flips back to raw; (e) again flips forward — both
    toggles are free (no re-processing). enhance_fn(in_path, out_path)
    overwrites out_path with the cleaned bytes.

    `enhance_fn=None` removes (e) from the per-take menu entirely (the
    CLI sets this when auphonic isn't reachable).

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
        text = render_transcript(seg.transcript, speaker_name)
        print(f"\n=== [{idx}/{len(segments)}] {seg.label} ===", file=sys.stderr)
        print(f"register: {seg.instructions}", file=sys.stderr)
        print(f"text:\n  {text}", file=sys.stderr)

        while True:  # per take loop
            prompt_input(
                f"press ENTER to start recording (max {seg.max_seconds:.0f}s, "
                "ENTER again to stop): "
            )
            raw_wav = work_dir / f"{idx:02d}-{seg.label}.wav"
            enhanced_wav = work_dir / f"{idx:02d}-{seg.label}.enhanced.wav"
            record_fn(raw_wav, seg.max_seconds, recorder=recorder)

            # Per-take enhance state. Lazily materialized when user hits
            # (e) the FIRST time. After that, (e)/(u) toggles the active
            # flag for free — the enhanced file stays on disk and the
            # toggle just decides which path gets passed to play / accept.
            is_enhanced = False
            enhanced_built = False

            def active_path() -> Path:
                return enhanced_wav if is_enhanced else raw_wav

            # Verdict loop: 'p' replays without re-recording, 'e'/'u'
            # toggle enhanced version, all other choices break out
            # (either to next segment or back to re-record).
            advance = False
            while not advance:
                v = prompt_verdict(
                    prompt_input=prompt_input,
                    is_enhanced=is_enhanced,
                    enhance_available=(enhance_fn is not None),
                )
                if v == "p":
                    play_fn(active_path())
                    continue
                if v == "e":
                    if not enhanced_built:
                        try:
                            enhance_fn(raw_wav, enhanced_wav)
                            enhanced_built = True
                            print(
                                f"enhanced: {enhanced_wav.name}",
                                file=sys.stderr,
                            )
                        except Exception as exc:
                            print(
                                f"warning: enhance failed ({exc}); "
                                "staying with raw take",
                                file=sys.stderr,
                            )
                            continue
                    is_enhanced = True
                    continue
                if v == "u":
                    is_enhanced = False
                    print(
                        f"reverted: {raw_wav.name} (raw)",
                        file=sys.stderr,
                    )
                    continue
                if v == "y":
                    accepted.append((active_path(), text))
                    advance = True
                elif v == "r":
                    advance = False
                    break  # leave verdict loop, re-enter take loop
                elif v == "s":
                    print(f"skipped {seg.label}", file=sys.stderr)
                    advance = True
                elif v == "q":
                    raise KeyboardInterrupt("user quit recording flow")
            if advance:
                break  # leave take loop, advance to next segment

    return accepted
