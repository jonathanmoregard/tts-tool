"""tts-tool entrypoint. CleanText -> MP3."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from . import chunk as chunkmod
from . import cache, clone, record, record_interactive, synthesize
from .stitch import StitchError, stitch_mp3s

# Fish Audio voice library reference_id. Cloned from a 90s sample of
# Jonathan Moregard reading English prose on 2026-05-26 via the clone
# subcommand below. Switch with --voice-id or FISH_AUDIO_VOICE_ID env.
DEFAULT_VOICE_ID = "282fa853838548af9803ed5b78226253"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tts-tool",
        description="Convert CleanText on stdin (or -i FILE) to MP3 via Fish Audio. "
                    "Use `tts-tool clone --help` to upload a voice sample.",
    )
    p.add_argument("-i", "--input", type=Path, default=None,
                   help="Read CleanText from FILE instead of stdin.")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Write MP3 to FILE instead of stdout.")
    p.add_argument("--no-cache", action="store_true",
                   help="Bypass the chunk cache for this run.")
    p.add_argument("--voice-id", type=str, default=None, metavar="ID",
                   help="Fish Audio voice library reference_id (32-hex). "
                        "Overrides FISH_AUDIO_VOICE_ID env. Default: "
                        f"Jonathan-cloned ({DEFAULT_VOICE_ID[:8]}...). "
                        "Browse Fish's library via SDK voices.list().")
    p.add_argument("-j", "--concurrency", type=int, default=3,
                   metavar="N",
                   help="Synthesize up to N chunks in parallel against the "
                        "Fish API (default 3). Paragraph-dense prose typically "
                        "produces 30-50 chunks; serial synth dominates wall "
                        "time. Higher concurrency trips Fish's per-key rate "
                        "limit (HTTP 429); 3 is empirically safe. 1 = strict "
                        "serial. Ignored when --prime-tail is set (chained "
                        "synth is inherently sequential).")
    p.add_argument("--prime-tail", type=float, default=None, metavar="SECONDS",
                   help="Feed the last N seconds of chunk N as a "
                        "ReferenceAudio to chunk N+1, reducing pitch drift "
                        "between chunks. Reset at paragraph boundaries "
                        "(chunks with silence_after > 0) so cross-paragraph "
                        "priming doesn't fight the natural pause. Typical: "
                        "2.0-3.0s. Forces sequential synth, breaks "
                        "concurrency. Requires ffmpeg + ffprobe on PATH.")
    return p


def _read_input(input_path: Path | None) -> str:
    if input_path is not None:
        return input_path.read_text(encoding="utf-8", errors="replace")
    data = sys.stdin.buffer.read()
    return data.decode("utf-8", errors="replace")


def _write_output(data: bytes, output: Path | None) -> None:
    if output is not None:
        output.write_bytes(data)
        return
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _build_clone_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tts-tool clone",
        description="Clone a voice via Fish Audio. Uploads one or more sample "
                    "audio files, prints the new reference_id. Plug that id into "
                    "`tts-tool --voice-id <id>` or `FISH_AUDIO_VOICE_ID` env.",
    )
    p.add_argument("--sample", type=Path, action="append", default=None,
                   metavar="FILE",
                   help="Path to a voice sample (WAV/MP3/FLAC/M4A/OGG; "
                        "anything Fish accepts). Repeatable. At least one of "
                        "--sample or --record is required; both may be combined.")
    p.add_argument("--record", type=float, default=None, metavar="SECONDS",
                   help="Capture a fresh sample inline for SECONDS via "
                        "pw-record (PipeWire), arecord (ALSA), or ffmpeg "
                        "(PulseAudio) — whichever is on PATH. Saved to a temp "
                        "WAV, appended to the sample list. Recording starts "
                        "immediately; use Ctrl-C to abort before the timer.")
    p.add_argument("--title", type=str, required=True,
                   help="Human-readable name for the cloned voice.")
    p.add_argument("--description", type=str, default=None,
                   help="Optional description shown in your Fish voice library.")
    p.add_argument("--text", type=str, action="append", default=None,
                   metavar="TRANSCRIPT",
                   help="Optional transcript for the corresponding --sample. "
                        "Repeat in same order as --sample, or omit entirely. "
                        "Transcripts improve cloning fidelity.")
    p.add_argument("--visibility", choices=["private", "unlist", "public"],
                   default="private",
                   help="Voice visibility on Fish (default: private).")
    return p


def _clone_main(argv: list[str]) -> int:
    args = _build_clone_parser().parse_args(argv)

    sample_paths: list[Path] = list(args.sample) if args.sample else []
    if not sample_paths and args.record is None:
        _log("error: provide at least one --sample FILE or --record SECONDS")
        return 6

    try:
        api_key = synthesize.read_api_key()
    except synthesize.MissingAPIKey as e:
        _log(f"error: {e}")
        return 5

    recorded_path: Path | None = None
    if args.record is not None:
        try:
            rec = record.find_recorder()
        except record.RecordError as e:
            _log(f"error: {e}")
            return 7
        recorded_path = Path(tempfile.mkstemp(prefix="tts-clone-", suffix=".wav")[1])
        _log(f"recording {args.record:.0f}s via {rec} -> {recorded_path}")
        try:
            record.record_wav(recorded_path, args.record, recorder=rec)
        except (record.RecordError, subprocess.CalledProcessError) as e:
            _log(f"error: recording failed: {e}")
            return 7
        sample_paths.append(recorded_path)

    client = synthesize.make_client(api_key)
    try:
        voice_id = clone.clone_voice(
            client,
            title=args.title,
            sample_paths=sample_paths,
            description=args.description,
            texts=list(args.text) if args.text else None,
            visibility=args.visibility,
        )
    except clone.CloneError as e:
        _log(f"error: {e}")
        return 6
    except Exception as e:
        _log(f"error: Fish Audio clone failed: {e}")
        return 1
    finally:
        if recorded_path is not None and recorded_path.exists():
            recorded_path.unlink()

    _log(f"cloned voice '{args.title}' -> reference_id:")
    print(voice_id)
    _log("use with: tts-tool --voice-id " + voice_id)
    _log("or set: export FISH_AUDIO_VOICE_ID=" + voice_id)
    return 0


def _build_record_clone_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tts-tool record-clone",
        description="Interactive multi-segment voice cloning. Walks you through "
                    "a register-varied script (warmup / tender / playful / "
                    "reflective / stability / intimate), recording each segment "
                    "with stop-on-ENTER and per-segment accept/redo prompts. "
                    "Bundles the accepted takes into one Fish Audio voice model.",
    )
    p.add_argument("--title", type=str, required=True,
                   help="Human-readable name for the cloned voice.")
    p.add_argument("--speaker-name", type=str, default="",
                   help="First name of the speaker. Substituted into the "
                        "warmup segment's '{name}' placeholder. Omit for "
                        "anonymous cloning (the name introduction is "
                        "dropped from the script).")
    p.add_argument("--description", type=str, default=None,
                   help="Optional description in your Fish voice library.")
    p.add_argument("--visibility", choices=["private", "unlist", "public"],
                   default="private",
                   help="Voice visibility on Fish (default: private).")
    p.add_argument("--keep-takes", action="store_true",
                   help="Keep the per-segment WAVs on disk after upload "
                        "(under --work-dir). Default: delete on success.")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="Where to write per-segment WAVs (default: a temp dir).")
    return p


def _clone_interactive_main(argv: list[str]) -> int:
    args = _build_record_clone_parser().parse_args(argv)

    try:
        api_key = synthesize.read_api_key()
    except synthesize.MissingAPIKey as e:
        _log(f"error: {e}")
        return 5

    try:
        recorder = record.find_recorder()
    except record.RecordError as e:
        _log(f"error: {e}")
        return 7

    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="tts-record-clone-"))

    try:
        accepted = record_interactive.run_interactive_capture(
            record_interactive.DEFAULT_SCRIPT,
            work_dir,
            recorder=recorder,
            speaker_name=args.speaker_name,
        )
    except KeyboardInterrupt:
        _log("aborted by user")
        return 130

    if not accepted:
        _log("error: no segments accepted, nothing to upload")
        return 6

    _log(f"uploading {len(accepted)} accepted segments to Fish Audio...")
    sample_paths = [w for w, _ in accepted]
    texts = [t for _, t in accepted]

    client = synthesize.make_client(api_key)
    try:
        voice_id = clone.clone_voice(
            client,
            title=args.title,
            sample_paths=sample_paths,
            description=args.description,
            texts=texts,
            visibility=args.visibility,
        )
    except clone.CloneError as e:
        _log(f"error: {e}")
        return 6
    except Exception as e:
        _log(f"error: Fish Audio clone failed: {e}")
        return 1
    finally:
        if not args.keep_takes and args.work_dir is None:
            for w in sample_paths:
                try:
                    w.unlink()
                except OSError:
                    pass

    _log(f"cloned voice '{args.title}' from {len(accepted)} segments -> reference_id:")
    print(voice_id)
    _log("use with: tts-tool --voice-id " + voice_id)
    _log("or set: export FISH_AUDIO_VOICE_ID=" + voice_id)
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "clone":
        return _clone_main(argv[1:])
    if argv and argv[0] == "record-clone":
        return _clone_interactive_main(argv[1:])

    args = _build_parser().parse_args(argv)

    if args.output is None and sys.stdout.isatty():
        _log("error: refusing to write MP3 binary to TTY; use -o FILE")
        return 3

    try:
        api_key = synthesize.read_api_key()
    except synthesize.MissingAPIKey as e:
        _log(f"error: {e}")
        return 5

    raw = _read_input(args.input)
    chunks = chunkmod.chunk_text(raw)
    if not chunks:
        _log("error: no text to synthesize")
        return 2

    model = os.environ.get("FISH_AUDIO_MODEL", "s2-pro").strip() or "s2-pro"
    voice_id = (
        args.voice_id
        or os.environ.get("FISH_AUDIO_VOICE_ID", "").strip()
        or DEFAULT_VOICE_ID
    )
    speed = float(os.environ.get("LISTEN_SPEED", "1.0"))
    cache_root = os.environ.get("LISTEN_CACHE_DIR") or None

    cdir = cache.chunks_dir(cache_root)
    client = synthesize.make_client(api_key)

    n = len(chunks)
    audio_per_chunk: list[bytes | None] = [None] * n
    cache_hits = 0

    if args.prime_tail is not None and args.prime_tail > 0:
        # Sequential prime-tail mode. Each chunk's tail conditions the
        # next chunk's synthesis. Resets at silence_after > 0 (paragraph
        # boundary) so cross-paragraph priming doesn't compete with the
        # natural pause.
        from .tail_extract import TailExtractError, tail_seconds
        _log(f"synthesizing {n} chunks (prime-tail={args.prime_tail:.1f}s, "
             "sequential)")
        prev_tail: bytes | None = None
        for slot, c in enumerate(chunks):
            key = cache.key_for(c.text, model, voice_id, speed,
                                prime_tail=prev_tail)
            cached = None if args.no_cache else cache.get(cdir, key)
            if cached is not None:
                cache_hits += 1
                _log(f"chunk {c.index + 1}/{n}: cache hit ({len(c.text)} chars)")
                audio_per_chunk[slot] = cached
            else:
                try:
                    audio = synthesize.synthesize_chunk(
                        client, c.text,
                        model=model, voice_id=voice_id, speed=speed,
                        prime_tail=prev_tail,
                    )
                except Exception as e:
                    _log(f"error: Fish Audio synthesis failed: {e}")
                    return 1
                if not args.no_cache:
                    cache.put(cdir, key, audio)
                audio_per_chunk[slot] = audio
                _log(f"chunk {c.index + 1}/{n}: done ({len(c.text)} chars)")

            # Decide tail for the NEXT chunk. Reset at paragraph break.
            if c.silence_after > 0:
                prev_tail = None
            else:
                try:
                    prev_tail = tail_seconds(
                        audio_per_chunk[slot], args.prime_tail
                    )
                except TailExtractError as e:
                    _log(f"warn: tail extract failed ({e}); resetting prime")
                    prev_tail = None
    else:
        to_synth: list[tuple[int, object, str]] = []
        for slot, c in enumerate(chunks):
            key = cache.key_for(c.text, model, voice_id, speed)
            cached = None if args.no_cache else cache.get(cdir, key)
            if cached is not None:
                cache_hits += 1
                _log(f"chunk {c.index + 1}/{n}: cache hit ({len(c.text)} chars)")
                audio_per_chunk[slot] = cached
                continue
            to_synth.append((slot, c, key))

        if to_synth:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            concurrency = max(1, args.concurrency)
            _log(f"synthesizing {len(to_synth)} chunks "
                 f"(concurrency={concurrency})")

            def _do_one(slot: int, c, key: str):
                audio = synthesize.synthesize_chunk(
                    client, c.text,
                    model=model, voice_id=voice_id, speed=speed,
                )
                return slot, c, key, audio

            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = [ex.submit(_do_one, slot, c, key)
                           for slot, c, key in to_synth]
                for fut in as_completed(futures):
                    try:
                        slot, c, key, audio = fut.result()
                    except Exception as e:
                        _log(f"error: Fish Audio synthesis failed: {e}")
                        return 1
                    if not args.no_cache:
                        cache.put(cdir, key, audio)
                    audio_per_chunk[slot] = audio
                    _log(f"chunk {c.index + 1}/{n}: done "
                         f"({len(c.text)} chars)")

    # All slots must be filled by now (cache hit or successful synth).
    assert all(a is not None for a in audio_per_chunk), "synth slot left empty"
    audio_per_chunk = [a for a in audio_per_chunk if a is not None]

    if len(audio_per_chunk) == 1:
        _write_output(audio_per_chunk[0], args.output)
        _log(f"done. 1 chunk ({cache_hits} cached).")
        return 0

    with tempfile.TemporaryDirectory(prefix="tts-tool-") as td:
        td_path = Path(td)
        parts: list[tuple[Path, float]] = []
        for i, (c, data) in enumerate(zip(chunks, audio_per_chunk)):
            p = td_path / f"{i:04d}.mp3"
            p.write_bytes(data)
            parts.append((p, c.silence_after))
        out_path = args.output if args.output is not None else td_path / "out.mp3"
        try:
            stitch_mp3s(parts, out_path)
        except StitchError as e:
            _log(f"error: {e}")
            return 4
        if args.output is None:
            _write_output(out_path.read_bytes(), None)

    _log(f"done. {n} chunks ({cache_hits} cached).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
