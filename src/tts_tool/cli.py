"""tts-tool entrypoint. CleanText -> MP3."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from . import chunk as chunkmod
from . import cache, synthesize
from .stitch import StitchError, stitch_mp3s

# Fish Audio voice library reference_id. "Adrian — A steady and reliable
# narrator", male/middle-aged/deep/measured/serious; sounded clean on
# long-form prose during empirical testing 2026-05-22. Switch with
# --voice-id or FISH_AUDIO_VOICE_ID env.
DEFAULT_VOICE_ID = "bf322df2096a46f18c579d0baa36f41d"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tts-tool",
        description="Convert CleanText on stdin (or -i FILE) to MP3 via Fish Audio.",
    )
    p.add_argument("-i", "--input", type=Path, default=None,
                   help="Read CleanText from FILE instead of stdin.")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Write MP3 to FILE instead of stdout.")
    p.add_argument("--no-cache", action="store_true",
                   help="Bypass the chunk cache for this run.")
    p.add_argument("--voice-id", type=str, default=None, metavar="ID",
                   help="Fish Audio voice library reference_id (32-hex). "
                        "Overrides FISH_AUDIO_VOICE_ID env. Default: Adrian "
                        f"({DEFAULT_VOICE_ID[:8]}...), 'steady reliable "
                        "narrator', empirically good for long-form prose. "
                        "Browse Fish's library via SDK voices.list().")
    p.add_argument("-j", "--concurrency", type=int, default=3,
                   metavar="N",
                   help="Synthesize up to N chunks in parallel against the "
                        "Fish API (default 3). Paragraph-dense prose typically "
                        "produces 30-50 chunks; serial synth dominates wall "
                        "time. Higher concurrency trips Fish's per-key rate "
                        "limit (HTTP 429); 3 is empirically safe. 1 = strict "
                        "serial.")
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


def main(argv: list[str] | None = None) -> int:
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
    to_synth: list[tuple[int, object, str]] = []  # (slot, chunk, cache_key)
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
        _log(f"synthesizing {len(to_synth)} chunks (concurrency={concurrency})")

        def _do_one(slot: int, c, key: str):
            audio = synthesize.synthesize_chunk(
                client, c.text, model=model, voice_id=voice_id, speed=speed,
            )
            return slot, c, key, audio

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(_do_one, slot, c, key) for slot, c, key in to_synth]
            for fut in as_completed(futures):
                try:
                    slot, c, key, audio = fut.result()
                except Exception as e:
                    _log(f"error: Fish Audio synthesis failed: {e}")
                    return 1
                if not args.no_cache:
                    cache.put(cdir, key, audio)
                audio_per_chunk[slot] = audio
                _log(f"chunk {c.index + 1}/{n}: done ({len(c.text)} chars)")

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
