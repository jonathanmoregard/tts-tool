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
    p.add_argument("--paragraph-silence", type=float, default=1.5,
                   metavar="SECONDS",
                   help="Insert SECONDS of silence between every concatenated "
                        "chunk at stitch time (default 1.5s). Empirically Fish "
                        "s2-pro drops `[pause]` tags at chunk edges, so this "
                        "is the only reliable way to get real paragraph beats "
                        "in the final audio. Set 0 to disable.")
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
    voice_id = os.environ.get("FISH_AUDIO_VOICE_ID", "").strip() or None
    speed = float(os.environ.get("LISTEN_SPEED", "1.0"))
    cache_root = os.environ.get("LISTEN_CACHE_DIR") or None

    cdir = cache.chunks_dir(cache_root)
    client = synthesize.make_client(api_key)

    n = len(chunks)
    audio_per_chunk: list[bytes] = []
    cache_hits = 0
    for c in chunks:
        key = cache.key_for(c.text, model, voice_id, speed)
        cached = None if args.no_cache else cache.get(cdir, key)
        if cached is not None:
            cache_hits += 1
            _log(f"chunk {c.index + 1}/{n}: cache hit ({len(c.text)} chars)")
            audio_per_chunk.append(cached)
            continue
        _log(f"chunk {c.index + 1}/{n}: synthesizing ({len(c.text)} chars)")
        try:
            audio = synthesize.synthesize_chunk(
                client, c.text, model=model, voice_id=voice_id, speed=speed,
            )
        except Exception as e:
            _log(f"error: Fish Audio synthesis failed on chunk {c.index + 1}: {e}")
            return 1
        if not args.no_cache:
            cache.put(cdir, key, audio)
        audio_per_chunk.append(audio)

    if len(audio_per_chunk) == 1:
        _write_output(audio_per_chunk[0], args.output)
        _log(f"done. 1 chunk ({cache_hits} cached).")
        return 0

    with tempfile.TemporaryDirectory(prefix="tts-tool-") as td:
        td_path = Path(td)
        parts: list[Path] = []
        for i, data in enumerate(audio_per_chunk):
            p = td_path / f"{i:04d}.mp3"
            p.write_bytes(data)
            parts.append(p)
        out_path = args.output if args.output is not None else td_path / "out.mp3"
        try:
            stitch_mp3s(
                parts, out_path,
                inter_chunk_silence_seconds=args.paragraph_silence,
            )
        except StitchError as e:
            _log(f"error: {e}")
            return 4
        if args.output is None:
            _write_output(out_path.read_bytes(), None)

    _log(f"done. {n} chunks ({cache_hits} cached).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
