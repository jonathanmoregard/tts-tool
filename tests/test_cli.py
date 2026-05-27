"""End-to-end CLI tests with mocked synth."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tts_tool import cli

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _silence_mp3(path: Path, seconds: float = 0.1) -> bytes:
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
            "-t", str(seconds), "-b:a", "128k",
            str(path),
        ],
        check=True,
    )
    return path.read_bytes()


def test_cli_exit_5_no_api_key(monkeypatch, capsys, tmp_path: Path):
    monkeypatch.delenv("FISH_AUDIO_API_KEY", raising=False)
    monkeypatch.delenv("FISH_AUDIO_API_KEY_FILE", raising=False)
    monkeypatch.setattr("sys.stdin", _StdinBytes(b"hello world."))
    rc = cli.main(["-o", str(tmp_path / "out.mp3")])
    assert rc == 5
    assert "FISH_AUDIO_API_KEY" in capsys.readouterr().err


def test_cli_exit_2_empty_input(monkeypatch, capsys, tmp_path: Path):
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    monkeypatch.setattr("sys.stdin", _StdinBytes(b"   \n\n  "))
    rc = cli.main(["-o", str(tmp_path / "out.mp3")])
    assert rc == 2
    assert "no text" in capsys.readouterr().err


def test_cli_exit_3_tty_without_output(monkeypatch, capsys):
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    rc = cli.main([])
    assert rc == 3
    assert "TTY" in capsys.readouterr().err


@pytest.mark.skipif(not HAS_FFMPEG, reason="requires ffmpeg")
def test_cli_end_to_end_single_chunk(monkeypatch, tmp_path: Path):
    """Short input -> 1 chunk -> direct write, no stitch."""
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    monkeypatch.setenv("LISTEN_CACHE_DIR", str(tmp_path / "cache"))

    fake = _silence_mp3(tmp_path / "fake.mp3", 0.1)
    fake_client = MagicMock()
    fake_client.tts.convert.return_value = fake
    monkeypatch.setattr("tts_tool.synthesize.make_client", lambda *a, **k: fake_client)

    src = tmp_path / "in.txt"
    src.write_text("Hello world.\n\nThis is a test.")
    out = tmp_path / "out.mp3"

    rc = cli.main(["-i", str(src), "-o", str(out)])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0
    assert fake_client.tts.convert.call_count == 1


@pytest.mark.skipif(not HAS_FFMPEG, reason="requires ffmpeg")
def test_cli_end_to_end_cache_hit_skips_synth(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    monkeypatch.setenv("LISTEN_CACHE_DIR", str(tmp_path / "cache"))

    fake = _silence_mp3(tmp_path / "fake.mp3", 0.1)
    fake_client = MagicMock()
    fake_client.tts.convert.return_value = fake
    monkeypatch.setattr("tts_tool.synthesize.make_client", lambda *a, **k: fake_client)

    src = tmp_path / "in.txt"
    src.write_text("Cached content.")
    out1 = tmp_path / "out1.mp3"
    out2 = tmp_path / "out2.mp3"

    assert cli.main(["-i", str(src), "-o", str(out1)]) == 0
    first_call_count = fake_client.tts.convert.call_count
    assert cli.main(["-i", str(src), "-o", str(out2)]) == 0
    assert fake_client.tts.convert.call_count == first_call_count


@pytest.mark.skipif(not HAS_FFMPEG, reason="requires ffmpeg")
def test_cli_no_cache_flag_skips_cache(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    monkeypatch.setenv("LISTEN_CACHE_DIR", str(tmp_path / "cache"))

    fake = _silence_mp3(tmp_path / "fake.mp3", 0.1)
    fake_client = MagicMock()
    fake_client.tts.convert.return_value = fake
    monkeypatch.setattr("tts_tool.synthesize.make_client", lambda *a, **k: fake_client)

    src = tmp_path / "in.txt"
    src.write_text("No cache please.")
    out1 = tmp_path / "out1.mp3"
    out2 = tmp_path / "out2.mp3"

    assert cli.main(["-i", str(src), "-o", str(out1), "--no-cache"]) == 0
    assert cli.main(["-i", str(src), "-o", str(out2), "--no-cache"]) == 0
    assert fake_client.tts.convert.call_count == 2


def _fake_chunks(specs: list[tuple[str, float]]):
    """Build a list of Chunk-like objects for tests. Each spec is (text, silence_after)."""
    from tts_tool.chunk import Chunk
    return [Chunk(text=t, silence_after=s, index=i) for i, (t, s) in enumerate(specs)]


@pytest.mark.skipif(not HAS_FFMPEG, reason="requires ffmpeg")
def test_cli_prime_tail_chains_within_paragraph_and_resets_at_break(
    monkeypatch, tmp_path: Path,
):
    """--prime-tail: priming chains within a paragraph and resets at silence_after > 0."""
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    monkeypatch.setenv("LISTEN_CACHE_DIR", str(tmp_path / "cache"))

    fake = _silence_mp3(tmp_path / "fake.mp3", 0.5)
    fake_client = MagicMock()
    fake_client.tts.convert.return_value = fake
    monkeypatch.setattr("tts_tool.synthesize.make_client", lambda *a, **k: fake_client)

    # 4 chunks: A (no break), B (paragraph break after), C (no break), D.
    chunks = _fake_chunks([
        ("Alpha.", 0.0),
        ("Bravo.", 0.6),
        ("Charlie.", 0.0),
        ("Delta.", 0.0),
    ])
    monkeypatch.setattr("tts_tool.cli.chunkmod.chunk_text", lambda _raw: chunks)

    src = tmp_path / "in.txt"
    src.write_text("any text; chunker is monkeypatched")
    out = tmp_path / "out.mp3"

    rc = cli.main(["-i", str(src), "-o", str(out), "--prime-tail", "0.2"])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0

    calls = fake_client.tts.convert.call_args_list
    assert len(calls) == 4
    # Chunk 0 (Alpha): first ever, no prior tail.
    assert calls[0].kwargs["references"] is None
    # Chunk 1 (Bravo): primed by Alpha's tail.
    assert calls[1].kwargs["references"] is not None
    assert len(calls[1].kwargs["references"]) == 1
    # Chunk 2 (Charlie): first chunk of new paragraph (Bravo had silence_after).
    # So priming was RESET — references should be None again.
    assert calls[2].kwargs["references"] is None
    # Chunk 3 (Delta): primed by Charlie's tail.
    assert calls[3].kwargs["references"] is not None


@pytest.mark.skipif(not HAS_FFMPEG, reason="requires ffmpeg")
def test_cli_prime_cross_paragraph_chains_through_silence(
    monkeypatch, tmp_path: Path,
):
    """--prime-cross-paragraph overrides the paragraph-break reset."""
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    monkeypatch.setenv("LISTEN_CACHE_DIR", str(tmp_path / "cache"))

    fake = _silence_mp3(tmp_path / "fake.mp3", 0.5)
    fake_client = MagicMock()
    fake_client.tts.convert.return_value = fake
    monkeypatch.setattr("tts_tool.synthesize.make_client", lambda *a, **k: fake_client)

    chunks = _fake_chunks([
        ("Alpha.", 0.6),   # paragraph break after
        ("Bravo.", 0.6),   # and another
        ("Charlie.", 0.0),
    ])
    monkeypatch.setattr("tts_tool.cli.chunkmod.chunk_text", lambda _raw: chunks)

    src = tmp_path / "in.txt"
    src.write_text("ignored")
    rc = cli.main([
        "-i", str(src), "-o", str(tmp_path / "out.mp3"),
        "--prime-tail", "0.2", "--prime-cross-paragraph",
    ])
    assert rc == 0
    calls = fake_client.tts.convert.call_args_list
    assert len(calls) == 3
    # Chunk 0: first ever, no prior tail.
    assert calls[0].kwargs["references"] is None
    # Chunks 1, 2: primed despite preceding silence_after > 0.
    assert calls[1].kwargs["references"] is not None
    assert calls[2].kwargs["references"] is not None


@pytest.mark.skipif(not HAS_FFMPEG, reason="requires ffmpeg")
def test_cli_prime_tail_serializes(monkeypatch, tmp_path: Path):
    """--prime-tail forces sequential synth even with -j set."""
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    monkeypatch.setenv("LISTEN_CACHE_DIR", str(tmp_path / "cache"))

    fake = _silence_mp3(tmp_path / "fake.mp3", 0.5)
    call_order: list[int] = []

    def fake_convert(**kwargs):
        call_order.append(len(call_order))
        return fake

    fake_client = MagicMock()
    fake_client.tts.convert.side_effect = fake_convert
    monkeypatch.setattr("tts_tool.synthesize.make_client", lambda *a, **k: fake_client)

    chunks = _fake_chunks([("One.", 0.0), ("Two.", 0.0), ("Three.", 0.0)])
    monkeypatch.setattr("tts_tool.cli.chunkmod.chunk_text", lambda _raw: chunks)

    src = tmp_path / "in.txt"
    src.write_text("any")
    rc = cli.main([
        "-i", str(src), "-o", str(tmp_path / "out.mp3"),
        "--prime-tail", "0.2", "-j", "5",
    ])
    assert rc == 0
    # Serial execution → call_order is 0,1,2 in deterministic order
    assert call_order == [0, 1, 2]


class _StdinBytes:
    """Minimal stdin stand-in: .buffer.read() returns the bytes."""
    def __init__(self, data: bytes) -> None:
        self.buffer = _Buf(data)

    def isatty(self) -> bool:
        return False


class _Buf:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data
