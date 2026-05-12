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
