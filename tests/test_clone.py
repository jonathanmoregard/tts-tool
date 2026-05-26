"""Tests for voice cloning."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tts_tool import cli
from tts_tool.clone import CloneError, clone_voice

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _silent_wav(path: Path, seconds: float = 0.2) -> bytes:
    """Generate a real silent mono 44.1 kHz s16 WAV via ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
         "-t", str(seconds), "-c:a", "pcm_s16le",
         str(path)],
        check=True,
    )
    return path.read_bytes()


def _fake_client(voice_id: str = "newvoiceid12345") -> MagicMock:
    client = MagicMock()
    client.voices.create.return_value = SimpleNamespace(id=voice_id)
    return client


def test_clone_voice_returns_id_and_uploads_bytes(tmp_path: Path):
    s1 = tmp_path / "a.wav"
    s1.write_bytes(b"AUDIO1")
    s2 = tmp_path / "b.wav"
    s2.write_bytes(b"AUDIO2")
    client = _fake_client("vid-abc")

    rid = clone_voice(
        client,
        title="Me",
        sample_paths=[s1, s2],
        description="my voice",
        texts=["transcript a", "transcript b"],
        visibility="private",
    )

    assert rid == "vid-abc"
    client.voices.create.assert_called_once_with(
        title="Me",
        voices=[b"AUDIO1", b"AUDIO2"],
        description="my voice",
        texts=["transcript a", "transcript b"],
        visibility="private",
    )


def test_clone_voice_omits_texts_when_none(tmp_path: Path):
    s1 = tmp_path / "a.wav"
    s1.write_bytes(b"X")
    client = _fake_client()
    clone_voice(client, title="t", sample_paths=[s1])
    kwargs = client.voices.create.call_args.kwargs
    assert kwargs["texts"] is None


def test_clone_voice_empty_samples_raises():
    with pytest.raises(CloneError, match="at least one sample"):
        clone_voice(MagicMock(), title="t", sample_paths=[])


def test_clone_voice_texts_count_mismatch_raises(tmp_path: Path):
    s1 = tmp_path / "a.wav"
    s1.write_bytes(b"X")
    with pytest.raises(CloneError, match="must match sample count"):
        clone_voice(
            MagicMock(),
            title="t",
            sample_paths=[s1],
            texts=["one", "two"],
        )


def test_clone_voice_missing_file_raises(tmp_path: Path):
    missing = tmp_path / "nope.wav"
    with pytest.raises(CloneError, match="sample not found"):
        clone_voice(MagicMock(), title="t", sample_paths=[missing])


def test_cli_clone_happy_path(monkeypatch, capsys, tmp_path: Path):
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    s = tmp_path / "voice.wav"
    s.write_bytes(b"RIFFFAKE")

    fake = _fake_client("returnedid999")
    monkeypatch.setattr("tts_tool.synthesize.make_client", lambda *a, **k: fake)

    rc = cli.main([
        "clone",
        "--sample", str(s),
        "--title", "Jonathan",
        "--description", "test clone",
    ])
    assert rc == 0
    cap = capsys.readouterr()
    assert "returnedid999" in cap.out
    assert "cloned voice 'Jonathan'" in cap.err
    fake.voices.create.assert_called_once()


def test_cli_clone_no_api_key(monkeypatch, capsys, tmp_path: Path):
    monkeypatch.delenv("FISH_AUDIO_API_KEY", raising=False)
    monkeypatch.delenv("FISH_AUDIO_API_KEY_FILE", raising=False)
    s = tmp_path / "voice.wav"
    s.write_bytes(b"X")
    rc = cli.main(["clone", "--sample", str(s), "--title", "T"])
    assert rc == 5
    assert "FISH_AUDIO_API_KEY" in capsys.readouterr().err


def test_cli_clone_missing_sample_file(monkeypatch, capsys, tmp_path: Path):
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    fake = _fake_client()
    monkeypatch.setattr("tts_tool.synthesize.make_client", lambda *a, **k: fake)
    rc = cli.main([
        "clone",
        "--sample", str(tmp_path / "missing.wav"),
        "--title", "T",
    ])
    assert rc == 6
    assert "sample not found" in capsys.readouterr().err
    fake.voices.create.assert_not_called()


def test_cli_clone_sdk_error_returns_1(monkeypatch, capsys, tmp_path: Path):
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    s = tmp_path / "voice.wav"
    s.write_bytes(b"X")
    fake = MagicMock()
    fake.voices.create.side_effect = RuntimeError("upstream 500")
    monkeypatch.setattr("tts_tool.synthesize.make_client", lambda *a, **k: fake)
    rc = cli.main(["clone", "--sample", str(s), "--title", "T"])
    assert rc == 1
    assert "Fish Audio clone failed" in capsys.readouterr().err


def test_cli_clone_requires_sample_or_record(monkeypatch, capsys):
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    rc = cli.main(["clone", "--title", "T"])
    assert rc == 6
    assert "at least one --sample" in capsys.readouterr().err


@pytest.mark.skipif(not HAS_FFMPEG, reason="requires ffmpeg")
def test_cli_clone_e2e_real_wav_sample(monkeypatch, capsys, tmp_path: Path):
    """End-to-end: real silent WAV on disk -> clone CLI -> bytes uploaded."""
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    wav = tmp_path / "voice.wav"
    wav_bytes = _silent_wav(wav, 0.5)
    assert wav_bytes.startswith(b"RIFF")  # real WAV header

    fake = _fake_client("realid111")
    monkeypatch.setattr("tts_tool.synthesize.make_client", lambda *a, **k: fake)

    rc = cli.main([
        "clone", "--sample", str(wav), "--title", "Jonathan",
    ])
    assert rc == 0
    assert "realid111" in capsys.readouterr().out
    call = fake.voices.create.call_args.kwargs
    assert call["voices"] == [wav_bytes]
    assert call["title"] == "Jonathan"


@pytest.mark.skipif(not HAS_FFMPEG, reason="requires ffmpeg")
def test_cli_clone_e2e_record_path(monkeypatch, capsys, tmp_path: Path):
    """End-to-end: --record causes recorder to produce a WAV, which is uploaded."""
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    fake = _fake_client("recid222")
    monkeypatch.setattr("tts_tool.synthesize.make_client", lambda *a, **k: fake)

    monkeypatch.setattr("tts_tool.record.find_recorder", lambda: "ffmpeg-stub")

    captured_paths: list[Path] = []

    def fake_record(out_path: Path, seconds: float, *, recorder=None):
        captured_paths.append(out_path)
        _silent_wav(out_path, 0.3)

    monkeypatch.setattr("tts_tool.record.record_wav", fake_record)

    rc = cli.main([
        "clone", "--record", "0.3", "--title", "Jonathan",
    ])
    assert rc == 0
    assert "recid222" in capsys.readouterr().out
    assert len(captured_paths) == 1
    # Temp file should be cleaned up after upload.
    assert not captured_paths[0].exists()
    call = fake.voices.create.call_args.kwargs
    assert len(call["voices"]) == 1
    assert call["voices"][0].startswith(b"RIFF")


@pytest.mark.skipif(not HAS_FFMPEG, reason="requires ffmpeg")
def test_cli_clone_record_plus_sample_combines(monkeypatch, capsys, tmp_path: Path):
    """--record appended to --sample list, both uploaded."""
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    existing = tmp_path / "existing.wav"
    existing_bytes = _silent_wav(existing, 0.2)

    fake = _fake_client("comboid333")
    monkeypatch.setattr("tts_tool.synthesize.make_client", lambda *a, **k: fake)
    monkeypatch.setattr("tts_tool.record.find_recorder", lambda: "ffmpeg-stub")
    monkeypatch.setattr(
        "tts_tool.record.record_wav",
        lambda out_path, seconds, recorder=None: _silent_wav(out_path, 0.2),
    )

    rc = cli.main([
        "clone",
        "--sample", str(existing),
        "--record", "0.2",
        "--title", "Combo",
    ])
    assert rc == 0
    voices = fake.voices.create.call_args.kwargs["voices"]
    assert len(voices) == 2
    assert voices[0] == existing_bytes
    assert voices[1].startswith(b"RIFF")


def test_cli_clone_no_recorder_returns_7(monkeypatch, capsys):
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    from tts_tool.record import RecordError

    def boom():
        raise RecordError("no recorder on PATH")

    monkeypatch.setattr("tts_tool.record.find_recorder", boom)
    rc = cli.main(["clone", "--record", "1", "--title", "T"])
    assert rc == 7
    assert "no recorder" in capsys.readouterr().err
