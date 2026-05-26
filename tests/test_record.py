"""Unit tests for tts_tool.record dispatch + error paths.

The actual recorder subprocess is not invoked here — the e2e CLI tests in
test_clone.py monkeypatch record_wav with an ffmpeg-generated silent WAV.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tts_tool import record
from tts_tool.record import RecordError, find_recorder, record_wav


def test_find_recorder_prefers_pw_record(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda c: c if c == "pw-record" else None)
    assert find_recorder() == "pw-record"


def test_find_recorder_falls_through_to_arecord(monkeypatch):
    monkeypatch.setattr(
        "shutil.which",
        lambda c: c if c in {"arecord", "ffmpeg"} else None,
    )
    assert find_recorder() == "arecord"


def test_find_recorder_falls_through_to_ffmpeg(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda c: c if c == "ffmpeg" else None)
    assert find_recorder() == "ffmpeg"


def test_find_recorder_none_raises(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda c: None)
    with pytest.raises(RecordError, match="no audio capture"):
        find_recorder()


def test_record_wav_zero_seconds_raises(tmp_path: Path):
    with pytest.raises(RecordError, match="positive"):
        record_wav(tmp_path / "x.wav", 0)


def test_record_wav_pw_record_dispatch(monkeypatch, tmp_path: Path):
    out = tmp_path / "out.wav"
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        out.write_bytes(b"RIFFFAKE")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("subprocess.run", fake_run)
    record_wav(out, 1.0, recorder="pw-record")
    assert calls[0][0] == "pw-record"
    assert "--rate=44100" in calls[0]


def test_record_wav_arecord_dispatch(monkeypatch, tmp_path: Path):
    out = tmp_path / "out.wav"
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        out.write_bytes(b"RIFFFAKE")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("subprocess.run", fake_run)
    record_wav(out, 2.0, recorder="arecord")
    assert calls[0][:5] == ["arecord", "-q", "-f", "S16_LE", "-r"]
    assert "-d" in calls[0]
    assert "2" in calls[0]


def test_record_wav_ffmpeg_dispatch(monkeypatch, tmp_path: Path):
    out = tmp_path / "out.wav"
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        out.write_bytes(b"RIFFFAKE")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("subprocess.run", fake_run)
    record_wav(out, 1.5, recorder="ffmpeg")
    assert calls[0][0] == "ffmpeg"
    assert "pulse" in calls[0]


def test_record_wav_empty_output_raises(monkeypatch, tmp_path: Path):
    out = tmp_path / "out.wav"
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0] if a else [], 0),
    )
    with pytest.raises(RecordError, match="produced no output"):
        record_wav(out, 1.0, recorder="pw-record")


def test_record_wav_unknown_recorder_raises(tmp_path: Path):
    with pytest.raises(RecordError, match="unknown recorder"):
        record_wav(tmp_path / "x.wav", 1.0, recorder="nope")
