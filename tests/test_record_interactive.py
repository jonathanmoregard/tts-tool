"""Tests for the interactive multi-segment cloning flow."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tts_tool import cli, record_interactive
from tts_tool.record_interactive import (
    Segment,
    prompt_verdict,
    run_interactive_capture,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _silent_wav(path: Path, seconds: float = 0.2) -> bytes:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
         "-t", str(seconds), "-c:a", "pcm_s16le", str(path)],
        check=True,
    )
    return path.read_bytes()


def _two_seg_script() -> tuple[Segment, ...]:
    return (
        Segment("warmup", "neutral", "Warmup text.", 10.0),
        Segment("tender", "soft", "Tender text.", 10.0),
    )


def _make_inputs(seq: list[str]):
    """Returns an input() stand-in that pops responses from seq."""
    it = iter(seq)
    return lambda _prompt="": next(it)


def test_prompt_verdict_accepts_y_r_s_q():
    for letter in ["y", "r", "s", "q"]:
        assert prompt_verdict(prompt_input=lambda _="": letter) == letter


def test_prompt_verdict_empty_defaults_to_y():
    assert prompt_verdict(prompt_input=lambda _="": "") == "y"


def test_prompt_verdict_loops_on_invalid(capsys):
    seq = iter(["xyz", "nope", "y"])
    assert prompt_verdict(prompt_input=lambda _="": next(seq)) == "y"
    assert "invalid choice" in capsys.readouterr().err


def test_run_interactive_capture_accept_all(monkeypatch, tmp_path: Path):
    """Two-segment script, user accepts both. Returns both paths + texts."""
    segs = _two_seg_script()

    inputs = _make_inputs([
        "",   # press ENTER to start seg 1
        "y",  # accept seg 1
        "",   # press ENTER to start seg 2
        "y",  # accept seg 2
    ])

    def fake_record(out_path, max_seconds, *, recorder=None):
        out_path.write_bytes(b"RIFFFAKE" + out_path.name.encode())

    accepted = run_interactive_capture(
        segs, tmp_path, recorder="pw-record",
        prompt_input=inputs, record_fn=fake_record, play_fn=lambda _p: None,
    )
    assert len(accepted) == 2
    assert [t for _, t in accepted] == ["Warmup text.", "Tender text."]
    for p, _ in accepted:
        assert p.exists()


def test_run_interactive_capture_redo_then_accept(monkeypatch, tmp_path: Path):
    """User redoes seg 1 once, then accepts both."""
    segs = _two_seg_script()
    inputs = _make_inputs(["", "r", "", "y", "", "y"])

    call_counts: dict[str, int] = {}

    def fake_record(out_path, max_seconds, *, recorder=None):
        call_counts[out_path.name] = call_counts.get(out_path.name, 0) + 1
        out_path.write_bytes(b"RIFFFAKE")

    accepted = run_interactive_capture(
        segs, tmp_path, recorder="pw-record",
        prompt_input=inputs, record_fn=fake_record, play_fn=lambda _p: None,
    )
    assert len(accepted) == 2
    # seg 1 recorded twice (initial + redo), seg 2 once
    assert call_counts["01-warmup.wav"] == 2
    assert call_counts["02-tender.wav"] == 1


def test_run_interactive_capture_skip(tmp_path: Path):
    segs = _two_seg_script()
    inputs = _make_inputs(["", "s", "", "y"])

    def fake_record(out_path, _ms, *, recorder=None):
        out_path.write_bytes(b"X")

    accepted = run_interactive_capture(
        segs, tmp_path, recorder="pw-record",
        prompt_input=inputs, record_fn=fake_record, play_fn=lambda _p: None,
    )
    assert [t for _, t in accepted] == ["Tender text."]


def test_run_interactive_capture_quit_raises(tmp_path: Path):
    segs = _two_seg_script()
    inputs = _make_inputs(["", "q"])

    def fake_record(out_path, _ms, *, recorder=None):
        out_path.write_bytes(b"X")

    with pytest.raises(KeyboardInterrupt):
        run_interactive_capture(
            segs, tmp_path, recorder="pw-record",
            prompt_input=inputs, record_fn=fake_record, play_fn=lambda _p: None,
        )


def test_default_script_six_segments_and_all_have_text():
    script = record_interactive.DEFAULT_SCRIPT
    assert len(script) == 6
    labels = [s.label for s in script]
    assert labels == ["warmup", "tender", "playful", "reflective",
                      "stability", "intimate"]
    for s in script:
        assert s.transcript.strip()
        assert 10 <= s.max_seconds <= 60


@pytest.mark.skipif(not HAS_FFMPEG, reason="requires ffmpeg")
def test_cli_record_clone_e2e_accept_all(monkeypatch, capsys, tmp_path: Path):
    """End-to-end: accept-all flow uploads N samples + N transcripts."""
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    monkeypatch.setattr("tts_tool.record.find_recorder", lambda: "pw-record-stub")

    def fake_record(out_path, max_seconds, *, recorder=None):
        _silent_wav(out_path, 0.2)

    monkeypatch.setattr(
        "tts_tool.record_interactive.record_until_enter", fake_record
    )
    monkeypatch.setattr("tts_tool.record_interactive.play_back", lambda _p: None)

    # 6 segments, accept all: pattern is "<enter>y" * 6 = 12 inputs
    inputs = iter([""] + ["y"] + ([""] + ["y"]) * 5)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    fake_client = MagicMock()
    fake_client.voices.create.return_value = SimpleNamespace(id="interactive-id-123")
    monkeypatch.setattr(
        "tts_tool.synthesize.make_client", lambda *a, **k: fake_client
    )

    work_dir = tmp_path / "takes"
    rc = cli.main([
        "record-clone",
        "--title", "Jonathan-interactive",
        "--work-dir", str(work_dir),
    ])
    assert rc == 0, capsys.readouterr().err
    assert "interactive-id-123" in capsys.readouterr().out

    call = fake_client.voices.create.call_args.kwargs
    assert call["title"] == "Jonathan-interactive"
    assert len(call["voices"]) == 6
    assert all(v.startswith(b"RIFF") for v in call["voices"])
    assert len(call["texts"]) == 6
    # --work-dir was explicit, so takes should be preserved
    assert sorted(work_dir.glob("*.wav"))


def test_cli_record_clone_no_recorder_returns_7(monkeypatch, capsys):
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    from tts_tool.record import RecordError

    def boom():
        raise RecordError("no audio capture on PATH")

    monkeypatch.setattr("tts_tool.record.find_recorder", boom)
    rc = cli.main(["record-clone", "--title", "T"])
    assert rc == 7
    assert "no audio capture" in capsys.readouterr().err


def test_cli_record_clone_no_api_key(monkeypatch, capsys):
    monkeypatch.delenv("FISH_AUDIO_API_KEY", raising=False)
    monkeypatch.delenv("FISH_AUDIO_API_KEY_FILE", raising=False)
    rc = cli.main(["record-clone", "--title", "T"])
    assert rc == 5
    assert "FISH_AUDIO_API_KEY" in capsys.readouterr().err


def test_cli_record_clone_all_skipped_returns_6(monkeypatch, capsys, tmp_path: Path):
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "k")
    monkeypatch.setattr("tts_tool.record.find_recorder", lambda: "pw-record-stub")

    def fake_record(out_path, _ms, *, recorder=None):
        out_path.write_bytes(b"X")

    monkeypatch.setattr(
        "tts_tool.record_interactive.record_until_enter", fake_record
    )
    monkeypatch.setattr("tts_tool.record_interactive.play_back", lambda _p: None)

    # 6 segments, skip all: <enter>s repeated
    inputs = iter(([""] + ["s"]) * 6)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    rc = cli.main([
        "record-clone", "--title", "Empty",
        "--work-dir", str(tmp_path / "takes"),
    ])
    assert rc == 6
    assert "no segments accepted" in capsys.readouterr().err
