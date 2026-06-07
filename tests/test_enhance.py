"""Tests for the Auphonic enhance pass."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from tts_tool.enhance import (
    DEFAULT_DEBREATH_DB,
    EnhanceError,
    auphonic_available,
    enhance_with_auphonic,
)


def test_auphonic_available_false_when_binary_missing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setenv("AUPHONIC_API_KEY", "k")
    assert auphonic_available() is False


def test_auphonic_available_false_when_api_key_missing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/auphonic")
    monkeypatch.delenv("AUPHONIC_API_KEY", raising=False)
    assert auphonic_available() is False


def test_auphonic_available_true_when_both_present(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/auphonic")
    monkeypatch.setenv("AUPHONIC_API_KEY", "k")
    assert auphonic_available() is True


def test_enhance_raises_when_binary_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setenv("AUPHONIC_API_KEY", "k")
    src = tmp_path / "in.wav"
    src.write_bytes(b"X")
    with pytest.raises(EnhanceError, match="not on PATH"):
        enhance_with_auphonic(src, tmp_path / "out.wav")


def test_enhance_raises_when_api_key_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/auphonic")
    monkeypatch.delenv("AUPHONIC_API_KEY", raising=False)
    src = tmp_path / "in.wav"
    src.write_bytes(b"X")
    with pytest.raises(EnhanceError, match="AUPHONIC_API_KEY"):
        enhance_with_auphonic(src, tmp_path / "out.wav")


def test_enhance_raises_when_input_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/auphonic")
    monkeypatch.setenv("AUPHONIC_API_KEY", "k")
    with pytest.raises(EnhanceError, match="input not found"):
        enhance_with_auphonic(tmp_path / "missing.wav", tmp_path / "out.wav")


def test_enhance_passes_debreath_flag(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/auphonic")
    monkeypatch.setenv("AUPHONIC_API_KEY", "k")
    src = tmp_path / "in.wav"
    src.write_bytes(b"RIFFFAKE")

    captured_cmds: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        download_dir = Path(cmd[cmd.index("--download-dir") + 1])
        out_name_idx = next(i for i, arg in enumerate(cmd) if arg.startswith("format=wav,filename="))
        out_name = cmd[out_name_idx].split("filename=", 1)[1]
        (download_dir / out_name).write_bytes(b"CLEANED")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("subprocess.run", fake_run)
    out = tmp_path / "out.wav"
    enhance_with_auphonic(src, out, debreath_db=18)
    assert out.read_bytes() == b"CLEANED"
    assert "--debreath-amount" in captured_cmds[0]
    assert "18" in captured_cmds[0]


def test_enhance_default_debreath_dB(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/auphonic")
    monkeypatch.setenv("AUPHONIC_API_KEY", "k")
    src = tmp_path / "in.wav"
    src.write_bytes(b"X")

    captured_cmds: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        download_dir = Path(cmd[cmd.index("--download-dir") + 1])
        out_name_idx = next(i for i, arg in enumerate(cmd) if arg.startswith("format=wav,filename="))
        out_name = cmd[out_name_idx].split("filename=", 1)[1]
        (download_dir / out_name).write_bytes(b"CLEAN")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("subprocess.run", fake_run)
    enhance_with_auphonic(src, tmp_path / "out.wav")
    cmd = captured_cmds[0]
    db = cmd[cmd.index("--debreath-amount") + 1]
    assert int(db) == DEFAULT_DEBREATH_DB


def test_enhance_subprocess_failure_wraps_as_enhance_error(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/auphonic")
    monkeypatch.setenv("AUPHONIC_API_KEY", "k")
    src = tmp_path / "in.wav"
    src.write_bytes(b"X")

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=2, cmd=cmd, stderr="auth failed: invalid key"
        )

    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(EnhanceError, match="exited 2"):
        enhance_with_auphonic(src, tmp_path / "out.wav")


def test_enhance_timeout_wraps_as_enhance_error(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/auphonic")
    monkeypatch.setenv("AUPHONIC_API_KEY", "k")
    src = tmp_path / "in.wav"
    src.write_bytes(b"X")

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1.0)

    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(EnhanceError, match="timed out"):
        enhance_with_auphonic(src, tmp_path / "out.wav", timeout=1.0)


def test_enhance_no_output_file_raises(monkeypatch, tmp_path: Path):
    """auphonic exits 0 but doesn't write the expected file."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/auphonic")
    monkeypatch.setenv("AUPHONIC_API_KEY", "k")
    src = tmp_path / "in.wav"
    src.write_bytes(b"X")

    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0),
    )
    with pytest.raises(EnhanceError, match="produced no output"):
        enhance_with_auphonic(src, tmp_path / "out.wav")
