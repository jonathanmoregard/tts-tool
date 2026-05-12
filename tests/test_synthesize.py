from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fishaudio import RateLimitError, ServerError

from tts_tool.synthesize import MissingAPIKey, read_api_key, synthesize_chunk


def test_read_api_key_from_env(monkeypatch):
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "abc123")
    monkeypatch.delenv("FISH_AUDIO_API_KEY_FILE", raising=False)
    assert read_api_key() == "abc123"


def test_read_api_key_from_file_overrides_env(monkeypatch, tmp_path: Path):
    p = tmp_path / "key"
    p.write_text("from-file\n")
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "from-env")
    monkeypatch.setenv("FISH_AUDIO_API_KEY_FILE", str(p))
    assert read_api_key() == "from-file"


def test_read_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("FISH_AUDIO_API_KEY", raising=False)
    monkeypatch.delenv("FISH_AUDIO_API_KEY_FILE", raising=False)
    with pytest.raises(MissingAPIKey):
        read_api_key()


def test_synthesize_chunk_success_passes_kwargs():
    client = MagicMock()
    client.tts.convert.return_value = b"audio-bytes"
    out = synthesize_chunk(
        client, "hello", model="s1", voice_id="vid", speed=1.1, sleep=lambda _: None
    )
    assert out == b"audio-bytes"
    client.tts.convert.assert_called_once_with(
        text="hello", reference_id="vid", format="mp3", speed=1.1, model="s1"
    )


def test_synthesize_chunk_retries_on_server_error():
    client = MagicMock()
    err = ServerError(500, "boom")
    client.tts.convert.side_effect = [err, err, b"ok"]
    sleeps: list[float] = []
    out = synthesize_chunk(
        client, "hi", max_retries=3, sleep=sleeps.append
    )
    assert out == b"ok"
    assert client.tts.convert.call_count == 3
    assert sleeps == [2.0, 4.0]


def test_synthesize_chunk_retries_on_rate_limit():
    client = MagicMock()
    client.tts.convert.side_effect = [RateLimitError(429, "slow down"), b"ok"]
    out = synthesize_chunk(client, "hi", max_retries=3, sleep=lambda _: None)
    assert out == b"ok"


def test_synthesize_chunk_gives_up_and_reraises_last_error():
    client = MagicMock()
    err = ServerError(503, "no")
    client.tts.convert.side_effect = [err, err, err, err]
    with pytest.raises(ServerError):
        synthesize_chunk(client, "hi", max_retries=3, sleep=lambda _: None)


def test_synthesize_chunk_non_retriable_propagates():
    client = MagicMock()
    client.tts.convert.side_effect = ValueError("bad input")
    with pytest.raises(ValueError):
        synthesize_chunk(client, "hi", sleep=lambda _: None)
