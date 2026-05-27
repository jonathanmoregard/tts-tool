from pathlib import Path

from tts_tool import cache


def test_key_for_is_deterministic():
    k1 = cache.key_for("hello", "s1", "voice-abc", 1.0)
    k2 = cache.key_for("hello", "s1", "voice-abc", 1.0)
    assert k1 == k2
    assert len(k1) == 64


def test_key_for_changes_with_any_field():
    base = cache.key_for("hello", "s1", "v1", 1.0)
    assert base != cache.key_for("HELLO", "s1", "v1", 1.0)
    assert base != cache.key_for("hello", "s2-pro", "v1", 1.0)
    assert base != cache.key_for("hello", "s1", "v2", 1.0)
    assert base != cache.key_for("hello", "s1", "v1", 1.1)


def test_key_for_voice_none_vs_empty_collide_by_design():
    assert cache.key_for("x", "s1", None, 1.0) == cache.key_for("x", "s1", "", 1.0)


def test_put_then_get_roundtrip(tmp_path: Path):
    cdir = cache.chunks_dir(tmp_path)
    key = cache.key_for("hi", "s1", None, 1.0)
    assert cache.get(cdir, key) is None
    cache.put(cdir, key, b"\xff\xfb\x90\x00\x00\x00")
    assert cache.get(cdir, key) == b"\xff\xfb\x90\x00\x00\x00"


def test_put_is_atomic_no_tmp_left(tmp_path: Path):
    cdir = cache.chunks_dir(tmp_path)
    key = cache.key_for("hi", "s1", None, 1.0)
    cache.put(cdir, key, b"data")
    leftover = list(cdir.glob("*.tmp"))
    assert leftover == []


def test_chunks_dir_creates_path(tmp_path: Path):
    target = tmp_path / "nested" / "deep"
    cdir = cache.chunks_dir(target)
    assert cdir.exists()
    assert cdir.name == "chunks"


def test_key_for_prime_tail_changes_key():
    base = cache.key_for("hi", "s1", "v", 1.0)
    with_tail = cache.key_for("hi", "s1", "v", 1.0, prime_tail=b"TAIL1")
    other_tail = cache.key_for("hi", "s1", "v", 1.0, prime_tail=b"TAIL2")
    assert base != with_tail
    assert with_tail != other_tail


def test_key_for_prime_tail_none_matches_unset():
    assert (
        cache.key_for("hi", "s1", "v", 1.0)
        == cache.key_for("hi", "s1", "v", 1.0, prime_tail=None)
    )
