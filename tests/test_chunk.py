"""Tests for tts-tool's pause-annotation-driven chunker.

Pause-shaped tags are parsed out of input text; the chunker emits
`Chunk(text, index, silence_after)` where silence_after carries the
tag's duration, and the segment text fed to Fish is FREE of pause
tags. Non-pause tags (`[emphasis]`, `[thoughtfully]`, ...) are
preserved in the segment text.
"""
from unittest.mock import MagicMock

import pytest

from tts_tool.chunk import (
    Chunk,
    chunk_text,
    normalize,
    parse_pause_duration,
    split_paragraphs,
)


def _fake_nlp():
    """Stub spaCy nlp that segments naively on `. ` / `! ` / `? `
    boundaries so chunk tests don't require the spaCy model."""
    import re as _re

    def _make_doc(text):
        # split on sentence terminators followed by whitespace
        parts = _re.split(r"(?<=[.!?])\s+", text.strip())
        return MagicMock(sents=[MagicMock(text=p) for p in parts if p])

    return _make_doc


def test_normalize_strips_and_collapses_crlf():
    assert normalize("  hello\r\nworld  ") == "hello\nworld"
    assert normalize("\r\nfoo\r\n") == "foo"


def test_split_paragraphs_drops_empty():
    assert split_paragraphs("a\n\nb\n\n\n\nc") == ["a", "b", "c"]
    assert split_paragraphs("") == []


# ---------- parse_pause_duration ----------

def test_parse_pause_duration_short():
    assert parse_pause_duration("short pause") == 0.3


def test_parse_pause_duration_long():
    assert parse_pause_duration("long pause") == 1.0


def test_parse_pause_duration_very_long():
    assert parse_pause_duration("very long pause") == 1.5


def test_parse_pause_duration_bare():
    assert parse_pause_duration("pause") == 0.5


def test_parse_pause_duration_digit_seconds():
    assert parse_pause_duration("3 second pause") == 3.0
    assert parse_pause_duration("pause for 2 seconds") == 2.0
    assert parse_pause_duration("1.5 second pause") == 1.5


def test_parse_pause_duration_word_seconds():
    assert parse_pause_duration("pause for two seconds") == 2.0
    assert parse_pause_duration("three second pause") == 3.0


def test_parse_pause_duration_case_insensitive():
    assert parse_pause_duration("VERY LONG PAUSE") == 1.5


# ---------- chunk_text ----------

def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_no_pause_tags_single_chunk():
    out = chunk_text("Hello world.", nlp=_fake_nlp())
    assert len(out) == 1
    assert out[0].text == "Hello world."
    assert out[0].silence_after == 0.0


def test_chunk_text_strips_pause_tag_into_silence():
    out = chunk_text(
        "First sentence. [very long pause] Second sentence.",
        nlp=_fake_nlp(),
    )
    assert len(out) == 2
    assert "pause" not in out[0].text.lower()
    assert out[0].silence_after == 1.5
    assert out[1].silence_after == 0.0  # last chunk = no trailing silence


def test_chunk_text_multiple_pause_tags():
    out = chunk_text(
        "A. [short pause] B. [long pause] C.",
        nlp=_fake_nlp(),
    )
    assert len(out) == 3
    assert out[0].silence_after == 0.3
    assert out[1].silence_after == 1.0
    assert out[2].silence_after == 0.0


def test_chunk_text_preserves_non_pause_tags():
    out = chunk_text(
        "Body. [emphasis] Stressed word. [very long pause] Next.",
        nlp=_fake_nlp(),
    )
    # [emphasis] stays in segment text; [very long pause] becomes silence
    assert "[emphasis]" in out[0].text
    assert "pause" not in out[0].text.lower()
    assert out[0].silence_after == 1.5


def test_chunk_text_indexes_sequential():
    out = chunk_text("A. [pause] B. [pause] C.", nlp=_fake_nlp())
    for i, c in enumerate(out):
        assert c.index == i


def test_chunk_text_pause_at_start_dropped():
    """A pause at the very start of input has no audio before it to
    end with silence — drop it rather than emitting a leading-silence
    chunk."""
    out = chunk_text("[very long pause] Hello.", nlp=_fake_nlp())
    assert len(out) == 1
    assert out[0].text == "Hello."
    assert out[0].silence_after == 0.0


def test_chunk_text_pause_at_end_attaches_to_previous():
    """A pause at end of input has no successor; the duration attaches
    to the previous chunk's silence_after but the LAST-chunk rule in
    stitch suppresses trailing silence anyway."""
    out = chunk_text("Hello. [very long pause]", nlp=_fake_nlp())
    assert len(out) == 1
    assert out[0].text == "Hello."
    # Duration was parsed; stitcher suppresses trailing silence on last chunk
    assert out[0].silence_after == 1.5


def test_chunk_text_adjacent_pause_tags_accumulate():
    """Two pause tags with no speech between -> durations sum onto the
    previous chunk's silence_after."""
    out = chunk_text(
        "A. [short pause] [long pause] B.",
        nlp=_fake_nlp(),
    )
    assert len(out) == 2
    assert out[0].silence_after == pytest.approx(0.3 + 1.0)


def test_chunk_text_pause_tag_with_digit_duration():
    out = chunk_text(
        "Body. [3 second pause] More.",
        nlp=_fake_nlp(),
    )
    assert len(out) == 2
    assert out[0].silence_after == 3.0


def test_chunk_text_long_segment_sub_chunked():
    """Segment between pause tags exceeds target -> sentence-pack
    split; only the LAST sub-chunk of the segment carries the silence."""
    long_segment = ". ".join(f"Sentence {i}" for i in range(100)) + "."
    out = chunk_text(
        f"{long_segment} [very long pause] Tail.",
        nlp=_fake_nlp(),
        target=200,
        soft=150,
    )
    # Multiple sub-chunks for the long segment; only the last has the
    # silence_after = 1.5 (the others have 0)
    pre_tail = out[:-1]
    assert len(pre_tail) > 1
    for c in pre_tail[:-1]:
        assert c.silence_after == 0.0
    assert pre_tail[-1].silence_after == 1.5
    # Last chunk is the tail, no trailing silence
    assert out[-1].text == "Tail."
    assert out[-1].silence_after == 0.0
