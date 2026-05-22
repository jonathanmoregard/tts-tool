"""Pause-annotation-driven chunker (Fish Audio Story-Studio pattern).

Input text may contain Fish s2-pro prosody tags. PAUSE-SHAPED tags
(`[short pause]`, `[long pause]`, `[very long pause]`, `[pause]`,
`[N second pause]`, `[pause for two seconds]`, etc.) are STRUCTURAL:
they are application-layer instructions to insert a silence of a
specified duration.

Empirically (probe 2026-05-22, 26 author-side variants) Fish s2-pro
has a HARD CEILING of ~1.0s on any silence requested via input syntax —
documented + confirmed by Fish maintainers on github.com/fishaudio/
fish-speech#896: "Currently our open source model can't achieve this".
Tags are tokenized as ordinary text and conditioned on a Slow-AR
transformer whose training distribution caps intra-utterance pauses
at ~1.5s. There is no hidden TTSConfig field, no SSML, no FlushEvent,
no voice ID that breaks this. Fish's own first-party answer for
multi-second beats is Story Studio: per-block synthesis + concat
with explicit silence (gap bubbles, 0.2-5s editable).

This chunker implements Story Studio's pattern client-side: parse
pause-shaped tags out of input, split text into Fish-callable
SEGMENTS at each tag, emit Chunk objects carrying a `silence_after`
duration. The stitcher concatenates chunks with silence-MP3s of that
duration between them. The annotation drives both the chunking AND
the pause length — single source of truth at the application layer.

Non-pause tags (`[emphasis]`, `[thoughtfully]`, `[reading aloud]`,
`[back to narration]`, etc.) remain inline in the segment text for
Fish to render — those are PROSODIC tags Fish handles correctly
mid-text (within the ~1s ceiling).

If a segment between pause tags is too long for Fish (>= TARGET_CHARS),
it is sentence-pack-split into multiple chunks; consecutive chunks of
the same logical segment carry `silence_after=0` between them.

Cost: paragraph-dense prose produces one Fish call per paragraph
boundary. Mitigated by parallel synthesis in cli.py — N chunks can
synth concurrently against Fish's rate limit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

TARGET_CHARS = 2000
SOFT_CLOSE_CHARS = 1200

# Pause-shaped tag detection. Matches bracket bodies containing the
# word "pause". Captures the body for duration parsing.
_PAUSE_TAG_RE = re.compile(r"\[(?P<body>[^\[\]\n]*\bpause\b[^\[\]\n]*)\]")


def parse_pause_duration(body: str) -> float:
    """Map a pause-tag body to a silence duration in seconds.

    Heuristic order (longest match wins):
      [N second pause] / [pause for N seconds] / [pause for two seconds]
      [very long pause]   -> 1.5
      [long pause]        -> 1.0
      [short pause]       -> 0.3
      [pause] / bare      -> 0.5
    """
    low = body.lower().strip()
    # Digit form: "3 second pause", "pause for 2 seconds"
    m = re.search(r"(\d+(?:\.\d+)?)\s*second", low)
    if m:
        return float(m.group(1))
    # Word-number form (common cases only — Sonnet emits these)
    word_to_n = {"one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0}
    m = re.search(r"\b(one|two|three|four|five)\b.*\bsecond", low)
    if m:
        return word_to_n[m.group(1)]
    if "very long" in low:
        return 1.5
    if "long" in low:
        return 1.0
    if "short" in low:
        return 0.3
    return 0.5  # bare [pause] or unrecognized variant


@dataclass(frozen=True)
class Chunk:
    text: str            # text sent to Fish (no pause tags)
    index: int
    silence_after: float = 0.0  # seconds of silent MP3 to append after this chunk


def normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


_NLP: Any | None = None


def _get_nlp() -> Any:
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load(
            "en_core_web_sm",
            exclude=["ner", "lemmatizer", "attribute_ruler", "tagger"],
        )
    return _NLP


def segment_paragraph(paragraph: str, *, nlp: Any | None = None) -> list[str]:
    nlp = nlp or _get_nlp()
    return [s.text.strip() for s in nlp(paragraph).sents if s.text.strip()]


# Internal: greedy sentence-pack ONE segment of text (no pause tags
# inside) into <= TARGET_CHARS chunks. Returns list of plain strings.
def _pack_segment(
    text: str,
    *,
    target: int,
    soft: int,
    nlp: Any | None,
) -> list[str]:
    if not text.strip():
        return []
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []
    # Flatten to (sentence, p_idx) tuples
    sentences: list[tuple[str, int]] = []
    for p_idx, para in enumerate(paragraphs):
        for sent in segment_paragraph(para, nlp=nlp):
            sentences.append((sent, p_idx))
    if not sentences:
        return []

    def render(items: list[tuple[str, int]]) -> str:
        parts: list[str] = []
        last_p: int | None = None
        for sentence, p_idx in items:
            if last_p is None:
                parts.append(sentence)
            elif p_idx != last_p:
                parts.append("\n\n")
                parts.append(sentence)
            else:
                parts.append(" ")
                parts.append(sentence)
            last_p = p_idx
        return "".join(parts)

    out: list[str] = []
    buf: list[tuple[str, int]] = []
    for sentence, p_idx in sentences:
        if buf and buf[-1][1] != p_idx and len(render(buf)) >= soft:
            out.append(render(buf))
            buf = []
        if buf and len(render(buf + [(sentence, p_idx)])) > target:
            out.append(render(buf))
            buf = []
        buf.append((sentence, p_idx))
    if buf:
        out.append(render(buf))
    return out


def chunk_text(
    text: str,
    *,
    nlp: Any | None = None,
    target: int = TARGET_CHARS,
    soft: int = SOFT_CLOSE_CHARS,
) -> list[Chunk]:
    """Parse pause-shaped tags out, split text at each, sentence-pack
    each between-tag segment, emit Chunks with silence_after duration.
    """
    text = normalize(text)
    if not text:
        return []

    # Find every pause-shaped tag's span + parsed duration
    matches = list(_PAUSE_TAG_RE.finditer(text))
    if not matches:
        # No pause annotations -> single-segment behaviour, no silences
        sub_chunks = _pack_segment(text, target=target, soft=soft, nlp=nlp)
        return [
            Chunk(text=t, index=i, silence_after=0.0)
            for i, t in enumerate(sub_chunks)
        ]

    chunks: list[Chunk] = []
    cursor = 0
    for m in matches:
        segment = text[cursor:m.start()]
        cursor = m.end()
        sub_chunks = _pack_segment(segment, target=target, soft=soft, nlp=nlp)
        if not sub_chunks:
            # Empty segment between adjacent tags (or pause at start).
            # We can't emit a Fish call with no text, but we still want
            # the pause to register. Attach the duration to the PREVIOUS
            # chunk's silence_after; if there's no previous chunk yet
            # (pause at very start of input), drop it — start-of-output
            # silence is meaningless anyway.
            if chunks:
                prev = chunks[-1]
                chunks[-1] = Chunk(
                    text=prev.text,
                    index=prev.index,
                    silence_after=prev.silence_after + parse_pause_duration(m.group("body")),
                )
            continue
        for i, t in enumerate(sub_chunks):
            silence = (
                parse_pause_duration(m.group("body"))
                if i == len(sub_chunks) - 1
                else 0.0
            )
            chunks.append(
                Chunk(text=t, index=len(chunks), silence_after=silence)
            )

    # Tail segment after the last pause tag
    tail = text[cursor:]
    sub_chunks = _pack_segment(tail, target=target, soft=soft, nlp=nlp)
    for t in sub_chunks:
        chunks.append(Chunk(text=t, index=len(chunks), silence_after=0.0))

    return chunks
