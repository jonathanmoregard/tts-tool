"""Normalize -> sentence-segment -> greedy-pack into TTS-sized chunks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TARGET_CHARS = 2000
SOFT_CLOSE_CHARS = 1200


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int


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


def _render(items: list[tuple[str, int]]) -> str:
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


def pack(
    sentences: list[tuple[str, int]],
    target: int = TARGET_CHARS,
    soft: int = SOFT_CLOSE_CHARS,
) -> list[Chunk]:
    """Pack (sentence, paragraph_idx) pairs into <=target-char chunks.

    Hard: never split a sentence. Soft: when crossing a paragraph
    boundary and the current chunk is already >= soft, close the chunk.
    A single sentence longer than target is emitted as its own chunk.
    """
    chunks: list[Chunk] = []
    buf: list[tuple[str, int]] = []

    def emit() -> None:
        nonlocal buf
        if buf:
            chunks.append(Chunk(text=_render(buf), index=len(chunks)))
            buf = []

    for sentence, p_idx in sentences:
        if buf and buf[-1][1] != p_idx and len(_render(buf)) >= soft:
            emit()
        if buf and len(_render(buf + [(sentence, p_idx)])) > target:
            emit()
        buf.append((sentence, p_idx))

    emit()
    return chunks


def chunk_text(text: str, *, nlp: Any | None = None) -> list[Chunk]:
    text = normalize(text)
    if not text:
        return []
    sentences: list[tuple[str, int]] = []
    for p_idx, paragraph in enumerate(split_paragraphs(text)):
        for sentence in segment_paragraph(paragraph, nlp=nlp):
            sentences.append((sentence, p_idx))
    return pack(sentences)
