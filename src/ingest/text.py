from __future__ import annotations

import re
import unicodedata

from src.domain.models import Introduction, Paragraph, Sentence


_BLANK_LINE = re.compile(r"\n\s*\n+")
_SENTENCE_END = frozenset(".!?。！？")
_CLOSERS = frozenset("\"'”’)]}）】")


def normalize_plain_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [" ".join(part.split()) for part in _BLANK_LINE.split(normalized) if part.strip()]
    return "\n\n".join(paragraphs)


def _sentence_ranges(paragraph: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    index = 0
    while index < len(paragraph):
        if paragraph[index] in _SENTENCE_END:
            end = index + 1
            while end < len(paragraph) and paragraph[end] in _CLOSERS:
                end += 1
            if end == len(paragraph) or paragraph[end].isspace():
                if paragraph[start:end].strip():
                    ranges.append((start, end))
                start = end
                while start < len(paragraph) and paragraph[start].isspace():
                    start += 1
                index = start
                continue
        index += 1
    if start < len(paragraph) and paragraph[start:].strip():
        ranges.append((start, len(paragraph)))
    return ranges


def normalize_introduction(text: str) -> Introduction:
    normalized = normalize_plain_text(text)
    if not normalized:
        raise ValueError("Introduction cannot be empty")
    paragraph_texts = normalized.split("\n\n")
    paragraphs: list[Paragraph] = []
    paragraph_start = 0
    for paragraph_id, paragraph_text in enumerate(paragraph_texts, start=1):
        sentences = tuple(
            Sentence(
                sentence_id=sentence_id,
                text=paragraph_text[local_start:local_end],
                char_start=paragraph_start + local_start,
                char_end=paragraph_start + local_end,
            )
            for sentence_id, (local_start, local_end) in enumerate(
                _sentence_ranges(paragraph_text), start=1
            )
        )
        paragraphs.append(Paragraph(paragraph_id=paragraph_id, sentences=sentences))
        paragraph_start += len(paragraph_text) + 2
    return Introduction(normalized_text=normalized, paragraphs=tuple(paragraphs))
