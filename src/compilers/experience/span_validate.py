from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.compilers.experience.extract import ExtractionCandidate
from src.domain.models import SourcePaper


@dataclass(frozen=True)
class ValidatedCandidate:
    candidate: ExtractionCandidate
    grounding_status: str  # "span_verified" | "rejected"
    rejection_reason: str | None = None


def _reject(reason: str) -> tuple[str, str]:
    return "rejected", reason


def validate_candidate(
    candidate: ExtractionCandidate,
    source_by_id: Mapping[str, SourcePaper],
) -> ValidatedCandidate:
    """Deterministic exact-span validation (spec §8.2). No LLM involved.

    State transition: unverified -> span_verified, or -> rejected on failure.
    """
    source = source_by_id.get(candidate.source_id)
    if source is None:
        return ValidatedCandidate(candidate, *_reject("unknown_source_id"))
    if not candidate.span.strip():
        return ValidatedCandidate(candidate, *_reject("empty_span"))

    location = candidate.location
    if location.paragraph < 1 or location.paragraph > len(source.introduction.paragraphs):
        return ValidatedCandidate(candidate, *_reject("paragraph_out_of_range"))
    paragraph = source.introduction.paragraphs[location.paragraph - 1]
    if location.sentence_start < 1 or location.sentence_end > len(paragraph.sentences):
        return ValidatedCandidate(candidate, *_reject("sentence_coordinates_out_of_range"))
    if location.sentence_start > location.sentence_end:
        return ValidatedCandidate(candidate, *_reject("sentence_range_reversed"))

    first = paragraph.sentences[location.sentence_start - 1]
    last = paragraph.sentences[location.sentence_end - 1]
    window = source.introduction.normalized_text[first.char_start:last.char_end]
    if candidate.span not in window:
        return ValidatedCandidate(candidate, *_reject("span_not_in_referenced_window"))
    if candidate.span not in source.introduction.normalized_text:
        return ValidatedCandidate(candidate, *_reject("span_not_in_normalized_text"))

    return ValidatedCandidate(candidate, "span_verified")


def validate_candidates(
    candidates: tuple[ExtractionCandidate, ...],
    source_by_id: Mapping[str, SourcePaper],
) -> tuple[list[ValidatedCandidate], list[dict[str, Any]]]:
    validated: list[ValidatedCandidate] = []
    trace: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        result = validate_candidate(candidate, source_by_id)
        validated.append(result)
        if result.grounding_status == "rejected":
            trace.append(
                {
                    "candidate_index": index,
                    "source_id": candidate.source_id,
                    "stage": "span_validate",
                    "level": "error",
                    "reason": result.rejection_reason,
                }
            )
    return validated, trace
