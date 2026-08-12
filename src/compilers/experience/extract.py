from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from src.adapters.model import ModelAdapter, ModelRequest
from src.compilers.config import CompilerSettings
from src.domain.models import EvidenceLocation, SourcePaper

_WORD = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "with", "by",
        "from", "that", "this", "these", "those", "we", "our", "as", "is", "are",
        "be", "was", "were", "it", "its", "their", "they", "at", "into", "across",
        "such", "more", "than", "then", "can", "may", "must", "should", "not",
        "only", "also", "under", "over", "each", "between", "while", "after",
        "before", "use", "using", "uses", "used", "provide", "provides", "based",
    }
)


@dataclass(frozen=True)
class ExtractionCandidate:
    """One atomic rhetorical candidate emitted by the single-pass open extractor."""

    source_id: str
    location: EvidenceLocation
    span: str
    observed_pattern: str
    strategy: str
    applicable_when: str


@dataclass(frozen=True)
class ExtractionOutcome:
    candidates: tuple[ExtractionCandidate, ...]
    trace: tuple[dict[str, Any], ...]
    adapter_mode: str


def _derive_topic(text: str) -> str:
    words = [word.lower() for word in _WORD.findall(text) if word.lower() not in _STOPWORDS]
    topic = " ".join(words[:4]).strip()
    return topic or text[:48].strip()


def _deterministic_candidate(source_id: str, paragraph_id: int, sentence: Any) -> ExtractionCandidate:
    topic = _derive_topic(sentence.text)
    return ExtractionCandidate(
        source_id=source_id,
        location=EvidenceLocation(
            section="Introduction",
            paragraph=paragraph_id,
            sentence_start=sentence.sentence_id,
            sentence_end=sentence.sentence_id,
        ),
        span=sentence.text,
        # observed_pattern stays source-local and topic-anchored so deterministic
        # adjudication stays meaningful; strategy/applicable_when are generalized so a
        # Writer does not copy source-domain vocabulary from them.
        observed_pattern=(
            f"The cited span states a claim about {topic!r} and places it in the surrounding argument."
        ),
        strategy=(
            "State the core claim directly in the opening, then connect it to the broader "
            "problem in the following sentences."
        ),
        applicable_when=(
            "When an Introduction needs to present a claim that establishes the paper's scope."
        ),
    )


def _extract_deterministic(
    sources: Iterable[SourcePaper],
    settings: CompilerSettings,
) -> tuple[list[ExtractionCandidate], list[dict[str, Any]]]:
    candidates: list[ExtractionCandidate] = []
    for source in sources:
        count = 0
        for paragraph in source.introduction.paragraphs:
            for sentence in paragraph.sentences:
                if count >= settings.max_candidates_per_source:
                    break
                candidates.append(_deterministic_candidate(source.source_id, paragraph.paragraph_id, sentence))
                count += 1
    return candidates, []


_EXTRACTION_SYSTEM = (
    "You are a scientific-writing observer. Extract atomic rhetorical candidates from the "
    "provided scientific Introduction text. One candidate corresponds to one observable "
    "rhetorical decision or pattern in the text. Do not infer author intention unless it is "
    "directly observable. Do not assign numerical confidence. Do not use a fixed taxonomy."
)


def _extraction_user(source: SourcePaper, settings: CompilerSettings) -> str:
    return (
        "[TASK: single-pass open atomic extraction]\n"
        "Source id: {source_id}\n"
        "Introduction (normalized):\n{text}\n\n"
        "Emit a JSON object with a single key \"candidates\" containing a list of objects. "
        "Each object has: "
        "\"location\" as {{\"paragraph\": int, \"sentence_start\": int, \"sentence_end\": int}} "
        "(1-based, \"section\" is always Introduction), "
        "\"span\" as the exact verbatim source substring covering those sentences, "
        "\"observed_pattern\" (a source-local descriptive account of the rhetorical action "
        "observable in the span, without asserting author intention), "
        "\"strategy\" (an actionable generalized writing strategy inferred from the observed "
        "pattern), and \"applicable_when\" (the writing conditions under which the strategy is "
        "expected to be useful). Return at most {max_candidates} candidates.\n"
    ).format(
        source_id=source.source_id,
        text=source.introduction.normalized_text,
        max_candidates=settings.max_candidates_per_source,
    )


def _parse_extraction_output(
    raw: str,
    source_id: str,
) -> tuple[list[ExtractionCandidate], str | None]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [], f"invalid_json: {exc}"
    if not isinstance(payload, dict) or "candidates" not in payload:
        return [], "missing_candidates_key"
    items = payload["candidates"]
    if not isinstance(items, list):
        return [], "candidates_not_list"

    candidates: list[ExtractionCandidate] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            return candidates[:0], f"candidate_{index}_not_object"
        try:
            location = item["location"]
            candidates.append(
                ExtractionCandidate(
                    source_id=source_id,
                    location=EvidenceLocation(
                        section="Introduction",
                        paragraph=int(location["paragraph"]),
                        sentence_start=int(location["sentence_start"]),
                        sentence_end=int(location["sentence_end"]),
                    ),
                    span=str(item["span"]),
                    observed_pattern=str(item["observed_pattern"]),
                    strategy=str(item["strategy"]),
                    applicable_when=str(item["applicable_when"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            return candidates, f"candidate_{index}_malformed: {exc}"
    return candidates, None


def _extract_with_model(
    sources: Iterable[SourcePaper],
    adapter: ModelAdapter,
    settings: CompilerSettings,
) -> tuple[list[ExtractionCandidate], list[dict[str, Any]]]:
    candidates: list[ExtractionCandidate] = []
    trace: list[dict[str, Any]] = []
    for source in sources:
        response = adapter.generate(
            ModelRequest(
                system_prompt=_EXTRACTION_SYSTEM,
                user_prompt=_extraction_user(source, settings),
                max_output_tokens=4096,
                temperature=0.0,
                top_p=1.0,
                seed=0,
            )
        )
        parsed, error = _parse_extraction_output(response.text, source.source_id)
        if error:
            trace.append(
                {
                    "source_id": source.source_id,
                    "stage": "extract",
                    "level": "error",
                    "message": error,
                }
            )
        else:
            candidates.extend(parsed)
    return candidates, trace


def extract_candidates(
    sources: Iterable[SourcePaper],
    settings: CompilerSettings,
    adapter: ModelAdapter | None = None,
) -> ExtractionOutcome:
    """Single-pass open atomic extraction (spec §8.1)."""
    if adapter is None:
        candidates, trace = _extract_deterministic(sources, settings)
        mode = "deterministic"
    else:
        candidates, trace = _extract_with_model(sources, adapter, settings)
        mode = f"model:{adapter.model_name}"
    return ExtractionOutcome(
        candidates=tuple(candidates),
        trace=tuple(trace),
        adapter_mode=mode,
    )
