from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from src.adapters.model import ModelAdapter, ModelRequest
from src.compilers.config import CompilerSettings
from src.common.structured_output import load_json_object
from src.domain.models import EvidenceLocation, SourcePaper

_WORD = re.compile(r"\w+", re.UNICODE)
_EXTRACTION_MAX_OUTPUT_TOKENS = 16384
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
    format_repair_count: int
    deterministic_fallback_count: int


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
    "directly observable. Do not assign numerical confidence. Do not use a fixed taxonomy. "
    "Return only the requested compact JSON object; do not emit analysis, reasoning, or Markdown."
)

_FORMAT_REPAIR_SYSTEM = (
    "Repair the supplied extraction result into valid JSON without adding, deleting, or "
    "semantically changing candidates. Return only one JSON object with key candidates."
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
        "expected to be useful). Return between 1 and {max_candidates} candidates for a non-empty "
        "Introduction. Keep each descriptive field to one concise sentence. Return only JSON.\n"
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
        payload = load_json_object(raw)
    except (json.JSONDecodeError, ValueError) as exc:
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
    *,
    formal_mode: bool = False,
    run_id: str = "unscoped",
    config_hash: str | None = None,
    data_manifest_hash: str | None = None,
) -> tuple[list[ExtractionCandidate], list[dict[str, Any]], int]:
    candidates: list[ExtractionCandidate] = []
    trace: list[dict[str, Any]] = []
    repair_count = 0
    for source in sources:
        request = ModelRequest(
            system_prompt=_EXTRACTION_SYSTEM,
            user_prompt=_extraction_user(source, settings),
            max_output_tokens=_EXTRACTION_MAX_OUTPUT_TOKENS,
            temperature=None if formal_mode else 0.0,
            top_p=None if formal_mode else 1.0,
            seed=None if formal_mode else 0,
            thinking_enabled=True if formal_mode else None,
            reasoning_effort="high" if formal_mode else None,
            response_format="json_object" if formal_mode else "text",
            run_id=run_id,
            role="experience_extractor",
            run_mode="formal" if formal_mode else "mechanics",
            config_hash=config_hash,
            data_manifest_hash=data_manifest_hash,
        )
        response = adapter.generate(request)
        parsed, error = _parse_extraction_output(response.text, source.source_id)
        call_metadata = dict(response.metadata)
        if error:
            repair_count += 1
            repair = adapter.generate(
                ModelRequest(
                    system_prompt=_FORMAT_REPAIR_SYSTEM,
                    user_prompt=(
                        f"Source id: {source.source_id}\n"
                        f"Parser error: {error}\n"
                        f"Invalid result:\n{response.text}"
                    ),
                    max_output_tokens=_EXTRACTION_MAX_OUTPUT_TOKENS,
                    temperature=None if formal_mode else 0.0,
                    top_p=None if formal_mode else 1.0,
                    seed=None if formal_mode else 0,
                    thinking_enabled=True if formal_mode else None,
                    reasoning_effort="high" if formal_mode else None,
                    response_format="json_object" if formal_mode else "text",
                    run_id=run_id,
                    role="experience_extractor_format_repair",
                    run_mode="formal" if formal_mode else "mechanics",
                    config_hash=config_hash,
                    data_manifest_hash=data_manifest_hash,
                )
            )
            parsed, repair_error = _parse_extraction_output(repair.text, source.source_id)
            trace.append(
                {
                    "source_id": source.source_id,
                    "stage": "extract_format_repair",
                    "level": "error" if repair_error else "info",
                    "initial_error": error,
                    "repair_error": repair_error,
                    "provider_call": dict(repair.metadata),
                }
            )
            error = repair_error
            call_metadata = dict(repair.metadata)
        if error:
            trace.append(
                {
                    "source_id": source.source_id,
                    "stage": "extract",
                    "level": "error",
                    "message": error,
                    "provider_call": call_metadata,
                }
            )
        else:
            candidates.extend(parsed)
            if formal_mode:
                trace.append(
                    {
                        "source_id": source.source_id,
                        "stage": "extract",
                        "level": "info",
                        "candidate_count": len(parsed),
                        "provider_call": call_metadata,
                    }
                )
    return candidates, trace, repair_count


def extract_candidates(
    sources: Iterable[SourcePaper],
    settings: CompilerSettings,
    adapter: ModelAdapter | None = None,
    *,
    formal_mode: bool = False,
    run_id: str = "unscoped",
    config_hash: str | None = None,
    data_manifest_hash: str | None = None,
) -> ExtractionOutcome:
    """Single-pass open atomic extraction (spec §8.1)."""
    if formal_mode and adapter is None:
        raise ValueError("Formal extraction requires a real model adapter")
    if adapter is None:
        candidates, trace = _extract_deterministic(sources, settings)
        mode = "deterministic"
        repair_count = 0
        fallback_count = 0
    else:
        candidates, trace, repair_count = _extract_with_model(
            sources,
            adapter,
            settings,
            formal_mode=formal_mode,
            run_id=run_id,
            config_hash=config_hash,
            data_manifest_hash=data_manifest_hash,
        )
        mode = f"model:{adapter.model_name}"
        fallback_count = 0
    return ExtractionOutcome(
        candidates=tuple(candidates),
        trace=tuple(trace),
        adapter_mode=mode,
        format_repair_count=repair_count,
        deterministic_fallback_count=fallback_count,
    )
