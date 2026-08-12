from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.adapters.model import ModelAdapter, ModelRequest
from src.compilers.config import CompilerSettings
from src.compilers.experience.verify import VerifiedCandidate

_WORD = re.compile(r"\w+", re.UNICODE)

ALL_RELATIONS = (
    "equivalent", "a_subsumes_b", "b_subsumes_a",
    "related_but_distinct", "contradictory", "unrelated",
)


@dataclass(frozen=True)
class AdjudicationResult:
    index_a: int
    index_b: int
    relation: str
    compatible_for_canonicalization: bool
    applicability_conflict: bool
    notes: str
    merges: bool


def _tokens(text: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(text) if word]


def _jaccard(left: list[str], right: list[str]) -> float:
    set_left = set(left)
    set_right = set(right)
    if not set_left and not set_right:
        return 0.0
    return len(set_left & set_right) / len(set_left | set_right)


def _deterministic_adjudication(
    left: VerifiedCandidate,
    right: VerifiedCandidate,
) -> dict[str, Any]:
    overlap = _jaccard(
        _tokens(f"{left.candidate.observed_pattern} {left.candidate.strategy}"),
        _tokens(f"{right.candidate.observed_pattern} {right.candidate.strategy}"),
    )
    if overlap >= 0.8:
        relation = "equivalent"
    elif overlap >= 0.5:
        relation = "related_but_distinct"
    else:
        relation = "unrelated"
    compatible = relation in ("equivalent", "a_subsumes_b", "b_subsumes_a")
    return {
        "relation": relation,
        "compatible_for_canonicalization": compatible,
        "applicability_conflict": False,
        "notes": "deterministic mechanics adjudication",
    }


_ADJUDICATOR_SYSTEM = (
    "You adjudicate whether two candidate rhetorical writing strategies are semantically "
    "compatible for consolidation into a single canonical strategy. You never merge without "
    "an explicit judgment."
)


def _adjudicator_user(left: VerifiedCandidate, right: VerifiedCandidate) -> str:
    return (
        "[TASK: semantic equivalence adjudication]\n"
        "Candidate A observed pattern: {a_pattern}\n"
        "Candidate A strategy: {a_strategy}\n"
        "Candidate A applicable when: {a_when}\n\n"
        "Candidate B observed pattern: {b_pattern}\n"
        "Candidate B strategy: {b_strategy}\n"
        "Candidate B applicable when: {b_when}\n\n"
        "Return a JSON object with keys \"relation\" (one of: equivalent, a_subsumes_b, "
        "b_subsumes_a, related_but_distinct, contradictory, unrelated), "
        "\"compatible_for_canonicalization\" (bool), \"applicability_conflict\" (bool, whether "
        "the two applicability conditions materially conflict), and \"notes\"."
    ).format(
        a_pattern=left.candidate.observed_pattern,
        a_strategy=left.candidate.strategy,
        a_when=left.candidate.applicable_when,
        b_pattern=right.candidate.observed_pattern,
        b_strategy=right.candidate.strategy,
        b_when=right.candidate.applicable_when,
    )


def _model_adjudication(
    left: VerifiedCandidate,
    right: VerifiedCandidate,
    adapter: ModelAdapter,
) -> dict[str, Any]:
    response = adapter.generate(
        ModelRequest(
            system_prompt=_ADJUDICATOR_SYSTEM,
            user_prompt=_adjudicator_user(left, right),
            max_output_tokens=512,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        )
    )
    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Adjudication output was not valid JSON: {exc}") from exc
    required = {
        "relation", "compatible_for_canonicalization", "applicability_conflict", "notes",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"Adjudication output missing keys: {sorted(required - set(payload))}")
    if payload["relation"] not in ALL_RELATIONS:
        raise ValueError(f"Adjudication output invalid relation: {payload['relation']}")
    return {
        "relation": str(payload["relation"]),
        "compatible_for_canonicalization": bool(payload["compatible_for_canonicalization"]),
        "applicability_conflict": bool(payload["applicability_conflict"]),
        "notes": str(payload["notes"]),
    }


def adjudicate_pair(
    left: VerifiedCandidate,
    right: VerifiedCandidate,
    index_a: int,
    index_b: int,
    settings: CompilerSettings,
    adapter: ModelAdapter | None = None,
) -> AdjudicationResult:
    """Semantic equivalence adjudication (spec §8.5)."""
    if adapter is None:
        payload = _deterministic_adjudication(left, right)
    else:
        payload = _model_adjudication(left, right, adapter)
    conflict_blocks = (
        payload["applicability_conflict"] and settings.require_nonconflicting_applicable_when
    )
    merges = payload["compatible_for_canonicalization"] and not conflict_blocks
    return AdjudicationResult(
        index_a=index_a,
        index_b=index_b,
        relation=payload["relation"],
        compatible_for_canonicalization=payload["compatible_for_canonicalization"],
        applicability_conflict=payload["applicability_conflict"],
        notes=payload["notes"],
        merges=merges,
    )
