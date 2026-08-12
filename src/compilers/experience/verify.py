from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.adapters.model import ModelAdapter, ModelRequest
from src.compilers.config import CompilerSettings
from src.compilers.experience.extract import ExtractionCandidate

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class VerifiedCandidate:
    candidate: ExtractionCandidate
    verifier_result: dict[str, Any]
    grounding_status: str  # "support_verified" | "rejected"
    rejection_reason: str | None = None


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def _significant(text: str) -> list[str]:
    lowered = text.lower()
    return [word for word in _words(text) if word.lower() not in _DETERMINISTIC_STOP]


_DETERMINISTIC_STOP = frozenset(
    {
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "with", "by",
        "from", "that", "this", "these", "those", "we", "our", "as", "is", "are",
        "be", "was", "were", "it", "its", "their", "they", "at", "into", "not",
    }
)


def _deterministic_verdict(candidate: ExtractionCandidate) -> dict[str, str]:
    """Content-based deterministic verdict used only for mechanics verification."""
    span_words = _significant(candidate.span)
    strategy_words = _words(candidate.strategy)
    observation = (
        "supported"
        if len(span_words) >= 3
        else ("partial" if span_words else "unsupported")
    )
    generalization = (
        "reasonable"
        if len(strategy_words) >= 12
        else ("overgeneralized" if len(strategy_words) >= 6 else "unsupported")
    )
    return {
        "observation_support": observation,
        "strategy_generalization": generalization,
        "notes": "deterministic mechanics verdict",
    }


_VERIFIER_SYSTEM = (
    "You are a skeptical blind support verifier. You judge whether a source span supports an "
    "observed rhetorical pattern, and whether a proposed writing strategy is a reasonable, "
    "bounded generalization from that observation. You do not see any extractor reasoning."
)


def _verifier_user(candidate: ExtractionCandidate) -> str:
    return (
        "[TASK: blind support verification]\n"
        "Evidence span: {span}\n"
        "Observed pattern: {observed_pattern}\n"
        "Strategy: {strategy}\n"
        "Applicable when: {applicable_when}\n\n"
        "Return a JSON object with keys \"observation_support\" "
        "(\"supported\" | \"partial\" | \"unsupported\"), \"strategy_generalization\" "
        "(\"reasonable\" | \"overgeneralized\" | \"unsupported\"), and \"notes\" (a short "
        "evidence-based explanation)."
    ).format(
        span=candidate.span,
        observed_pattern=candidate.observed_pattern,
        strategy=candidate.strategy,
        applicable_when=candidate.applicable_when,
    )


def _model_verdict(candidate: ExtractionCandidate, adapter: ModelAdapter) -> dict[str, str]:
    response = adapter.generate(
        ModelRequest(
            system_prompt=_VERIFIER_SYSTEM,
            user_prompt=_verifier_user(candidate),
            max_output_tokens=512,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        )
    )
    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Verifier output was not valid JSON: {exc}") from exc
    required = {"observation_support", "strategy_generalization", "notes"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"Verifier output missing keys: {sorted(required - set(payload))}")
    for key in ("observation_support", "strategy_generalization"):
        if payload[key] not in (
            ("supported", "partial", "unsupported")
            if key == "observation_support"
            else ("reasonable", "overgeneralized", "unsupported")
        ):
            raise ValueError(f"Verifier output invalid {key}: {payload[key]}")
    return {key: str(payload[key]) for key in required}


def _admit(result: dict[str, Any], settings: CompilerSettings) -> tuple[str, str | None]:
    admitted = (
        result.get("observation_support") in settings.admission_observation_support
        and result.get("strategy_generalization")
        in settings.admission_strategy_generalization
    )
    if admitted:
        return "support_verified", None
    return "rejected", "verifier_admission_failed"


def verify_candidate(
    candidate: ExtractionCandidate,
    settings: CompilerSettings,
    adapter: ModelAdapter | None = None,
) -> VerifiedCandidate:
    """Blind support verifier (spec §8.3)."""
    if adapter is None:
        result = _deterministic_verdict(candidate)
    else:
        result = _model_verdict(candidate, adapter)
    status, reason = _admit(result, settings)
    return VerifiedCandidate(
        candidate=candidate,
        verifier_result=result,
        grounding_status=status,
        rejection_reason=reason,
    )
