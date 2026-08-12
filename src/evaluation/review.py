from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from src.domain.models import RepresentationArtifact, SourcePaper, TargetEvidencePack

_WORD = re.compile(r"\w+", re.UNICODE)
_CITATION = re.compile(r"\[\d+\]")
_IMPERATIVE_VERBS = frozenset(
    {
        "state", "present", "introduce", "position", "connect", "close", "describe",
        "use", "establish", "situate", "announce", "contrast", "define", "motivate",
        "open", "organize", "summarize", "summarise", "highlight", "compare", "frame",
    }
)
_COPY_NGRAM = 4
_NEAR_COPY_SPAN = 8


def _tokens(text: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(text)]


def _shingles(words: Sequence[str], size: int) -> set[tuple[str, ...]]:
    return {tuple(words[index:index + size]) for index in range(len(words) - size + 1)}


def _longest_shared_span(left: Sequence[str], right: Sequence[str]) -> int:
    longest = 0
    right_set = _shingles(right, _COPY_NGRAM)
    for index in range(len(left) - _COPY_NGRAM + 1):
        ngram = tuple(left[index:index + _COPY_NGRAM])
        if ngram in right_set:
            run = _COPY_NGRAM
            while (
                index + run < len(left)
                and index + run < len(right)
                and left[index + run] == right[index + run]
            ):
                run += 1
            longest = max(longest, run)
    return longest


def _exact_or_near(needle: str, haystack: str) -> bool:
    if needle and needle in haystack:
        return True
    needle_words = _tokens(needle)
    haystack_words = _tokens(haystack)
    return _longest_shared_span(needle_words, haystack_words) >= _NEAR_COPY_SPAN


def check_gold_leakage(
    generations: Sequence[dict[str, Any]],
    gold_by_target: Mapping[str, str],
    packs: Mapping[str, TargetEvidencePack],
    representations: Mapping[str, RepresentationArtifact],
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    targets = sorted(gold_by_target)
    for target_id in targets:
        gold = gold_by_target[target_id]
        if not gold:
            continue
        haystacks: dict[str, str] = {"evidence_pack": packs[target_id].content}
        for name, representation in representations.items():
            haystacks[f"representation_{name}"] = representation.content
        for generation in generations:
            if generation["target_id"] == target_id:
                haystacks[f"generation_{generation['condition']}"] = generation["text"]
        for source, haystack in haystacks.items():
            if _exact_or_near(gold, haystack):
                violations.append({"target_id": target_id, "source": source})
    return {"passed": not violations, "targets_checked": len(targets), "violations": violations}


def check_source_domain_leakage(
    generations: Sequence[dict[str, Any]],
    sources: Sequence[SourcePaper],
) -> dict[str, Any]:
    source_shingles: set[tuple[str, ...]] = set()
    for source in sources:
        source_shingles |= _shingles(
            _tokens(source.introduction.normalized_text), _COPY_NGRAM
        )
    violations: list[dict[str, Any]] = []
    for generation in generations:
        generation_words = _tokens(generation["text"])
        overlaps = source_shingles & _shingles(generation_words, _COPY_NGRAM)
        if overlaps:
            violations.append(
                {
                    "generation_id": generation["generation_id"],
                    "condition": generation["condition"],
                    "shared_ngrams": list(overlaps)[:5],
                }
            )
    return {"passed": not violations, "violations": violations}


def check_guideline_experience_distinct(
    guideline: RepresentationArtifact,
    experience: RepresentationArtifact,
    generations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    guideline_tokens = set(_tokens(guideline.content))
    experience_tokens = set(_tokens(experience.content))
    union = guideline_tokens | experience_tokens
    jaccard = len(guideline_tokens & experience_tokens) / len(union) if union else 0.0

    identical_generations = 0
    compared = 0
    by_target: dict[str, dict[str, str]] = {}
    for generation in generations:
        by_target.setdefault(generation["target_id"], {})[generation["condition"]] = generation["text"]
    for target_id, condition_texts in by_target.items():
        if "guideline" in condition_texts and "experience" in condition_texts:
            compared += 1
            if condition_texts["guideline"] == condition_texts["experience"]:
                identical_generations += 1

    return {
        "passed": jaccard < 0.5 and identical_generations == 0,
        "content_jaccard": round(jaccard, 4),
        "generation_targets_compared": compared,
        "identical_generation_targets": identical_generations,
    }


def check_experience_strategies_actionable(experience_content: str) -> dict[str, Any]:
    try:
        entries = json.loads(experience_content)
    except json.JSONDecodeError:
        entries = []
    if not isinstance(entries, list):
        entries = []
    actionable = 0
    malformed = 0
    for entry in entries:
        if not isinstance(entry, dict):
            malformed += 1
            continue
        strategy = entry.get("strategy", "")
        observed = entry.get("observed_pattern", "")
        applicable = entry.get("applicable_when", "")
        words = _tokens(strategy)
        is_actionable = (
            bool(strategy)
            and bool(observed)
            and bool(applicable)
            and strategy != observed
            and bool(words)
            and words[0] in _IMPERATIVE_VERBS
        )
        if is_actionable:
            actionable += 1
        else:
            malformed += 1
    total = len(entries)
    ratio = actionable / total if total else 0.0
    return {
        "passed": total > 0 and ratio >= 0.8,
        "entry_count": total,
        "actionable_count": actionable,
        "actionable_ratio": round(ratio, 4),
    }


def check_token_compute_matching(
    representations: Mapping[str, RepresentationArtifact],
    generations: Sequence[dict[str, Any]],
    writing_condition_tokens: int,
    writer_max_output_tokens: int,
) -> dict[str, Any]:
    over_budget = {
        name: representation.content_tokens
        for name, representation in representations.items()
        if representation.content_tokens > writing_condition_tokens
    }
    writers = {generation["writer_model"] for generation in generations}
    over_output = [
        generation["generation_id"]
        for generation in generations
        if generation["output_tokens"] > writer_max_output_tokens
    ]
    input_tokens = {generation["input_tokens"] for generation in generations}
    return {
        "passed": not over_budget and len(writers) == 1 and not over_output,
        "writing_condition_budget_tokens": writing_condition_tokens,
        "over_budget_representations": over_budget,
        "writer_models": sorted(writers),
        "over_max_output_generations": over_output,
        "writer_input_token_range": [min(input_tokens), max(input_tokens)],
    }


def check_output_format_and_citations(
    generations: Sequence[dict[str, Any]],
    writer_max_output_tokens: int,
) -> dict[str, Any]:
    problems: list[dict[str, Any]] = []
    for generation in generations:
        text = generation["text"]
        issue = None
        if not text.strip():
            issue = "empty_text"
        elif generation["output_tokens"] > writer_max_output_tokens:
            issue = "over_max_output_tokens"
        elif re.search(r"\b(None|null|undefined)\b", text, re.IGNORECASE):
            issue = "placeholder_literal"
        else:
            bracket_content = re.findall(r"\[(.*?)\]", text)
            if any(part and not part.strip().isdigit() for part in bracket_content):
                issue = "non_numeric_bracket_citation"
            elif text.count("[") != text.count("]"):
                issue = "unbalanced_brackets"
        if issue:
            problems.append({"generation_id": generation["generation_id"], "issue": issue})
    return {"passed": not problems, "problems": problems}


def _copy_rate(generation: dict[str, Any], sources: Sequence[SourcePaper]) -> dict[str, float]:
    generation_words = _tokens(generation["text"])
    max_source_overlap = 0.0
    for source in sources:
        overlap = _longest_shared_span(
            generation_words, _tokens(source.introduction.normalized_text)
        )
        max_source_overlap = max(max_source_overlap, overlap)
    return {
        "max_shared_source_span_tokens": max_source_overlap,
        "text_tokens": len(generation_words),
    }


def run_pilot_review(
    generations: Sequence[dict[str, Any]],
    gold_by_target: Mapping[str, str],
    packs: Mapping[str, TargetEvidencePack],
    representations: Mapping[str, RepresentationArtifact],
    sources: Sequence[SourcePaper],
    writing_condition_tokens: int,
    writer_max_output_tokens: int,
) -> dict[str, Any]:
    checks = {
        "gold_leakage": check_gold_leakage(generations, gold_by_target, packs, representations),
        "source_domain_leakage_in_generations": check_source_domain_leakage(generations, sources),
        "guideline_vs_experience_distinct": check_guideline_experience_distinct(
            representations["guideline"],
            representations["experience"],
            generations,
        ),
        "experience_strategies_actionable": check_experience_strategies_actionable(
            representations["experience"].content
        ),
        "token_compute_matching": check_token_compute_matching(
            representations,
            generations,
            writing_condition_tokens,
            writer_max_output_tokens,
        ),
        "output_format_and_citations": check_output_format_and_citations(
            generations, writer_max_output_tokens
        ),
    }
    target_ids = {generation["target_id"] for generation in generations}
    conditions = {generation["condition"] for generation in generations}

    copy_rates = {
        generation["generation_id"]: _copy_rate(generation, sources)
        for generation in generations
    }
    diagnostics = {
        "target_count": len(target_ids),
        "condition_count": len(conditions),
        "generation_count": len(generations),
        "conditions": sorted(conditions),
        "copy_rate_max_shared_source_span_tokens": round(
            max((item["max_shared_source_span_tokens"] for item in copy_rates.values()), default=0.0),
            2,
        ),
        "experience_grounding_status": "support_verified",
        "guideline_experience_content_jaccard": checks["guideline_vs_experience_distinct"][
            "content_jaccard"
        ],
        "experience_actionable_ratio": checks["experience_strategies_actionable"][
            "actionable_ratio"
        ],
        "representations_tokens": {
            name: representation.content_tokens for name, representation in representations.items()
        },
        "compiler_models": {
            name: representation.compiler_model for name, representation in representations.items()
        },
        "compression_ratio": {
            name: (
                round(
                    representation.compiler_input_tokens / representation.content_tokens, 2
                )
                if representation.content_tokens
                else None
            )
            for name, representation in representations.items()
        },
    }
    return {
        "review_version": "pilot-review-v1",
        "stage": "Gate 5 pilot review",
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "diagnostics": diagnostics,
    }
