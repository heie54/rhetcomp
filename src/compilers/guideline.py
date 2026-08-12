from __future__ import annotations

from typing import Iterable

from src.adapters.counting import CountingAdapter
from src.adapters.model import ModelAdapter, ModelRequest
from src.budget.selection import atomic_budget
from src.budget.tokenizer import BudgetController, Tokenizer
from src.compilers.base import build_representation, source_reading_tokens
from src.compilers.config import CompilerSettings
from src.compilers.corpus import source_corpus_hash
from src.compilers.summary import _corpus_text
from src.domain.models import RepresentationArtifact, SourcePaper


_GUIDELINE_SYSTEM = (
    "You are a scientific-writing instructor. Generate conditional scientific-writing "
    "guidelines derived from the provided source corpus."
)


def _guideline_user(sources: Iterable[SourcePaper], budget_tokens: int) -> str:
    return (
        "[TASK: compute-matched generated guideline]\n"
        "Source corpus:\n{corpus}\n\n"
        "Generate actionable conditional writing guidelines (when X, do Y) for writing "
        "scientific Introductions, derived from these sources, within {budget} tokens."
    ).format(corpus=_corpus_text(sources), budget=budget_tokens)


def _deterministic_guidelines(sources: Iterable[SourcePaper], tokenizer: Tokenizer, limit: int) -> str:
    """Source-derived conditional guidelines, independent of the Experience pipeline."""
    ordered = sorted(sources, key=lambda source: source.source_id)
    lines: list[str] = []
    for source in ordered:
        first_paragraph = source.introduction.paragraphs[0]
        opening = first_paragraph.sentences[0].text
        lines.append(
            f"Guideline: when opening an Introduction, state the problem or context directly "
            f"(example: \"{opening}\")."
        )
        if len(first_paragraph.sentences) > 1:
            second = first_paragraph.sentences[1].text
            lines.append(
                f"Guideline: after opening, position prior work before announcing the "
                f"contribution (example: \"{second}\")."
            )
        if len(source.introduction.paragraphs) > 1:
            contribution = source.introduction.paragraphs[-1].sentences[0].text
            lines.append(
                f"Guideline: close the Introduction by announcing the contribution "
                f"(example: \"{contribution}\")."
            )
    return atomic_budget(lines, tokenizer, limit).content


def compile_guideline(
    sources: Iterable[SourcePaper],
    settings: CompilerSettings,
    tokenizer: Tokenizer,
    writing_condition_tokens: int,
    adapter: ModelAdapter | None = None,
) -> RepresentationArtifact:
    """B3 — Compute-Matched Generated Guideline (spec §5). Adversarial baseline."""
    ordered = list(sources)
    corpus_hash = source_corpus_hash(sources)
    controller = BudgetController(tokenizer)

    if adapter is None:
        content = _deterministic_guidelines(ordered, tokenizer, writing_condition_tokens)
        content_tokens = len(tokenizer.encode(content))
        return build_representation(
            representation_type="guideline",
            content=content,
            content_tokens=content_tokens,
            source_corpus_hash=corpus_hash,
            compiler_model=settings.model or "deterministic-mock-1",
            compiler_prompt_version=settings.guideline_prompt_version,
            compiler_input_tokens=source_reading_tokens(ordered, tokenizer),
            compiler_output_tokens=content_tokens,
            compiler_calls=0,
        )

    counting = CountingAdapter(adapter)
    response = counting.generate(
        ModelRequest(
            system_prompt=_GUIDELINE_SYSTEM,
            user_prompt=_guideline_user(ordered, writing_condition_tokens),
            max_output_tokens=writing_condition_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        )
    )
    budget = controller.apply(response.text, writing_condition_tokens)
    return build_representation(
        representation_type="guideline",
        content=budget.content,
        content_tokens=budget.post_truncation_tokens,
        source_corpus_hash=corpus_hash,
        compiler_model=adapter.model_name,
        compiler_prompt_version=settings.guideline_prompt_version,
        compiler_input_tokens=counting.input_tokens,
        compiler_output_tokens=counting.output_tokens,
        compiler_calls=counting.calls,
    )
