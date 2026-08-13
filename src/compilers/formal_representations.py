from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from src.adapters.counting import CountingAdapter
from src.adapters.model import ModelAdapter, ModelRequest
from src.budget.tokenizer import BudgetController, Tokenizer
from src.compilers.base import build_representation
from src.compilers.config import CompilerSettings
from src.compilers.corpus import source_corpus_hash
from src.compilers.experience.pipeline import ExperienceLibraryResult
from src.compilers.experience.representation import experience_prompt_version
from src.compilers.guideline import _GUIDELINE_SYSTEM
from src.compilers.summary import _SUMMARY_SYSTEM, _corpus_text, _summary_user
from src.domain.models import RepresentationArtifact, SourcePaper
from src.formal_metadata import FormalArtifactMetadata


@dataclass(frozen=True)
class ComputeEnvelope:
    calls: int
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if self.calls <= 0 or self.input_tokens <= 0 or self.output_tokens < 0:
            raise ValueError("Compute envelope requires positive calls/input and nonnegative output")


@dataclass(frozen=True)
class RepresentationBudgetMetrics:
    pre_budget_tokens: int
    post_budget_tokens: int
    compression_ratio: float
    included_items: tuple[str, ...]
    excluded_items: tuple[str, ...]
    tokenizer_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metrics(
    *,
    pre: int,
    post: int,
    included: Sequence[str],
    excluded: Sequence[str],
    tokenizer: Tokenizer,
) -> RepresentationBudgetMetrics:
    return RepresentationBudgetMetrics(
        pre_budget_tokens=pre,
        post_budget_tokens=post,
        compression_ratio=round(post / pre, 8) if pre else 0.0,
        included_items=tuple(included),
        excluded_items=tuple(excluded),
        tokenizer_version=tokenizer.version,
    )


def compile_raw_formal(
    sources: Sequence[SourcePaper],
    settings: CompilerSettings,
    tokenizer: Tokenizer,
    limit: int,
    metadata: FormalArtifactMetadata,
    provider_profile_hash: str,
) -> tuple[RepresentationArtifact, RepresentationBudgetMetrics]:
    ordered = sorted(sources, key=lambda source: source.source_id)
    blocks = [
        (source.source_id, f"[{source.source_id}] {source.introduction.normalized_text}")
        for source in ordered
    ]
    included: list[str] = []
    excluded: list[str] = []
    content_blocks: list[str] = []
    pre = len(tokenizer.encode("\n\n".join(block for _, block in blocks)))
    for source_id, block in blocks:
        tentative = "\n\n".join([*content_blocks, block])
        if len(tokenizer.encode(tentative)) <= limit:
            included.append(source_id)
            content_blocks.append(block)
        else:
            excluded.append(source_id)
    content = "\n\n".join(content_blocks)
    post = len(tokenizer.encode(content))
    if not content:
        raise ValueError("No complete raw ACL exemplar fits the formal writing budget")
    artifact = build_representation(
        "raw",
        content,
        post,
        source_corpus_hash(sources),
        "none",
        "stage3r-raw-selection-1",
        0,
        0,
        0,
        run_id=metadata.run_id,
        run_mode="formal",
        config_hash=metadata.config_hash,
        data_manifest_hash=metadata.data_manifest_hash,
        provider_profile_hash=provider_profile_hash,
    )
    return artifact, _metrics(
        pre=pre, post=post, included=included, excluded=excluded, tokenizer=tokenizer
    )


def compile_summary_formal(
    sources: Sequence[SourcePaper],
    settings: CompilerSettings,
    tokenizer: Tokenizer,
    limit: int,
    adapter: ModelAdapter,
    *,
    run_id: str,
    metadata: FormalArtifactMetadata,
    provider_profile_hash: str,
) -> tuple[RepresentationArtifact, RepresentationBudgetMetrics]:
    counting = CountingAdapter(adapter)
    response = counting.generate(
        ModelRequest(
            system_prompt=_SUMMARY_SYSTEM,
            user_prompt=_summary_user(sources, limit),
            max_output_tokens=limit,
            thinking_enabled=True,
            reasoning_effort="high",
            response_format="text",
            run_id=run_id,
            role="summary_compiler",
            run_mode="formal",
            config_hash=metadata.config_hash,
            data_manifest_hash=metadata.data_manifest_hash,
        )
    )
    budget = BudgetController(tokenizer).apply(response.text, limit)
    artifact = build_representation(
        "summary",
        budget.content,
        budget.post_truncation_tokens,
        source_corpus_hash(sources),
        adapter.model_name,
        settings.summary_prompt_version,
        counting.input_tokens,
        counting.output_tokens,
        counting.calls,
        run_id=metadata.run_id,
        run_mode="formal",
        config_hash=metadata.config_hash,
        data_manifest_hash=metadata.data_manifest_hash,
        provider_profile_hash=provider_profile_hash,
    )
    return artifact, _metrics(
        pre=budget.pre_truncation_tokens,
        post=budget.post_truncation_tokens,
        included=("summary_response",),
        excluded=(),
        tokenizer=tokenizer,
    )


def _cyclic_token_slice(tokens: list[Any], start: int, count: int) -> tuple[list[Any], int]:
    if not tokens or count <= 0:
        return [], start
    output: list[Any] = []
    position = start
    while len(output) < count:
        remaining = count - len(output)
        take = min(remaining, len(tokens) - position)
        output.extend(tokens[position : position + take])
        position = (position + take) % len(tokens)
    return output, position


def compile_guideline_formal(
    sources: Sequence[SourcePaper],
    settings: CompilerSettings,
    tokenizer: Tokenizer,
    limit: int,
    adapter: ModelAdapter,
    envelope: ComputeEnvelope,
    *,
    run_id: str,
    metadata: FormalArtifactMetadata,
    provider_profile_hash: str,
) -> tuple[RepresentationArtifact, RepresentationBudgetMetrics]:
    corpus_tokens = list(tokenizer.encode(_corpus_text(sources)))
    if not corpus_tokens:
        raise ValueError("Guideline compilation requires source corpus text")
    counting = CountingAdapter(adapter)
    fragments: list[str] = []
    position = 0
    target_per_call = max(1, envelope.input_tokens // envelope.calls)
    output_per_call = max(64, math.ceil(limit / envelope.calls))
    for index in range(envelope.calls):
        prefix = (
            f"[TASK: compute-matched generated guideline pass {index + 1}/{envelope.calls}]\n"
            "Generate complementary actionable conditional scientific-writing guidelines "
            "(when X, do Y) from this ACL source-corpus slice. Do not emit provenance, exact "
            "spans, verifier output, Experience schema, or semantic consolidation.\n"
            "Source-corpus slice:\n"
        )
        fixed_tokens = len(tokenizer.encode(f"{_GUIDELINE_SYSTEM}\n{prefix}"))
        slice_size = max(1, target_per_call - fixed_tokens)
        selected, position = _cyclic_token_slice(corpus_tokens, position, slice_size)
        response = counting.generate(
            ModelRequest(
                system_prompt=_GUIDELINE_SYSTEM,
                user_prompt=f"{prefix}{tokenizer.decode(selected)}",
                max_output_tokens=output_per_call,
                thinking_enabled=True,
                reasoning_effort="high",
                response_format="text",
                run_id=run_id,
                role="guideline_compiler",
                run_mode="formal",
                config_hash=metadata.config_hash,
                data_manifest_hash=metadata.data_manifest_hash,
            )
        )
        fragments.append(response.text)
    combined = "\n\n".join(fragments)
    budget = BudgetController(tokenizer).apply(combined, limit)
    included: list[str] = []
    excluded: list[str] = []
    consumed = 0
    for index, fragment in enumerate(fragments, start=1):
        item = f"guideline_pass_{index}"
        fragment_tokens = len(tokenizer.encode(fragment))
        if consumed < limit:
            included.append(item)
            consumed += fragment_tokens
        else:
            excluded.append(item)
    artifact = build_representation(
        "guideline",
        budget.content,
        budget.post_truncation_tokens,
        source_corpus_hash(sources),
        adapter.model_name,
        settings.guideline_prompt_version,
        counting.input_tokens,
        counting.output_tokens,
        counting.calls,
        run_id=metadata.run_id,
        run_mode="formal",
        config_hash=metadata.config_hash,
        data_manifest_hash=metadata.data_manifest_hash,
        provider_profile_hash=provider_profile_hash,
    )
    return artifact, _metrics(
        pre=budget.pre_truncation_tokens,
        post=budget.post_truncation_tokens,
        included=included,
        excluded=excluded,
        tokenizer=tokenizer,
    )


def representation_from_experience_result(
    result: ExperienceLibraryResult,
    settings: CompilerSettings,
    tokenizer: Tokenizer,
    metadata: FormalArtifactMetadata,
    provider_profile_hash: str,
) -> tuple[RepresentationArtifact, RepresentationBudgetMetrics]:
    artifact = build_representation(
        "experience",
        result.library.content,
        result.library.content_tokens,
        result.source_corpus_hash,
        settings.model or "deepseek-v4-flash",
        experience_prompt_version(settings),
        result.compiler_input_tokens,
        result.compiler_output_tokens,
        result.compiler_calls,
        run_id=metadata.run_id,
        run_mode="formal",
        config_hash=metadata.config_hash,
        data_manifest_hash=metadata.data_manifest_hash,
        provider_profile_hash=provider_profile_hash,
    )
    metrics = _metrics(
        pre=result.library.pre_budget_tokens,
        post=result.library.content_tokens,
        included=result.library.included_experience_ids,
        excluded=result.library.excluded_experience_ids,
        tokenizer=tokenizer,
    )
    return artifact, metrics
