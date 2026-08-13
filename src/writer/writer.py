from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from src.adapters.counting import CountingAdapter
from src.adapters.model import ModelAdapter, ModelRequest
from src.budget.tokenizer import Tokenizer
from src.common.jsonio import canonical_json, sha256_text
from src.domain.models import RepresentationArtifact, TargetEvidencePack
from src.writer.config import WRITER_CONDITIONS, WriterSettings
from src.writer.prompts import (
    base_prompt_hash,
    build_condition_text,
    build_system_prompt,
    build_task_prompt,
    full_prompt_hash,
    prompt_template_hash,
)


@dataclass(frozen=True)
class GenerationArtifact:
    generation_id: str
    target_id: str
    condition: str
    writer_model: str
    writer_prompt_hash: str
    prompt_template_hash: str
    base_prompt_hash: str
    target_evidence_hash: str
    representation_hash: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: int
    text: str
    citation_indices: tuple[int, ...] = ()
    citation_valid: bool = True
    provider_metadata: dict[str, Any] | None = None
    run_mode: str = "mechanics"
    run_id: str = "unscoped"
    config_hash: str | None = None
    data_manifest_hash: str | None = None
    provider_profile_hash: str | None = None

    def __post_init__(self) -> None:
        if self.run_mode not in {"mechanics", "formal"}:
            raise ValueError("Invalid generation run mode")
        if self.run_mode == "formal" and any(
            not value
            for value in (
                self.run_id,
                self.config_hash,
                self.data_manifest_hash,
                self.provider_profile_hash,
            )
        ):
            raise ValueError("Formal generation requires complete artifact metadata")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationArtifact":
        payload = dict(value)
        payload["citation_indices"] = tuple(payload.get("citation_indices", ()))
        return cls(**payload)


_CITATION = re.compile(r"\[(\d+(?:\s*(?:,|[-–])\s*\d+)*)\]")
_BRACKET_CONTENT = re.compile(r"\[(.*?)\]")
_SOURCE_CITATION = re.compile(r"\[(?:\s*\d+\s*(?:,|[-–])?\s*)+\]")


def citation_indices(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for match in _CITATION.finditer(text):
        for part in re.split(r"\s*,\s*", match.group(1)):
            range_match = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", part)
            if range_match:
                start, end = map(int, range_match.groups())
                if start <= end:
                    values.extend(range(start, end + 1))
                continue
            values.append(int(part))
    return tuple(values)


def evidence_prompt_content(pack: TargetEvidencePack) -> str:
    """Remove source-paper citation markers while preserving the frozen evidence artifact."""
    try:
        payload = json.loads(pack.content)
    except json.JSONDecodeError:
        return pack.content
    if not isinstance(payload, dict):
        return pack.content
    prompt_payload = dict(payload)
    for key in ("abstract",):
        value = prompt_payload.get(key)
        if isinstance(value, str):
            prompt_payload[key] = _SOURCE_CITATION.sub("", value)
    non_intro = prompt_payload.get("non_intro_body")
    if isinstance(non_intro, dict):
        prompt_payload["non_intro_body"] = {
            key: _SOURCE_CITATION.sub("", value) if isinstance(value, str) else value
            for key, value in non_intro.items()
        }
    references = prompt_payload.get("reference_metadata")
    if isinstance(references, list):
        normalized_references = []
        for index, reference in enumerate(references, start=1):
            if not isinstance(reference, dict):
                normalized_references.append(reference)
                continue
            normalized = dict(reference)
            normalized["idx"] = index
            if isinstance(normalized.get("title"), str):
                normalized["title"] = _SOURCE_CITATION.sub("", normalized["title"])
            normalized_references.append(normalized)
        prompt_payload["reference_metadata"] = normalized_references
    return json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def citation_availability_instruction(pack: TargetEvidencePack) -> str:
    try:
        payload = json.loads(pack.content)
    except json.JSONDecodeError:
        payload = {}
    references = payload.get("reference_metadata", []) if isinstance(payload, dict) else []
    reference_count = len(references) if isinstance(references, list) else 0
    if reference_count == 0:
        return (
            "Citation availability: reference_metadata is empty. Do not emit square-bracket "
            "citations or any citation numbers in the Introduction."
        )
    return (
        f"Citation availability: reference_metadata has {reference_count} entries. Every expanded "
        f"citation index must be between 1 and {reference_count}, inclusive."
    )


def citations_within_target_evidence(text: str, pack: TargetEvidencePack) -> tuple[tuple[int, ...], bool]:
    try:
        payload = json.loads(pack.content)
    except json.JSONDecodeError:
        payload = {}
    references = payload.get("reference_metadata", [])
    reference_count = len(references) if isinstance(references, list) else 0
    indices = citation_indices(text)
    bracket_content = _BRACKET_CONTENT.findall(text)
    syntax_valid = all(bool(part) and _CITATION.fullmatch(f"[{part}]") for part in bracket_content)
    brackets_balanced = text.count("[") == text.count("]")
    references_valid = all(1 <= index <= reference_count for index in indices)
    return indices, syntax_valid and brackets_balanced and references_valid


def _generation_id(target_id: str, condition: str, prompt_hash: str, text: str) -> str:
    digest = sha256_text(
        canonical_json(
            {"target_id": target_id, "condition": condition, "prompt": prompt_hash, "text": text}
        )
    )
    return f"gen_{digest[:20]}"


def _deterministic_introduction(
    pack: TargetEvidencePack,
    condition: str,
    representation: RepresentationArtifact | None,
) -> str:
    try:
        data = json.loads(pack.content)
    except json.JSONDecodeError:
        data = {}
    title = data.get("title", pack.target_id)
    abstract = data.get("abstract", "")
    body = data.get("non_intro_body", {})
    parts = [f"This paper presents {title}."]
    if abstract:
        parts.append(abstract.rstrip(".") + ".")
    for section_name in ("Methods", "Results", "Conclusion"):
        if section_name in body:
            parts.append(f"We describe the {section_name.lower()} of this study.")
    references = data.get("reference_metadata", [])
    if references:
        citation = "[" + "], [".join(str(index) for index in range(1, len(references) + 1)) + "]"
        parts.append(f"We build on prior work cited as {citation}.")
    if condition in ("raw", "summary", "guideline", "experience") and representation is not None:
        # Mechanics-mode writer reference lines stay generic so the pilot source-domain
        # leakage check is meaningful. Only Experience quotes its (generalized) strategy.
        if condition == "experience":
            entries = json.loads(representation.content or "[]")
            if entries:
                parts.append(f"A guiding strategy for this introduction is: {entries[0]['strategy']}")
        elif condition == "guideline":
            parts.append("We follow a generated writing guideline for the introduction structure.")
        elif condition == "summary":
            parts.append("We build on a compressed summary of the source corpus.")
        else:
            parts.append("We draw on raw source exemplars for rhetorical guidance.")
    parts.append("The remainder of the paper is organized as follows.")
    return " ".join(part for part in parts if part)


class Writer:
    """One shared Writer for all conditions (spec §11). Condition-invariant harness."""

    def __init__(
        self,
        settings: WriterSettings,
        tokenizer: Tokenizer,
        adapter: ModelAdapter | None = None,
    ) -> None:
        self.settings = settings
        self.tokenizer = tokenizer
        self.adapter = adapter

    def generate(
        self,
        pack: TargetEvidencePack,
        condition: str,
        representation: RepresentationArtifact | None = None,
        *,
        formal_mode: bool = False,
        run_id: str = "unscoped",
        config_hash: str | None = None,
        data_manifest_hash: str | None = None,
        provider_profile_hash: str | None = None,
    ) -> GenerationArtifact:
        if condition not in WRITER_CONDITIONS:
            raise ValueError(f"Unknown writer condition: {condition}")
        if condition == "evidence_only":
            if representation is not None:
                raise ValueError("evidence_only must not receive a representation")
            representation_hash = None
        else:
            if representation is None:
                raise ValueError(f"{condition} requires a representation")
            representation_hash = representation.content_hash

        system_prompt = build_system_prompt(self.settings)
        task_prompt = "\n".join(
            [build_task_prompt(pack.target_id), citation_availability_instruction(pack)]
        )
        condition_text = build_condition_text(
            condition, representation.content if representation else None
        )
        evidence_content = evidence_prompt_content(pack)
        target_evidence_hash = sha256_text(evidence_content)
        template_hash = prompt_template_hash(system_prompt, task_prompt)
        base_hash = base_prompt_hash(system_prompt, task_prompt, evidence_content)
        full_hash = full_prompt_hash(
            system_prompt,
            task_prompt,
            evidence_content,
            condition,
            representation.content if representation else None,
        )

        if formal_mode and self.adapter is None:
            raise ValueError("Formal Writer requires a real model adapter")
        if formal_mode and any(
            not value
            for value in (run_id, config_hash, data_manifest_hash, provider_profile_hash)
        ):
            raise ValueError("Formal Writer requires complete artifact metadata")
        provider_metadata = None
        if self.adapter is None:
            text = _deterministic_introduction(pack, condition, representation)
            input_tokens = len(
                self.tokenizer.encode(
                    "\n\n".join([system_prompt, task_prompt, evidence_content, condition_text])
                )
            )
            output_tokens = len(self.tokenizer.encode(text))
            latency_ms = 0
            writer_model = self.settings.model or "deterministic-mock-1"
        else:
            counting = CountingAdapter(self.adapter)
            response = counting.generate(
                ModelRequest(
                    system_prompt=system_prompt,
                    user_prompt="\n\n".join([task_prompt, evidence_content, condition_text]),
                    max_output_tokens=self.settings.max_output_tokens,
                    temperature=self.settings.temperature,
                    top_p=self.settings.top_p,
                    seed=self.settings.seed,
                    thinking_enabled=False if formal_mode else None,
                    reasoning_effort=None,
                    response_format="text",
                    run_id=run_id,
                    role=f"writer:{condition}",
                    run_mode="formal" if formal_mode else "mechanics",
                    config_hash=config_hash,
                    data_manifest_hash=data_manifest_hash,
                )
            )
            text = response.text
            input_tokens = counting.input_tokens
            output_tokens = counting.output_tokens
            latency_ms = counting.latency_ms
            writer_model = self.adapter.model_name
            provider_metadata = dict(response.metadata)

        cited_indices, citations_valid = citations_within_target_evidence(text, pack)
        generation_id = _generation_id(pack.target_id, condition, full_hash, text)
        return GenerationArtifact(
            generation_id=generation_id,
            target_id=pack.target_id,
            condition=condition,
            writer_model=writer_model,
            writer_prompt_hash=full_hash,
            prompt_template_hash=template_hash,
            base_prompt_hash=base_hash,
            target_evidence_hash=target_evidence_hash,
            representation_hash=representation_hash,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            text=text,
            citation_indices=cited_indices,
            citation_valid=citations_valid,
            provider_metadata=provider_metadata,
            run_mode="formal" if formal_mode else "mechanics",
            run_id=run_id,
            config_hash=config_hash,
            data_manifest_hash=data_manifest_hash,
            provider_profile_hash=provider_profile_hash,
        )
