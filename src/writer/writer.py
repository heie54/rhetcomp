from __future__ import annotations

import json
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationArtifact":
        return cls(**value)


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
        task_prompt = build_task_prompt(pack.target_id)
        condition_text = build_condition_text(
            condition, representation.content if representation else None
        )
        evidence_content = pack.content
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
                )
            )
            text = response.text
            input_tokens = counting.input_tokens
            output_tokens = counting.output_tokens
            latency_ms = counting.latency_ms
            writer_model = self.adapter.model_name

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
        )
