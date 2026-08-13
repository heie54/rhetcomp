from __future__ import annotations

from typing import Any

from src.common.jsonio import canonical_json, sha256_text
from src.writer.config import WriterSettings

_CONDITION_LABELS = {
    "raw": "raw source exemplars",
    "summary": "source-corpus summary",
    "guideline": "generated guideline",
    "experience": "provenance-grounded experience library",
}


def build_system_prompt(settings: WriterSettings) -> str:
    return (
        "You are a scientific writer writing the Introduction section of a research paper. "
        "Write in clear, formal scientific prose. The Target Evidence Pack is the only factual "
        "source. Writing references may guide rhetoric and organization only; never copy their "
        "facts, models, benchmarks, conclusions, or citations into the target paper. Do not "
        "invent facts outside the Target Evidence Pack. Every citation must resolve to an item "
        "in Target Evidence Pack.reference_metadata. Cite evidence using "
        f"{settings.citation_format} citations. Citation numbers are the 1-based positions in "
        "reference_metadata, never source-paper numbering. Use [1], [1, 2], or a compact range "
        "such as [1-4] only when every expanded index is within reference_metadata. If "
        "reference_metadata is empty, do not emit any citations. "
        f"Target Introduction length: about {settings.desired_introduction_length} tokens."
    )


def build_task_prompt(target_id: str) -> str:
    return (
        "[WRITER TASK]\n"
        f"Target id: {target_id}\n"
        "Write the Introduction of this paper using the Target Evidence Pack below."
    )


def build_condition_text(
    condition: str,
    representation_content: str | None,
) -> str:
    if condition == "evidence_only":
        return "Writing reference: none. Use only the Target Evidence Pack for facts and citations."
    return (
        f"Writing reference ({_CONDITION_LABELS[condition]}; rhetorical/organizational guidance "
        f"only, never a factual or citation source):\n{representation_content}"
    )


def prompt_template_hash(system_prompt: str, task_prompt: str) -> str:
    return sha256_text(canonical_json({"system": system_prompt, "task": task_prompt}))


def base_prompt_hash(system_prompt: str, task_prompt: str, evidence_content: str) -> str:
    return sha256_text(
        canonical_json(
            {"system": system_prompt, "task": task_prompt, "evidence": evidence_content}
        )
    )


def full_prompt_hash(
    system_prompt: str,
    task_prompt: str,
    evidence_content: str,
    condition: str,
    representation_content: str | None,
) -> str:
    payload: dict[str, Any] = {
        "system": system_prompt,
        "task": task_prompt,
        "evidence": evidence_content,
        "condition": condition,
        "representation": representation_content,
    }
    return sha256_text(canonical_json(payload))
