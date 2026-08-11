from __future__ import annotations

from src.budget.tokenizer import BudgetController
from src.common.jsonio import canonical_json, sha256_json
from src.domain.models import TargetEvidence, TargetEvidencePack, TargetVisible


SOURCE_FIELDS = ("title", "abstract", "non_intro_body", "reference_metadata")


def build_target_evidence_pack(
    visible: TargetVisible,
    evidence: TargetEvidence,
    budget_tokens: int,
    budget_controller: BudgetController,
) -> TargetEvidencePack:
    if visible.target_id != evidence.target_id:
        raise ValueError("Visible and evidence records must have the same target_id")
    source_payload = {
        "target_id": visible.target_id,
        "title": visible.title,
        "abstract": visible.abstract,
        "non_intro_body": evidence.non_intro_sections,
        "reference_metadata": evidence.reference_metadata,
    }
    serialized = canonical_json(source_payload)
    result = budget_controller.apply(serialized, budget_tokens)
    return TargetEvidencePack(
        target_id=visible.target_id,
        budget_tokens=budget_tokens,
        content=result.content,
        source_fields=SOURCE_FIELDS,
        input_hash=sha256_json(source_payload),
        tokenizer_version=result.tokenizer_version,
        pre_truncation_tokens=result.pre_truncation_tokens,
        post_truncation_tokens=result.post_truncation_tokens,
    )
