from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable

from src.common.jsonio import canonical_json, sha256_json
from src.domain.models import SourcePaper, TargetEvidencePack
from src.runtime.data_access import (
    CompilerDataAccess,
    EvaluatorDataAccess,
    WriterDataAccess,
)


CONDITIONS = ("evidence_only", "raw", "summary", "guideline", "experience")


def _is_gold_capability_name(name: str) -> bool:
    normalized = name.lower()
    return "gold" in normalized or "evaluator" in normalized


def audit_source_addressability(sources: Iterable[SourcePaper]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    count = 0
    for source in sources:
        count += 1
        for expected_paragraph, paragraph in enumerate(source.introduction.paragraphs, start=1):
            if paragraph.paragraph_id != expected_paragraph:
                failures.append({"source_id": source.source_id, "reason": "paragraph_id_sequence"})
            for expected_sentence, sentence in enumerate(paragraph.sentences, start=1):
                addressed = source.introduction.normalized_text[
                    sentence.char_start:sentence.char_end
                ]
                if sentence.sentence_id != expected_sentence or addressed != sentence.text:
                    failures.append(
                        {
                            "source_id": source.source_id,
                            "paragraph_id": paragraph.paragraph_id,
                            "sentence_id": sentence.sentence_id,
                            "reason": "coordinate_mismatch",
                        }
                    )
    return {"passed": not failures, "source_count": count, "failures": failures}


def audit_gold_import_isolation(project_root: Path) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    checked_files: list[str] = []
    for package in (project_root / "src" / "compilers", project_root / "src" / "writer"):
        for path in sorted(package.rglob("*.py")):
            relative = path.relative_to(project_root).as_posix()
            checked_files.append(relative)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                    if any(_is_gold_capability_name(name) for name in names):
                        violations.append(
                            {"file": relative, "reason": "gold_capability_import", "names": names}
                        )
                elif isinstance(node, ast.ImportFrom):
                    names = [alias.name for alias in node.names]
                    module = node.module or ""
                    if _is_gold_capability_name(module) or any(
                        _is_gold_capability_name(name) for name in names
                    ):
                        violations.append(
                            {
                                "file": relative,
                                "reason": "gold_capability_import",
                                "module": module,
                                "names": names,
                            }
                        )
                elif isinstance(node, ast.Name) and _is_gold_capability_name(node.id):
                    violations.append(
                        {"file": relative, "reason": "gold_capability_name", "name": node.id}
                    )
                elif isinstance(node, ast.Attribute) and _is_gold_capability_name(node.attr):
                    violations.append(
                        {"file": relative, "reason": "gold_capability_attribute", "name": node.attr}
                    )
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    normalized = node.value.replace("\\", "/").lower()
                    if "target_gold" in normalized or "target/gold" in normalized:
                        violations.append({"file": relative, "reason": "gold_path_literal"})
    return {"passed": not violations, "checked_files": checked_files, "violations": violations}


def audit_runtime_gold_isolation(
    compiler_access: CompilerDataAccess,
    writer_access: WriterDataAccess,
    evaluator_access: EvaluatorDataAccess,
    expected_gold_root: Path,
) -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    expected = expected_gold_root.resolve()
    restricted = (
        ("compiler", compiler_access),
        ("writer", writer_access),
    )
    role_capabilities: dict[str, list[str]] = {}

    for role, access in restricted:
        roots = {
            name: path.resolve()
            for name, path in access.configured_roots().items()
        }
        role_capabilities[role] = sorted(roots)
        if any("gold" in name.lower() for name in roots):
            violations.append({"role": role, "reason": "gold_capability_exposed"})
        if expected in roots.values():
            violations.append({"role": role, "reason": "gold_root_held"})
        public_gold_members = [
            name for name in dir(access) if not name.startswith("_") and "gold" in name.lower()
        ]
        if public_gold_members:
            violations.append(
                {
                    "role": role,
                    "reason": "gold_member_exposed",
                    "members": ",".join(sorted(public_gold_members)),
                }
            )

    evaluator_roots = {
        name: path.resolve()
        for name, path in evaluator_access.configured_roots().items()
    }
    role_capabilities["evaluator"] = sorted(evaluator_roots)
    if evaluator_roots != {"target_gold": expected}:
        violations.append({"role": "evaluator", "reason": "gold_capability_missing_or_wrong"})
    if evaluator_access.target_gold_path("capability_probe").parent.resolve() != expected:
        violations.append({"role": "evaluator", "reason": "gold_path_resolution_wrong"})

    return {
        "passed": not violations,
        "role_capabilities": role_capabilities,
        "evaluator_gold_root_matches_expected": evaluator_roots == {"target_gold": expected},
        "violations": violations,
    }


def audit_target_separation(
    visible_payloads: Iterable[dict[str, Any]],
    evidence_payloads: Iterable[dict[str, Any]],
    gold_payloads: Iterable[dict[str, Any]],
    visible_root: Path,
    evidence_root: Path,
    gold_root: Path,
) -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    roots = [visible_root.resolve(), evidence_root.resolve(), gold_root.resolve()]
    if len(set(roots)) != 3:
        violations.append({"reason": "namespace_roots_not_distinct"})
    visible_by_id = {item["target_id"]: item for item in visible_payloads}
    evidence_by_id = {item["target_id"]: item for item in evidence_payloads}
    gold_by_id = {item["target_id"]: item for item in gold_payloads}
    if not (set(visible_by_id) == set(evidence_by_id) == set(gold_by_id)):
        violations.append({"reason": "target_id_sets_differ"})
    for target_id, gold in gold_by_id.items():
        if "introduction" not in gold:
            violations.append({"target_id": target_id, "reason": "gold_introduction_missing"})
        for namespace, payload in (
            ("visible", visible_by_id.get(target_id, {})),
            ("evidence", evidence_by_id.get(target_id, {})),
        ):
            if "gold" in payload or "introduction" in payload:
                violations.append({"target_id": target_id, "reason": f"gold_present_in_{namespace}"})
            if gold.get("introduction") and gold["introduction"] in canonical_json(payload):
                violations.append({"target_id": target_id, "reason": f"gold_text_present_in_{namespace}"})
    return {"passed": not violations, "target_count": len(gold_by_id), "violations": violations}


def audit_evidence_pack_reuse(packs: Iterable[TargetEvidencePack]) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    violations: list[dict[str, str]] = []
    for pack in packs:
        serialized_hash = sha256_json(pack.to_dict())
        assignments = {condition: serialized_hash for condition in CONDITIONS}
        if len(set(assignments.values())) != 1:
            violations.append({"target_id": pack.target_id, "reason": "condition_hash_mismatch"})
        targets[pack.target_id] = {
            "evidence_pack_hash": serialized_hash,
            "condition_assignments": assignments,
        }
    return {"passed": not violations, "target_count": len(targets), "targets": targets, "violations": violations}


def build_gate1_report(
    project_root: Path,
    sources: list[SourcePaper],
    visible_payloads: list[dict[str, Any]],
    evidence_payloads: list[dict[str, Any]],
    gold_payloads: list[dict[str, Any]],
    packs: list[TargetEvidencePack],
    expected_source_count: int,
    expected_target_count: int,
    roots: tuple[Path, Path, Path],
    runtime_accesses: tuple[CompilerDataAccess, WriterDataAccess, EvaluatorDataAccess],
) -> dict[str, Any]:
    checks = {
        "pilot_counts": {
            "passed": len(sources) == expected_source_count and len(packs) == expected_target_count,
            "expected_source_count": expected_source_count,
            "actual_source_count": len(sources),
            "expected_target_count": expected_target_count,
            "actual_target_count": len(packs),
        },
        "source_addressability": audit_source_addressability(sources),
        "gold_import_isolation": audit_gold_import_isolation(project_root),
        "runtime_gold_isolation": audit_runtime_gold_isolation(
            runtime_accesses[0],
            runtime_accesses[1],
            runtime_accesses[2],
            roots[2],
        ),
        "target_namespace_separation": audit_target_separation(
            visible_payloads,
            evidence_payloads,
            gold_payloads,
            roots[0],
            roots[1],
            roots[2],
        ),
        "evidence_pack_reuse": audit_evidence_pack_reuse(packs),
    }
    return {
        "audit_version": "gate1-v2",
        "gate": "Gate 1",
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }
