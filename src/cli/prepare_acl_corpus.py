from __future__ import annotations

import argparse
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from src.common.jsonio import sha256_json, sha256_text, write_json
from src.compilers.corpus import source_corpus_hash
from src.config import load_config
from src.ingest.formal_data import (
    deterministic_select,
    extract_introduction,
    parse_acl_anthology_metadata,
)
from src.ingest.network import fetch_bytes
from src.ingest.source import normalize_source_record


ROOT = Path(__file__).resolve().parents[2]


def _resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _pdf_to_markdown(path: Path) -> str:
    """Extract ACL proceedings text in two-column reading order."""

    import pdfplumber

    with pdfplumber.open(path) as document:
        location: tuple[int, float, float] | None = None
        for page_index, page in enumerate(document.pages):
            for match in page.search(r"(?i)\bintroduction\b", regex=True):
                line = page.crop(
                    (
                        0.0,
                        max(0.0, float(match["top"]) - 4.0),
                        page.width,
                        min(page.height, float(match["bottom"]) + 4.0),
                    )
                ).extract_text(x_tolerance=2, y_tolerance=3) or ""
                if re.search(r"(?i)\b(?:1[.)]?\s+)?introduction\b", line):
                    location = (page_index, float(match["top"]), float(match["x0"]))
                    break
            if location is not None:
                break

        if location is None:
            from markitdown import MarkItDown

            result = MarkItDown(enable_plugins=False).convert(str(path))
            text = getattr(result, "text_content", None) or getattr(result, "markdown", None)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"PDF extractors returned no text for {path}")
            return text

        start_page, heading_top, heading_x = location
        parts: list[str] = []
        for page_index in range(start_page, len(document.pages)):
            page = document.pages[page_index]
            midpoint = page.width / 2
            bottom = max(1.0, page.height - 48.0)
            if page_index == start_page:
                top = max(0.0, heading_top - 4.0)
                columns = (
                    ((midpoint, top, page.width, bottom),)
                    if heading_x >= midpoint
                    else (
                        (0.0, top, midpoint, bottom),
                        (midpoint, top, page.width, bottom),
                    )
                )
            else:
                top = 42.0
                columns = (
                    (0.0, top, midpoint, bottom),
                    (midpoint, top, page.width, bottom),
                )
            for bounds in columns:
                text = page.crop(bounds).extract_text(x_tolerance=2, y_tolerance=3)
                if text:
                    parts.append(text)
        return "\n\n".join(parts)


def prepare_acl_corpus(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    accept_source_licenses: bool,
    fetcher: Callable[[str], bytes] = fetch_bytes,
    pdf_to_markdown: Callable[[Path], str] = _pdf_to_markdown,
) -> tuple[dict[str, Any], Path]:
    if not accept_source_licenses:
        raise PermissionError(
            "ACL source processing requires explicit confirmation of source-paper licenses"
        )
    config_file = Path(config_path).resolve()
    root = Path(project_root).resolve() if project_root else config_file.parent.parent
    config = load_config(config_file)
    if config.get("run_mode") != "formal":
        raise ValueError("prepare_acl_corpus requires a formal dataset config")
    source_config = config["source"]
    paths = config["paths"]
    metadata_path = _resolve(root, paths["acl_metadata_raw"])
    pdf_root = _resolve(root, paths["acl_pdf_root"])
    markdown_root = _resolve(root, paths["acl_markdown_root"])
    normalized_root = _resolve(root, paths["source_normalized"])
    manifest_path = _resolve(root, paths["manifests"]) / "acl_pilot.json"

    if not metadata_path.exists():
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_bytes(fetcher(str(source_config["metadata_url"])))
    metadata_bytes = metadata_path.read_bytes()
    all_papers = parse_acl_anthology_metadata(
        metadata_bytes, volume_id=str(source_config["volume"])
    )
    selected = deterministic_select(
        all_papers,
        id_field="source_id",
        count=int(source_config["expected_count"]),
        seed=str(source_config["selection_seed"]),
    )

    entries: list[dict[str, Any]] = []
    normalized_sources = []
    for metadata in selected:
        source_id = metadata["source_id"]
        pdf_path = pdf_root / f"{source_id}.pdf"
        markdown_path = markdown_root / f"{source_id}.md"
        entry = {
            "source_id": source_id,
            "title": metadata["title"],
            "authors": metadata["authors"],
            "source_url": metadata["source_url"],
            "pdf_url": metadata["pdf_url"],
            "doi": metadata["doi"],
            "status": "pending",
        }
        try:
            if not pdf_path.exists():
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                pdf_path.write_bytes(fetcher(metadata["pdf_url"]))
            pdf_bytes = pdf_path.read_bytes()
            if not pdf_bytes.startswith(b"%PDF"):
                raise ValueError("Downloaded source is not a PDF")
            markdown = pdf_to_markdown(pdf_path)
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
            introduction = extract_introduction(markdown)
            source = normalize_source_record({**metadata, "introduction": introduction})
            normalized_payload = source.to_dict()
            write_json(normalized_root / f"{source.source_id}.json", normalized_payload)
            normalized_sources.append(source)
            entry.update(
                {
                    "status": "ready",
                    "pdf_sha256": sha256(pdf_bytes).hexdigest(),
                    "markdown_sha256": sha256_text(markdown),
                    "introduction_sha256": sha256_text(source.introduction.normalized_text),
                    "document_hash": source.document_hash,
                    "normalized_payload_hash": sha256_json(normalized_payload),
                    "extraction_method": "markitdown+heading-boundary-v1",
                }
            )
        except Exception as exc:
            entry.update({"status": "failed", "error": str(exc)})
        entries.append(entry)

    ready_count = sum(entry["status"] == "ready" for entry in entries)
    manifest = {
        "manifest_version": "acl-pilot-formal-1",
        "run_mode": "formal",
        "dataset_config_version": config["config_version"],
        "provider": source_config["provider"],
        "metadata_url": source_config["metadata_url"],
        "license_identifier": source_config["license_identifier"],
        "license_url": source_config["license_url"],
        "source_license_confirmation": True,
        "metadata_sha256": sha256(metadata_bytes).hexdigest(),
        "selection_seed": source_config["selection_seed"],
        "expected_count": int(source_config["expected_count"]),
        "ready_count": ready_count,
        "source_corpus_hash": (
            source_corpus_hash(normalized_sources) if ready_count == len(entries) else None
        ),
        "entries": entries,
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    write_json(manifest_path, manifest)
    return manifest, manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare real ACL 2024 Introduction corpus")
    parser.add_argument("--profile", choices=("pilot",), default="pilot")
    parser.add_argument("--config", default="configs/dataset_formal_pilot.yaml")
    parser.add_argument(
        "--accept-source-licenses",
        action="store_true",
        help="confirm that applicable source-paper licenses were reviewed",
    )
    args = parser.parse_args(argv)
    try:
        manifest, path = prepare_acl_corpus(
            args.config, accept_source_licenses=args.accept_source_licenses
        )
    except PermissionError as exc:
        print(f"ACL_CORPUS=BLOCKED")
        print(f"BLOCKER={exc}")
        return 2
    passed = manifest["ready_count"] == manifest["expected_count"]
    print(f"ACL_CORPUS={'PASS' if passed else 'FAIL'}")
    print(f"MANIFEST={path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
