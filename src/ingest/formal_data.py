from __future__ import annotations

import re
import statistics
import xml.etree.ElementTree as ET
from hashlib import sha256
from typing import Any, Iterable, Sequence

from src.budget.tokenizer import Tokenizer
from src.ingest.text import normalize_plain_text


_MARKDOWN_HEADING = re.compile(
    r"^(?P<marks>#{1,6})\s*(?:\d+(?:\.\d+)*[.)]?\s*)?introduction\s*$",
    re.IGNORECASE,
)
_PLAIN_INTRO_HEADING = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?introduction\s*$", re.IGNORECASE
)
_PLAIN_SECTION_HEADING = re.compile(
    r"^\s*(?P<number>\d{1,2}(?:\.\d{1,2})*)[.)]?\s+(?P<title>\S.*)\s*$"
)
_WORD = re.compile(r"\b\w+\b", re.UNICODE)
_ACL_LAYOUT_ARTIFACT = re.compile(
    r"^(?:\d+|\d{1,2}\s+\S+/\S+|[*†‡§¶|]+\s*(?:corresponding\s+authors?\.?|"
    r"these\s+authors?\s+contribute.*|this\s+work.*|work\s+done.*|equal\s+contribution.*)|"
    r"proceedings?\s+of\s+the\s+62nd.*|august\s+11.*association\s+for\s+computational\s+linguistics.*)$",
    re.IGNORECASE,
)


def _is_plain_section_heading(line: str) -> bool:
    match = _PLAIN_SECTION_HEADING.match(line)
    if match is None:
        return False
    number = match.group("number")
    title = match.group("title").strip()
    return (
        not number.startswith("0")
        and "/" not in title
        and 3 <= len(title) <= 100
        and len(title.split()) <= 14
        and title[-1] not in ".,;:"
        and bool(re.search(r"[A-Za-z]", title))
    )


def deterministic_select(
    records: Sequence[dict[str, Any]],
    *,
    id_field: str,
    count: int,
    seed: str,
) -> list[dict[str, Any]]:
    if count <= 0 or count > len(records):
        raise ValueError(f"Selection count {count} is invalid for {len(records)} records")
    ids = [record.get(id_field) for record in records]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError(f"Selection requires non-empty string field {id_field}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Selection requires unique {id_field} values")
    return sorted(
        records,
        key=lambda record: (
            sha256(f"{seed}\0{record[id_field]}".encode("utf-8")).hexdigest(),
            record[id_field],
        ),
    )[:count]


def parse_acl_anthology_metadata(xml_bytes: bytes, volume_id: str = "long") -> list[dict[str, Any]]:
    root = ET.fromstring(xml_bytes)
    volume = next((node for node in root.findall("volume") if node.get("id") == volume_id), None)
    if volume is None:
        raise ValueError(f"ACL metadata does not contain volume {volume_id}")
    records: list[dict[str, Any]] = []
    for paper in volume.findall("paper"):
        paper_number = paper.get("id")
        url_node = paper.find("url")
        title_node = paper.find("title")
        if not paper_number or url_node is None or not url_node.text or title_node is None:
            raise ValueError("ACL metadata contains a paper without id, URL, or title")
        anthology_id = url_node.text.strip()
        authors: list[str] = []
        for author in paper.findall("author"):
            first = " ".join("".join(author.find("first").itertext()).split()) if author.find("first") is not None else ""
            last = " ".join("".join(author.find("last").itertext()).split()) if author.find("last") is not None else ""
            name = " ".join(part for part in (first, last) if part)
            if name:
                authors.append(name)
        records.append(
            {
                "source_id": anthology_id,
                "anthology_id": anthology_id,
                "title": normalize_plain_text("".join(title_node.itertext())),
                "authors": authors,
                "venue": "ACL 2024",
                "track": f"main-{volume_id}",
                "doi": paper.findtext("doi"),
                "source_url": f"https://aclanthology.org/{anthology_id}/",
                "pdf_url": f"https://aclanthology.org/{anthology_id}.pdf",
                "metadata_volume": f"2024.acl-{volume_id}",
                "paper_number": int(paper_number),
            }
        )
    return records


def extract_introduction(markdown_text: str) -> str:
    normalized_lines = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_lines = re.sub(r"(?<=\w)-\n(?=[a-z])", "", normalized_lines)
    lines = [
        line
        for line in normalized_lines.split("\n")
        if not _ACL_LAYOUT_ARTIFACT.match(line.strip())
    ]
    start: int | None = None
    stop: int | None = None
    heading_level: int | None = None
    for index, line in enumerate(lines):
        match = _MARKDOWN_HEADING.match(line.strip())
        if match:
            start = index + 1
            heading_level = len(match.group("marks"))
            break
    if start is not None and heading_level is not None:
        for index in range(start, len(lines)):
            match = re.match(r"^(#{1,6})\s+\S", lines[index].strip())
            if match and len(match.group(1)) <= heading_level:
                stop = index
                break
    else:
        for index, line in enumerate(lines):
            if _PLAIN_INTRO_HEADING.match(line):
                start = index + 1
                break
        if start is not None:
            for index in range(start, len(lines)):
                if _is_plain_section_heading(lines[index]) and not _PLAIN_INTRO_HEADING.match(
                    lines[index]
                ):
                    stop = index
                    break
    if start is None:
        raise ValueError("Could not locate an Introduction heading")
    introduction = normalize_plain_text("\n".join(lines[start:stop]))
    if len(introduction) < 200:
        raise ValueError("Extracted Introduction is implausibly short")
    return introduction


def adapt_nc_physics_record(record: dict[str, Any]) -> dict[str, Any]:
    required = ("unique_id", "title", "abstract", "sections", "references")
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"NC_Physics record missing fields: {', '.join(missing)}")
    sections = record["sections"]
    if not isinstance(sections, list):
        raise ValueError("NC_Physics sections must be a list")
    introductions: list[str] = []
    non_intro: dict[str, str] = {}
    for item in sections:
        if not isinstance(item, dict):
            raise ValueError("NC_Physics section entries must be objects")
        name = normalize_plain_text(str(item.get("section", "")))
        content = normalize_plain_text(str(item.get("content", "")))
        normalized_name = re.sub(r"^\d+(?:\.\d+)*[.)]?\s*", "", name).strip().casefold()
        if normalized_name == "introduction":
            introductions.append(content)
            continue
        if not name or not content:
            continue
        if "intro" in normalized_name:
            raise ValueError(f"Ambiguous Introduction-like section: {name}")
        if name in non_intro:
            non_intro[name] = f"{non_intro[name]}\n\n{content}"
        else:
            non_intro[name] = content
    if len(introductions) != 1 or not introductions[0]:
        raise ValueError("NC_Physics record must contain exactly one Introduction section")
    unique_id = str(record["unique_id"])
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", unique_id).strip("._")
    if not safe_id:
        raise ValueError("NC_Physics unique_id cannot form a safe target id")
    references = record["references"]
    if not isinstance(references, list):
        raise ValueError("NC_Physics references must be a list")
    return {
        "target_id": f"ncphysics_{safe_id}",
        "title": record["title"],
        "abstract": record["abstract"],
        "non_intro_sections": non_intro,
        "reference_metadata": references,
        "gold_introduction": introductions[0],
        "acquisition_metadata": {
            "dataset": "Xiao-Youth/NC_Physics",
            "split": "train",
            "unique_id": unique_id,
            "subfield": record.get("subfield"),
            "excluded_dataset_fields": ["core_idea", "entities"],
        },
    }


def select_nc_physics_development_records(
    records: Sequence[dict[str, Any]],
    *,
    split_name: str,
    validation_start: int,
    validation_count: int,
    target_count: int,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if split_name != "train":
        raise ValueError("Formal pilot may only select NC_Physics records from the train split")
    if validation_start < 0 or validation_count <= 0:
        raise ValueError("Validation slice must be positive and zero-based")
    validation_pool = list(records[validation_start : validation_start + validation_count])
    if len(validation_pool) != validation_count:
        raise ValueError("NC_Physics input does not contain the configured validation slice")
    selected = deterministic_select(
        validation_pool, id_field="unique_id", count=target_count, seed=seed
    )
    return validation_pool, selected


def derive_development_length_statistics(
    introductions: Iterable[str], tokenizer: Tokenizer
) -> dict[str, Any]:
    word_lengths: list[int] = []
    token_lengths: list[int] = []
    for introduction in introductions:
        if introduction and introduction.strip():
            word_lengths.append(len(_WORD.findall(introduction)))
            token_lengths.append(len(tokenizer.encode(introduction)))
    if not word_lengths:
        raise ValueError("Development length statistics require Introduction text")
    word_lengths.sort()
    token_lengths.sort()
    word_quartiles = statistics.quantiles(word_lengths, n=4, method="inclusive")
    token_quartiles = statistics.quantiles(token_lengths, n=4, method="inclusive")
    return {
        "source_scope": "nc_physics_trainval_validation_aggregate",
        "count": len(word_lengths),
        "median_words": float(statistics.median(word_lengths)),
        "p25_words": word_quartiles[0],
        "p75_words": word_quartiles[2],
        "median_tokens": float(statistics.median(token_lengths)),
        "p25_tokens": token_quartiles[0],
        "p75_tokens": token_quartiles[2],
        "tokenizer_version": tokenizer.version,
    }
