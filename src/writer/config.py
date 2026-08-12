from __future__ import annotations

from dataclasses import dataclass

from src.config import load_config

WRITER_CONDITIONS = ("evidence_only", "raw", "summary", "guideline", "experience")


@dataclass(frozen=True, slots=True)
class WriterSettings:
    config_version: str
    model: str | None
    temperature: float
    top_p: float
    max_output_tokens: int
    seed: int | None
    citation_format: str
    system_prompt_version: str
    task_prompt_version: str
    desired_introduction_length: int


def load_writer_settings(writer_config_path: str) -> WriterSettings:
    config = load_config(writer_config_path)
    return WriterSettings(
        config_version=config["config_version"],
        model=config.get("model"),
        temperature=float(config["temperature"]),
        top_p=float(config["top_p"]),
        max_output_tokens=int(config["max_output_tokens"]),
        seed=config.get("seed"),
        citation_format=str(config["citation_format"]),
        system_prompt_version=config["system_prompt_version"],
        task_prompt_version=config["task_prompt_version"],
        desired_introduction_length=int(config["desired_introduction_length"]),
    )
