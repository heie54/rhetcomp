from __future__ import annotations

import argparse
from pathlib import Path

from src.audits.gate0_formal import write_gate0_formal_report


ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the static Stage 0R gate audit")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "artifacts"
        / "formal_pilot"
        / "gate0r-bootstrap"
        / "audits"
        / "gate0_formal.json",
    )
    args = parser.parse_args(argv)
    report = write_gate0_formal_report(ROOT, args.output)
    print(f"GATE_0R={report['status']}")
    print(f"AUDIT_PATH={args.output}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
