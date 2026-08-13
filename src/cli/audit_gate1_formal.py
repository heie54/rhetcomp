from __future__ import annotations

import argparse
from pathlib import Path

from src.audits.gate1_formal import write_gate1_formal_report


ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the real-data Stage 1R gate audit")
    parser.add_argument("--dataset-config", default="configs/dataset_formal_pilot.yaml")
    parser.add_argument("--budget-config", default="configs/budget_formal.yaml")
    parser.add_argument(
        "--output",
        default="artifacts/formal_pilot/formal-pilot-v1/audits/gate1_formal.json",
    )
    args = parser.parse_args(argv)
    report = write_gate1_formal_report(
        ROOT,
        (ROOT / args.dataset_config).resolve(),
        (ROOT / args.budget_config).resolve(),
        (ROOT / args.output).resolve(),
    )
    print(f"GATE_1R={report['status']}")
    if report["status"] == "BLOCKED":
        for blocker in report["blockers"]:
            print(f"BLOCKER={blocker['path']}")
        return 2
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
