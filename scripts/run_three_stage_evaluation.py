#!/usr/bin/env python3
"""Run raw-model, notation-normalized, and final live-pipeline evaluations."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("metrics") or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument(
        "--intent-dataset",
        type=Path,
        default=Path("data/eval/operational_intent_raw_100.jsonl"),
    )
    parser.add_argument(
        "--final-dataset",
        type=Path,
        default=Path("data/eval/operational_tokyo_routes_100.jsonl"),
    )
    parser.add_argument("--url", default="http://127.0.0.1:8091/query")
    parser.add_argument(
        "--behavior-log-dir", type=Path, default=Path("artifacts/behavior_logs")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_json = args.output_dir / "operational_intent_raw_model.json"
    normalized_json = args.output_dir / "operational_intent_normalized.json"
    final_json = args.output_dir / "operational_tokyo_routes_100_final.json"

    common = [
        sys.executable,
        "eval_toolcall.py",
        "--dataset",
        str(args.intent_dataset),
        "--run-model",
        "--adapter",
        args.adapter,
        "--clarification-tool",
    ]
    run(
        common
        + [
            "--output",
            str(raw_json),
            "--markdown-output",
            str(args.output_dir / "operational_intent_raw_model.md"),
            "--failures-output",
            str(args.output_dir / "failures_operational_intent_raw_model.jsonl"),
        ]
    )
    run(
        common
        + [
            "--normalize-ja",
            "--schema-constraint",
            "--output",
            str(normalized_json),
            "--markdown-output",
            str(args.output_dir / "operational_intent_normalized.md"),
            "--failures-output",
            str(args.output_dir / "failures_operational_intent_normalized.jsonl"),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/run_operational_samples.py",
            "--url",
            args.url,
            "--dataset",
            str(args.final_dataset),
            "--behavior-log-dir",
            str(args.behavior_log_dir),
            "--output",
            str(final_json),
            "--markdown-output",
            str(args.output_dir / "operational_tokyo_routes_100_final.md"),
            "--timeout",
            str(args.timeout),
            "--delay",
            str(args.delay),
        ]
    )

    summary = {
        "raw_model": metrics(raw_json),
        "normalized": metrics(normalized_json),
        "final_pipeline": metrics(final_json),
        "semantic_fallback": False,
    }
    summary_path = args.output_dir / "operational_three_stage_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = [
        "# Operational three-stage evaluation",
        "",
        "| Stage | Metrics |",
        "|---|---|",
        f"| raw model | `{json.dumps(summary['raw_model'], ensure_ascii=False)}` |",
        f"| normalized | `{json.dumps(summary['normalized'], ensure_ascii=False)}` |",
        f"| final pipeline | `{json.dumps(summary['final_pipeline'], ensure_ascii=False)}` |",
        "",
        "Semantic fallback: disabled.",
    ]
    (args.output_dir / "operational_three_stage_summary.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
