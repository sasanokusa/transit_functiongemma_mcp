#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.eval_toolcall import evaluate
from scripts.convert_intent_router_eval import convert as convert_intent_row
from transit_functiongemma.config import DEFAULT_SCHEMA_PATH, MODEL_ID
from transit_functiongemma.schemas import load_mcp_tools


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    if "expected_tool" in row:
        return row
    if "assistant" in row:
        return convert_intent_row(row)
    raise ValueError(f"{row.get('id', '<unknown>')}: unsupported dev row shape")


def sample_rows(rows: list[dict[str, Any]], sample_size: int | None, seed: int) -> list[dict[str, Any]]:
    if not sample_size or sample_size >= len(rows):
        return rows
    indexes = sorted(random.Random(seed).sample(range(len(rows)), sample_size))
    return [rows[index] for index in indexes]


def epoch_dirs(output_dir: Path) -> list[Path]:
    def key(path: Path) -> tuple[int, str]:
        match = re.fullmatch(r"epoch-(\d+)", path.name)
        return (int(match.group(1)) if match else 10**9, path.name)

    return sorted(
        [path for path in output_dir.glob("epoch-*") if path.is_dir()],
        key=key,
    )


def render_value(value: Any) -> str:
    if isinstance(value, str):
        return f"<escape>{value}<escape>"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + ",".join(render_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{render_value(item)}" for key, item in value.items()) + "}"
    return str(value)


def render_tool_call(tool_name: str, arguments: dict[str, Any]) -> str:
    body = ",".join(f"{key}:{render_value(value)}" for key, value in arguments.items())
    return f"<start_function_call>call:{tool_name}{{{body}}}<end_function_call>"


def mock_predictions(rows: list[dict[str, Any]]) -> dict[str, str]:
    predictions: dict[str, str] = {}
    for row in rows:
        tool = row.get("expected_tool")
        if not tool:
            predictions[row["id"]] = "<end_of_turn>"
            continue
        predictions[row["id"]] = render_tool_call(tool, row.get("expected_arguments") or {})
    return predictions


def evaluate_dry_run(rows: list[dict[str, Any]], schema: Path, clarification_tool: bool) -> dict[str, Any]:
    return evaluate(
        rows,
        mock_predictions(rows),
        load_mcp_tools(schema),
        clarification_tool=clarification_tool,
        schema_constraint=True,
        bind_normalized_arguments=True,
    )


def run_epoch_eval(
    rows_path: Path,
    epoch_dir: Path,
    output_json: Path,
    output_md: Path,
    failures_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "evaluation/eval_toolcall.py",
        "--dataset",
        str(rows_path),
        "--run-model",
        "--base-model",
        args.base_model,
        "--adapter",
        str(epoch_dir),
        "--schema",
        str(args.schema),
        "--clarification-tool",
        "--normalize-ja",
        "--schema-constraint",
        "--bind-normalized-arguments",
        "--output",
        str(output_json),
        "--markdown-output",
        str(output_md),
        "--failures-output",
        str(failures_path),
    ]
    if args.constrained_decode:
        command.append("--constrained-decode")
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(output_json.read_text(encoding="utf-8"))


def metric_value(report: dict[str, Any], key: str) -> float:
    value = report.get("metrics", {}).get(key)
    return float(value) if isinstance(value, (int, float)) else -1.0


def report_markdown(run_name: str, rows: list[dict[str, Any]], results: list[dict[str, Any]], dry_run: bool) -> str:
    lines = [
        f"# Checkpoint selection: {run_name}",
        "",
        f"- Dev rows: {len(rows)}",
        f"- Dry run: {dry_run}",
        "- Selection metric: semantic_success_rate",
        "",
        "| Epoch | Semantic | Expected args | Tool | Datetime | Report |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in results:
        metrics = item["report"].get("metrics", {})
        lines.append(
            "| {epoch} | {semantic} | {expected_args} | {tool} | {datetime} | {report_path} |".format(
                epoch=item["epoch"],
                semantic=metrics.get("semantic_success_rate"),
                expected_args=metrics.get("expected_arguments_match_rate"),
                tool=metrics.get("tool_name_accuracy"),
                datetime=metrics.get("datetime_normalization_success_rate"),
                report_path=item.get("markdown_path", ""),
            )
        )
    best = max(results, key=lambda item: metric_value(item["report"], "semantic_success_rate"))
    lines.extend(["", f"Best epoch: `{best['epoch']}`"])
    if dry_run:
        lines.append("Dry run did not create or update the `best` symlink.")
    else:
        lines.append(f"Best symlink: `{best['best_symlink']}`")
    lines.append("")
    return "\n".join(lines)


def update_best_symlink(output_dir: Path, best_epoch_dir: Path) -> Path:
    link = output_dir / "best"
    if link.exists() or link.is_symlink():
        if link.is_dir() and not link.is_symlink():
            raise FileExistsError(f"refusing to replace non-symlink directory: {link}")
        link.unlink()
    link.symlink_to(best_epoch_dir.name, target_is_directory=True)
    return link


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best epoch adapter by external dev metrics.")
    parser.add_argument("--run", type=Path, required=True, help="Output run directory containing epoch-N adapters.")
    parser.add_argument("--dev", type=Path, default=Path("data/eval/intent_router_dev_950.jsonl"))
    parser.add_argument("--dev-sample", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--base-model", default=MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--constrained-decode", action="store_true")
    args = parser.parse_args()

    run_dir = args.run
    epochs = epoch_dirs(run_dir)
    if not epochs:
        raise SystemExit(f"no epoch-N adapter directories found under {run_dir}")

    rows = sample_rows([to_eval_row(row) for row in read_jsonl(args.dev)], args.dev_sample, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_dir.name
    dev_path = args.output_dir / f"checkpoint_selection_{run_name}_dev.jsonl"
    write_jsonl(dev_path, rows)

    results: list[dict[str, Any]] = []
    for epoch_dir in epochs:
        epoch = epoch_dir.name
        output_json = args.output_dir / f"checkpoint_selection_{run_name}_{epoch}.json"
        output_md = args.output_dir / f"checkpoint_selection_{run_name}_{epoch}.md"
        failures_path = args.output_dir / f"checkpoint_selection_{run_name}_{epoch}_failures.jsonl"
        if args.dry_run:
            report = evaluate_dry_run(rows, args.schema, clarification_tool=True)
            output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            output_md.write_text("# Dry-run mock evaluation\n", encoding="utf-8")
        else:
            report = run_epoch_eval(dev_path, epoch_dir, output_json, output_md, failures_path, args)
        results.append(
            {
                "epoch": epoch,
                "epoch_dir": str(epoch_dir),
                "report": report,
                "json_path": str(output_json),
                "markdown_path": str(output_md),
            }
        )

    best = max(results, key=lambda item: metric_value(item["report"], "semantic_success_rate"))
    if not args.dry_run:
        link = update_best_symlink(run_dir, Path(best["epoch_dir"]))
        best["best_symlink"] = str(link)
    selection_md = args.output_dir / f"checkpoint_selection_{run_name}.md"
    selection_json = args.output_dir / f"checkpoint_selection_{run_name}.json"
    selection_md.write_text(report_markdown(run_name, rows, results, args.dry_run), encoding="utf-8")
    selection_json.write_text(
        json.dumps({"run": str(run_dir), "best_epoch": best["epoch"], "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"selection report: {selection_md}")
    print(f"best epoch: {best['epoch']}")


if __name__ == "__main__":
    main()
