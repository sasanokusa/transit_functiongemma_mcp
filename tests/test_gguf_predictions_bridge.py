import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import AutoProcessor  # noqa: E402

from scripts.generate_gguf_predictions import (  # noqa: E402
    LlamaServer,
    free_port,
    read_jsonl,
    rendered_tools,
    row_prompt,
    single_bos_token_count,
)
from transit_functiongemma.config import DEFAULT_SCHEMA_PATH  # noqa: E402
from transit_functiongemma.constrained_decode import END_FUNCTION_CALL, END_OF_TURN  # noqa: E402
from transit_functiongemma.infer import ToolRouter  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
MERGED_MODEL = ROOT / "gguf_work" / "merged_r8b"
F16_GGUF = ROOT / "gguf_work" / "r8b_f16.gguf"
DATASET = ROOT / "data" / "eval" / "sonnet5_holdout_60.jsonl"
BRIDGE_SAMPLE_IDS = [
    "sonnet2-hnc-003",
    "sonnet2-hnc-005",
    "sonnet2-hnc-006",
    "sonnet2-hnc-007",
    "sonnet2-hnc-008",
    "sonnet2-hnc-010",
    "sonnet2-hmp-001",
    "sonnet2-hmp-003",
    "sonnet2-hmp-004",
    "sonnet2-hmp-006",
    "sonnet2-hmp-007",
    "sonnet2-hmp-008",
    "sonnet2-hmp-009",
    "sonnet2-hmp-021",
    "sonnet2-hmp-022",
    "sonnet2-hmp-023",
    "sonnet2-hmp-024",
    "sonnet2-hmp-025",
    "sonnet2-hmp-026",
    "sonnet2-hmp-027",
]


def sample_mixed_rows(limit: int = 20):
    rows_by_id = {row["id"]: row for row in read_jsonl(DATASET)}
    return [rows_by_id[row_id] for row_id in BRIDGE_SAMPLE_IDS[:limit]]


def stop_like_llama_server(text: str) -> str:
    candidates = []
    end_call = text.find(END_FUNCTION_CALL)
    if end_call >= 0:
        candidates.append((end_call, end_call + len(END_FUNCTION_CALL)))
    end_turn = text.find(END_OF_TURN)
    if end_turn >= 0:
        candidates.append((end_turn, end_turn))
    if not candidates:
        return text
    _start, stop_end = min(candidates, key=lambda item: item[0])
    return text[:stop_end]


def require_bridge_assets():
    server = shutil.which("llama-server") or "/opt/homebrew/bin/llama-server"
    missing = [
        str(path)
        for path in (MERGED_MODEL, F16_GGUF, DATASET)
        if not path.exists()
    ]
    if not Path(server).exists():
        missing.append("llama-server")
    if missing:
        pytest.skip(f"GGUF bridge assets unavailable: {', '.join(missing)}")
    return server


def test_gguf_f16_bridge_matches_pytorch_greedy_outputs():
    server_executable = require_bridge_assets()
    rows = sample_mixed_rows(20)
    processor = AutoProcessor.from_pretrained(MERGED_MODEL, local_files_only=True)
    tools = rendered_tools(DEFAULT_SCHEMA_PATH, "baked", False)
    prompts = [
        row_prompt(processor, tools, row, normalize_ja=False)
        for row in rows
    ]
    assert all(single_bos_token_count(processor, prompt) == 1 for prompt in prompts)

    router = ToolRouter(
        base_model=str(MERGED_MODEL),
        schema_mode="baked",
        clarification_tool=False,
        normalize_ja=False,
    )
    pytorch_outputs = []
    for row in rows:
        prompt = row.get("user") if isinstance(row.get("user"), str) else None
        pytorch_outputs.append(
            stop_like_llama_server(
                router.generate(
                    prompt,
                    row.get("reference_datetime"),
                    max_new_tokens=128,
                    history=row.get("history"),
                )
            )
        )

    mismatches = []
    with LlamaServer(
        executable=server_executable,
        gguf=F16_GGUF,
        host="127.0.0.1",
        port=free_port(),
        ctx_size=2048,
        threads=4,
        gpu_layers=0,
        extra_args=[],
    ) as server:
        bos_token_id = int(processor.bos_token_id)
        assert server.tokenize(prompts[0]).count(bos_token_id) == 1
        gguf_outputs = [
            server.complete(
                prompt,
                max_new_tokens=128,
                stop=[END_FUNCTION_CALL, END_OF_TURN],
                timeout=120,
            )
            for prompt in prompts
        ]

    for row, pytorch_output, gguf_output in zip(rows, pytorch_outputs, gguf_outputs):
        if pytorch_output != gguf_output:
            mismatches.append(
                f"{row['id']}\nPyTorch: {pytorch_output}\nGGUF:    {gguf_output}"
            )
    exact_rate = (len(rows) - len(mismatches)) / len(rows)
    assert exact_rate >= 0.95, (
        f"GGUF bridge exact match rate {exact_rate:.1%} < 95%.\n"
        + "\n\n".join(mismatches)
    )
