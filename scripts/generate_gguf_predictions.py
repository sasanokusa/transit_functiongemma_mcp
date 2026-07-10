#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoProcessor

from transit_functiongemma.config import DEFAULT_SCHEMA_PATH
from transit_functiongemma.constrained_decode import END_FUNCTION_CALL, END_OF_TURN
from transit_functiongemma.infer import (
    build_router_messages,
    render_router_prompt,
)
from transit_functiongemma.schemas import (
    compact_functiongemma_tools,
    load_mcp_tools,
    tools_with_clarification,
    to_functiongemma_tools,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def infer_tokenizer_path(gguf: Path) -> Path:
    candidates = [
        gguf.parent / "merged_r8b",
        gguf.parent / "hf_cache",
        gguf.parent,
    ]
    for candidate in candidates:
        if (candidate / "tokenizer.json").exists() or (
            candidate / "tokenizer.model"
        ).exists():
            return candidate
    raise FileNotFoundError(
        "Could not infer tokenizer path; pass --tokenizer explicitly."
    )


def rendered_tools(
    schema_path: Path,
    schema_mode: str,
    clarification_tool: bool,
) -> list[dict[str, Any]]:
    mcp_tools = tools_with_clarification(
        load_mcp_tools(schema_path), clarification_tool
    )
    if schema_mode == "full":
        return to_functiongemma_tools(mcp_tools)
    if schema_mode == "compact":
        return compact_functiongemma_tools(mcp_tools)
    return []


def row_prompt(
    processor: Any,
    tools: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    normalize_ja: bool,
) -> str:
    prompt = (
        row.get("user")
        if isinstance(row.get("user"), str) and row.get("user").strip()
        else None
    )
    messages = build_router_messages(
        prompt,
        row.get("reference_datetime"),
        row.get("history"),
        normalize_ja=normalize_ja,
    )
    return render_router_prompt(processor, messages, tools)


def token_ids_for_prompt(processor: Any, prompt: str) -> list[int]:
    encoded = processor(prompt, add_special_tokens=False)
    return [int(token_id) for token_id in encoded["input_ids"]]


def single_bos_token_count(processor: Any, prompt: str) -> int:
    bos_token_id = getattr(processor, "bos_token_id", None)
    if bos_token_id is None:
        return 0
    return token_ids_for_prompt(processor, prompt).count(int(bos_token_id))


class LlamaServer:
    def __init__(
        self,
        *,
        executable: str,
        gguf: Path,
        host: str,
        port: int,
        ctx_size: int,
        threads: int,
        gpu_layers: int,
        extra_args: list[str],
    ) -> None:
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.process: subprocess.Popen[str] | None = None
        self.command = [
            executable,
            "--model",
            str(gguf),
            "--host",
            host,
            "--port",
            str(port),
            "--no-webui",
            "--parallel",
            "1",
            "--ctx-size",
            str(ctx_size),
            "--n-gpu-layers",
            str(gpu_layers),
            "--override-kv",
            "tokenizer.ggml.add_bos_token=bool:false",
            "--log-disable",
            *extra_args,
        ]
        if threads > 0:
            self.command.extend(["--threads", str(threads)])

    def __enter__(self) -> "LlamaServer":
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.wait_ready()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)

    def wait_ready(self, timeout_seconds: float = 120.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(f"llama-server exited early: {stderr[-4000:]}")
            try:
                with urllib.request.urlopen(
                    f"{self.base_url}/health", timeout=2
                ) as response:
                    if response.status == 200:
                        return
            except Exception as exc:
                last_error = exc
            time.sleep(0.25)
        raise TimeoutError(f"llama-server did not become healthy: {last_error}")

    def post_json(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server HTTP {exc.code}: {detail}") from exc

    def tokenize(self, prompt: str) -> list[int]:
        data = self.post_json(
            "/tokenize",
            {"content": prompt, "add_special": False},
            timeout=30,
        )
        return [int(token_id) for token_id in data.get("tokens", [])]

    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        stop: list[str],
        timeout: float,
    ) -> str:
        data = self.post_json(
            "/completion",
            {
                "prompt": prompt,
                "n_predict": max_new_tokens,
                "temperature": 0,
                "seed": 0,
                "cache_prompt": False,
                "stream": False,
                "stop": stop,
            },
            timeout=timeout,
        )
        content = str(data.get("content", ""))
        stopping_word = data.get("stopping_word")
        if stopping_word == END_FUNCTION_CALL and not content.endswith(
            END_FUNCTION_CALL
        ):
            content += END_FUNCTION_CALL
        return content


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate eval_toolcall.py predictions with a llama.cpp GGUF model."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument(
        "--schema-mode", choices=("baked", "compact", "full"), default="baked"
    )
    parser.add_argument("--clarification-tool", action="store_true")
    parser.add_argument("--normalize-ja", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=max(os.cpu_count() or 1, 1))
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--llama-server",
        default=shutil.which("llama-server") or "/opt/homebrew/bin/llama-server",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Extra llama-server argument; repeat for multiple arguments.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    gguf = args.gguf.resolve()
    tokenizer_path = args.tokenizer or infer_tokenizer_path(gguf)
    processor = AutoProcessor.from_pretrained(tokenizer_path, local_files_only=True)
    tools = rendered_tools(args.schema, args.schema_mode, args.clarification_tool)
    rows = read_jsonl(args.dataset)
    prompts = [
        row_prompt(processor, tools, row, normalize_ja=args.normalize_ja)
        for row in rows
    ]
    if prompts:
        bos_count = single_bos_token_count(processor, prompts[0])
        if bos_count != 1:
            raise RuntimeError(
                f"Rendered prompt must contain exactly one BOS token; got {bos_count}."
            )

    server = LlamaServer(
        executable=args.llama_server,
        gguf=gguf,
        host=args.host,
        port=args.port or free_port(),
        ctx_size=args.ctx_size,
        threads=args.threads,
        gpu_layers=args.gpu_layers,
        extra_args=args.server_arg,
    )
    stop = [END_FUNCTION_CALL, END_OF_TURN]
    predictions: list[dict[str, str]] = []
    with server:
        if prompts:
            bos_token_id = getattr(processor, "bos_token_id", None)
            server_bos_count = (
                server.tokenize(prompts[0]).count(int(bos_token_id))
                if bos_token_id is not None
                else 0
            )
            if server_bos_count != 1:
                raise RuntimeError(
                    "llama-server prompt tokenization must contain exactly one "
                    f"BOS token; got {server_bos_count}."
                )
        for index, (row, prompt) in enumerate(zip(rows, prompts), start=1):
            output = server.complete(
                prompt,
                max_new_tokens=args.max_new_tokens,
                stop=stop,
                timeout=args.timeout,
            )
            predictions.append({"id": str(row["id"]), "model_output": output})
            print(
                f"[{index}/{len(rows)}] {row['id']}: {output[:120]}",
                file=sys.stderr,
                flush=True,
            )
    write_jsonl(args.output, predictions)
    print(f"Wrote predictions: {args.output}")


if __name__ == "__main__":
    main()
