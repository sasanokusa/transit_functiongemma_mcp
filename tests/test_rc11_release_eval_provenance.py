import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "run_rc11_release_eval.sh"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _environment(tmp_path: Path) -> tuple[dict[str, str], dict[str, Path]]:
    root = tmp_path / "root"
    venv = tmp_path / "venv"
    run = tmp_path / "run"
    output = tmp_path / "output"
    base = tmp_path / "base"
    gguf = tmp_path / "gguf"
    llama = tmp_path / "llama"

    _write(venv / "bin" / "activate", f'export PATH="{venv / "bin"}:$PATH"\n')
    (venv / "bin" / "python").symlink_to(sys.executable)
    files = {
        "runner": root / "scripts" / "run_rc11_release_eval.sh",
        "evaluator": root / "evaluation" / "eval_toolcall.py",
        "generator": root / "scripts" / "generate_gguf_predictions.py",
        "route_intent": root / "transit_functiongemma" / "route_intent.py",
        "dataset": root / "data" / "eval" / "mixed_dev_selection.jsonl",
        "v1_model": gguf / "v100_Q6_K.gguf",
    }
    _write(files["runner"], SCRIPT.read_text())
    _write(files["evaluator"], "# evaluator v1\n")
    _write(files["generator"], "# generator v1\n")
    _write(root / "transit_functiongemma" / "__init__.py", "# package\n")
    _write(files["route_intent"], "# route intent v1\n")
    _write(files["dataset"], '{"id": "mixed"}\n')
    _write(root / "data" / "eval" / "independent_holdout_300.jsonl", '{"id": "independent"}\n')
    _write(root / "data" / "eval" / "manual_practical_100.jsonl", '{"id": "manual"}\n')
    _write(root / "data" / "eval" / "operational_semantic_holdout_300_eval.jsonl", '{"id": "route"}\n')
    _write(root / "data" / "eval" / "mixed_dev_schema.json", "{}\n")
    _write(root / "data" / "tool_schema.json", "{}\n")
    _write(root / "tools" / "local_tools_schema.json", "{}\n")

    _write(base / "model.safetensors", "base model\n")
    for epoch in (1, 2, 3):
        _write(run / f"epoch-{epoch}" / "adapter_model.safetensors", f"adapter {epoch}\n")
        _write(run / f"epoch-{epoch}" / "adapter_config.json", "{}\n")
    _write(files["v1_model"], "v1 q6 model\n")
    _write(gguf / "merged_v100" / "tokenizer.json", "{}\n")
    _write(llama / "convert_hf_to_gguf.py", "# converter\n")
    _write(llama / "build-server-noui" / "bin" / "llama-server", "server\n")
    _write(llama / "build" / "bin" / "llama-quantize", "quantize\n")

    environment = os.environ.copy()
    environment.update(
        {
            "ROOT": str(root),
            "VENV": str(venv),
            "RUN": str(run),
            "OUT": str(output),
            "BASE_SNAPSHOT": str(base),
            "GGUF_DIR": str(gguf),
            "LLAMA_ROOT": str(llama),
            "PROVENANCE_ONLY": "1",
            "RESUME": "0",
        }
    )
    environment["EVALUATION_RUNNER"] = str(files["runner"])
    return environment, files


def _run(environment: dict[str, str], *, resume: bool) -> subprocess.CompletedProcess[str]:
    run_environment = environment.copy()
    run_environment["RESUME"] = "1" if resume else "0"
    return subprocess.run(
        ["bash", environment["EVALUATION_RUNNER"]],
        env=run_environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_fresh_run_requires_empty_output_and_matching_resume(tmp_path: Path) -> None:
    environment, _ = _environment(tmp_path)

    fresh = _run(environment, resume=False)
    assert fresh.returncode == 0, fresh.stderr
    provenance_path = Path(environment["OUT"]) / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    assert provenance["hash_policy"] == "full-content-sha256"
    assert {item["name"] for item in provenance["inputs"]} >= {
        "evaluation_runner",
        "evaluator",
        "evaluation_runtime_sources",
        "dataset_mixeddev",
        "schema_mixeddev_manual100",
        "adapter_epoch_1",
        "v1_q6_model",
    }

    accidental_fresh = _run(environment, resume=False)
    assert accidental_fresh.returncode != 0
    assert "OUT is not empty" in accidental_fresh.stderr

    matching_resume = _run(environment, resume=True)
    assert matching_resume.returncode == 0, matching_resume.stderr


@pytest.mark.parametrize(
    ("changed_file", "expected_label"),
    (
        ("evaluator", "evaluator"),
        ("route_intent", "evaluation_runtime_sources"),
        ("runner", "evaluation_runner"),
        ("dataset", "dataset_mixeddev"),
        ("v1_model", "v1_q6_model"),
    ),
)
def test_resume_fails_closed_on_provenance_mismatch(
    tmp_path: Path, changed_file: str, expected_label: str
) -> None:
    environment, files = _environment(tmp_path)
    fresh = _run(environment, resume=False)
    assert fresh.returncode == 0, fresh.stderr
    original_provenance = (Path(environment["OUT"]) / "provenance.json").read_text()

    files[changed_file].write_text(files[changed_file].read_text() + "changed\n")
    resume = _run(environment, resume=True)

    assert resume.returncode != 0
    assert "provenance mismatch" in resume.stderr
    assert f"changed: {expected_label}" in resume.stderr
    assert (Path(environment["OUT"]) / "provenance.json").read_text() == original_provenance
