"""Execute the teaching notebook; stage and validate every output before publishing the notebook."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from alaska2.identity import git_source, runtime_versions, sha256_file
from tools.notebooks.build_walkthrough import build_walkthrough


def _execution_summary(notebook: nbformat.NotebookNode) -> dict:
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = [item for cell in code for item in cell.outputs if item.output_type == "error"]
    images = [item for cell in code for item in cell.outputs if "image/png" in item.get("data", {})]
    if errors or not images or any(cell.execution_count is None for cell in code):
        raise ValueError("walkthrough must retain complete execution and plotted outputs")
    fixture_cell = next(cell for cell in code if cell.id == "execute-fixture")
    fixture = ast.literal_eval(
        next(item.data["text/plain"] for item in fixture_cell.outputs if item.output_type == "execute_result")
    )
    if fixture["result_label"] != "FIXTURE_NOT_A_MODEL_RESULT" or fixture["recovery_state"] != "delivered":
        raise ValueError("fixture scope or recovery result differs")
    return {
        "code_cells": len(code),
        "error_cells": len(errors),
        "image_outputs": len(images),
        "fixture": fixture,
    }


def _stage(root: Path, staging: Path) -> dict:
    notebook = build_walkthrough(root)
    env = {**os.environ, "ALASKA2_OUTPUT_ROOT": str(staging), "CUDA_VISIBLE_DEVICES": ""}
    NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        record_timing=False,
        resources={"metadata": {"path": str(root)}},
    ).execute(env=env)
    nbformat.write(notebook, staging / "executed.ipynb")
    saved = nbformat.read(staging / "executed.ipynb", as_version=4)
    summary = _execution_summary(saved)
    preview, _ = HTMLExporter().from_notebook_node(saved)
    (staging / "walkthrough.html").write_text(preview)
    figure = staging / "quality-and-method.png"
    if not figure.is_file():
        raise ValueError("walkthrough figure artifact missing")
    return {
        "kind": "executed_walkthrough",
        "measurement_status": "MEASURED",
        "executed": True,
        "notebook_sha256": sha256_file(staging / "executed.ipynb"),
        **summary,
        "figure_sha256": sha256_file(figure),
        "source": git_source(root),
        "versions": runtime_versions(),
        "sources": notebook.metadata["alaska2"]["sources"],
        "real_model_training_in_notebook": False,
        "hosted_kaggle_execution": "unverified",
    }


def execute_walkthrough(root: Path, output: Path, artifacts: Path, *, publish_figure: bool = False) -> dict:
    """Retain prior notebook/figures if execution, validation, HTML export or provenance collection fails."""
    if output.suffix != ".ipynb":
        raise ValueError("walkthrough output must be a notebook")
    root, output, artifacts = root.resolve(), output.resolve(), artifacts.resolve()
    artifacts.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="alaska2-notebook-", dir=artifacts.parent) as temporary:
        staging = Path(temporary)
        receipt = _stage(root, staging)
        artifacts.mkdir(parents=True, exist_ok=True)
        for name in ("walkthrough.html", "quality-and-method.png"):
            shutil.copyfile(staging / name, artifacts / name)
        if publish_figure:
            target = root / "docs/figures/quality-and-method.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staging / "quality-and-method.png", target)
        output.parent.mkdir(parents=True, exist_ok=True)
        pending = output.with_name(output.stem + ".executed.partial.ipynb")
        shutil.copyfile(staging / "executed.ipynb", pending)
        pending.replace(output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("notebooks/01-method-and-rerun.ipynb"))
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--publish-figure", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    result = execute_walkthrough(root, args.out, args.artifacts, publish_figure=args.publish_figure)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({key: result[key] for key in ("executed", "code_cells", "error_cells", "image_outputs")})
    )


if __name__ == "__main__":
    main()
