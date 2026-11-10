"""The walkthrough is generated from reviewed receipts and executes only an explicitly labeled CPU fixture."""

from pathlib import Path

from tools.notebooks.build_walkthrough import build_walkthrough


def test_walkthrough_has_stable_cells_and_both_decision_stories():
    root = Path(__file__).resolve().parents[1]
    first, second = build_walkthrough(root), build_walkthrough(root)
    assert [cell.id for cell in first.cells] == [cell.id for cell in second.cells]
    text = "\n".join(cell.source for cell in first.cells)
    assert "FIXTURE_NOT_A_MODEL_RESULT" in text
    assert "quality" in text and "recovery" in text
    assert set(first.metadata["alaska2"]["sources"]) == {"evaluation", "delivery", "split"}
    assert first.metadata["alaska2"]["hosted_kaggle_execution"] == "unverified"
    assert first.metadata["alaska2"]["real_model_training_in_notebook"] is False


def test_walkthrough_executes_with_saved_outputs_and_explicit_fixture_scope(tmp_path):
    import nbformat

    from tools.notebooks.execute_walkthrough import execute_walkthrough

    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "walkthrough.ipynb"
    receipt = execute_walkthrough(root, output, tmp_path / "artifacts")
    saved = nbformat.read(output, as_version=4)
    assert receipt["executed"] is True and receipt["error_cells"] == 0
    assert receipt["image_outputs"] >= 1
    assert receipt["fixture"]["result_label"] == "FIXTURE_NOT_A_MODEL_RESULT"
    assert receipt["fixture"]["recovery_state"] == "delivered"
    assert all(cell.execution_count is not None for cell in saved.cells if cell.cell_type == "code")


def test_kaggle_entry_remains_explicitly_unverified_and_checks_offline_source():
    from tools.notebooks.kaggle_entry import build_entry

    notebook = build_entry()
    assert notebook.metadata["alaska2"]["hosted_kaggle_execution"] == "unverified"
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "source checksum mismatch" in source
    assert "tools.check_locked_runtime" in source
    assert 'SOURCE / ".venv/bin/python"' in source
    assert "tools.notebooks.execute_walkthrough" in source


def test_failed_export_preserves_previous_notebook(tmp_path, monkeypatch):
    import pytest

    import tools.notebooks.execute_walkthrough as runner

    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "previous.ipynb"
    previous = b"last known good notebook"
    output.write_bytes(previous)

    def failed_export(*args, **kwargs):
        raise RuntimeError("controlled HTML export failure")

    monkeypatch.setattr(runner.HTMLExporter, "from_notebook_node", failed_export)
    with pytest.raises(RuntimeError, match="controlled HTML"):
        runner.execute_walkthrough(root, output, tmp_path / "artifacts")
    assert output.read_bytes() == previous
