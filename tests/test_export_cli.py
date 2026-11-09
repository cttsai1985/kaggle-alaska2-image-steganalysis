"""A real CPU fixture model crosses the same prediction-to-CSV boundary as retained model outputs."""

import json

import polars as pl

from alaska2.cli import main
from alaska2.evaluate import evaluate, verify
from alaska2.train import train


def test_verified_fixture_exports_in_actual_sample_order(tmp_path, fixture_cfg, fixture_data):
    run = train(fixture_cfg(), run_dir=tmp_path / "run")
    checkpoint = run / "checkpoints/last.pt"
    prediction = evaluate(checkpoint, out_dir=tmp_path / "predictions", split="test")
    assert verify(prediction)["passed"]
    images = pl.read_parquet(prediction)["image"].to_list()[::-1]
    sample = tmp_path / "sample.csv"
    pl.DataFrame({"Id": images, "Label": [0] * len(images)}).write_csv(sample)
    output = tmp_path / "submission.csv"
    assert (
        main(
            [
                "export",
                "--predictions",
                str(prediction),
                "--checkpoint",
                str(checkpoint),
                "--sample",
                str(sample),
                "--out",
                str(output),
                "--data-root",
                str(fixture_data["root"]),
            ]
        )
        == 0
    )
    assert pl.read_csv(output)["Id"].to_list() == images
    receipt = json.loads(output.with_suffix(".json").read_text())
    assert receipt["project_id"] == "alaska2"
    assert receipt["result_label"] == "FIXTURE_NOT_A_MODEL_RESULT"
    assert receipt["submission"]["kaggle_submission"] is False
