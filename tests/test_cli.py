from __future__ import annotations

from dialog_emo_models.cli import main
from dialog_emo_models.schema import load_full_csv, load_parsed_csv


def test_run_command_writes_parsed_and_scored_csv(tmp_path) -> None:
    parsed_path = tmp_path / "parsed.csv"
    scored_path = tmp_path / "scored.csv"

    main(
        [
            "run",
            "--input",
            "data/result.json",
            "--parsed-output",
            str(parsed_path),
            "--output",
            str(scored_path),
            "--model",
            "dummy",
        ]
    )

    assert len(load_parsed_csv(parsed_path)) == 20
    assert len(load_full_csv(scored_path)) == 20
