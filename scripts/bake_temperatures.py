"""Bake the accepted (ECE + deadband) temperatures into the saved checkpoints.

Writes ``artifacts/models/temperatures.json`` mapping each clean checkpoint name to
its fitted ``T_ece`` from the calibration run. ``load_checkpoint`` reads this sidecar
and applies the temperature automatically, so ``dialog-emo run --checkpoint X`` scores
with the calibrated T by default — no manual lookup. Non-destructive (the checkpoint
artifacts are untouched) and reversible (delete the file for raw behaviour).

    python scripts/bake_temperatures.py
    python scripts/bake_temperatures.py --column T_nll   # bake a different objective
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Write temperatures.json into the models dir.")
    parser.add_argument(
        "--csv", type=Path,
        default=Path("artifacts/experiments/calibration/temperature_calibration.csv"),
    )
    parser.add_argument("--column", default="T_ece", help="Temperature column to bake (default T_ece)")
    parser.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    table = {str(row["model"]): round(float(row[args.column]), 4) for _, row in frame.iterrows()}

    out = args.models_dir / "temperatures.json"
    out.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    applied = {k: v for k, v in sorted(table.items()) if v != 1.0}
    identity = sorted(k for k, v in table.items() if v == 1.0)
    print(f"wrote {out} ({len(table)} models, column {args.column!r})")
    print("\ncalibrated (T != 1, applied on load):")
    for name, t in applied.items():
        print(f"  {name:24} T = {t}")
    print(f"\nleft at T = 1 (no change on load): {', '.join(identity)}")


if __name__ == "__main__":
    main()
