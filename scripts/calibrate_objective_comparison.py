"""Fit temperature by ECE vs by NLL, report ECE and KL for every checkpoint.

The main pipeline selects T by ECE. The textbook choice (Guo et al., 2017) is to
fit T by NLL (a smooth, proper scoring rule) and only *report* ECE. This script runs
both objectives on the same held-out protocol so the thesis can compare them:

* T is fit per model on RuGo validation, by ECE and by NLL (free fit, no deadband,
  so the comparison isolates the objective).
* Both ECE and KL are reported on held-out RuGo test and native CEDR test, under
  raw (T=1), the ECE-fit T, and the NLL-fit T.

ECE = top-label calibration (the timeline's confidence bar); KL = soft-distribution
faithfulness (the full 7-way emotion vector). The question is which objective gives
the better test ECE / KL, and where they disagree.

    python scripts/calibrate_objective_comparison.py
    python scripts/calibrate_objective_comparison.py --only logreg-char,fasttext --skip-heavy
"""

from __future__ import annotations

import os

os.environ.setdefault("EMO_SCHEME", "7")  # checkpoints are 7-class

import argparse
import gc
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from dialog_emo_models.checkpoints import available_checkpoints, load_checkpoint
from dialog_emo_models.datasets import load_cedr, load_rugoemotions
from dialog_emo_models.metrics import apply_temperature, best_temperature, quality_metrics

CEDR_URL = ("https://huggingface.co/datasets/sagteam/cedr_v1/resolve/"
            "refs%2Fconvert%2Fparquet/main/test/0000.parquet")
HEAVY = {
    "rubert-tiny2-finetune", "hf-seara-rubert-tiny2",
    "hf-fyaronskiy-deberta", "hf-maxkazak-rubert-base", "tree-rf",
}
COLUMNS = ["model", "kind", "T_ece", "T_nll"] + [
    f"{dom}_{metric}_{fit}"
    for dom in ("rugo", "cedr")
    for metric in ("ece", "kl")
    for fit in ("raw", "Tece", "Tnll")
]


def _ece_kl(logits: np.ndarray, y: np.ndarray, t: float) -> tuple[float, float]:
    m = quality_metrics(y, apply_temperature(logits, t))
    return round(m["ece"], 4), round(m["kl"], 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ECE-fit vs NLL-fit temperature.")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/datasets/ru_go_emotions/simplified"))
    parser.add_argument("--cedr-dir", type=Path, default=Path("artifacts/datasets/cedr"))
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/calibration"))
    parser.add_argument("--skip-heavy", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cedr_dir.mkdir(parents=True, exist_ok=True)

    rval_x, rval_y = load_rugoemotions(args.data_dir / "validation.parquet")
    rtest_x, rtest_y = load_rugoemotions(args.data_dir / "test.parquet")
    cedr_path = args.cedr_dir / "test.parquet"
    if not cedr_path.exists():
        urlretrieve(CEDR_URL, cedr_path)
    ctest_x, ctest_y = load_cedr(cedr_path)
    print(f"val={len(rval_x)} rugo_test={len(rtest_x)} cedr_test={len(ctest_x)}", flush=True)

    checkpoints = available_checkpoints(args.models_dir)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        checkpoints = {n: p for n, p in checkpoints.items() if n in wanted}
    elif args.skip_heavy:
        checkpoints = {n: p for n, p in checkpoints.items() if n not in HEAVY}
    order = sorted(checkpoints, key=lambda n: (n in HEAVY, n))
    print(f"checkpoints: {', '.join(order)}", flush=True)

    rows: list[dict] = []
    for name in order:
        print(f"RUN {name}", flush=True)
        try:
            model = load_checkpoint(name, models_dir=args.models_dir, apply_temperature=False)
            if hasattr(model, "temperature"):
                model.temperature = 1.0
            vl = np.asarray(model.predict_logits(rval_x), dtype=float)
            rl = np.asarray(model.predict_logits(rtest_x), dtype=float)
            cl = np.asarray(model.predict_logits(ctest_x), dtype=float)
            del model
            gc.collect()

            t_ece, _ = best_temperature(vl, rval_y, objective="ece")  # free fit
            t_nll, _ = best_temperature(vl, rval_y, objective="nll")
            row = {"model": name, "kind": "heavy" if name in HEAVY else "light",
                   "T_ece": round(t_ece, 3), "T_nll": round(t_nll, 3)}
            for dom, (lg, y) in (("rugo", (rl, rtest_y)), ("cedr", (cl, ctest_y))):
                for fit, t in (("raw", 1.0), ("Tece", t_ece), ("Tnll", t_nll)):
                    ece, kl = _ece_kl(lg, y, t)
                    row[f"{dom}_ece_{fit}"] = ece
                    row[f"{dom}_kl_{fit}"] = kl
            rows.append(row)
            print(f"  T_ece={t_ece:.2f} T_nll={t_nll:.2f} | rugo ECE {row['rugo_ece_raw']}"
                  f"/{row['rugo_ece_Tece']}/{row['rugo_ece_Tnll']} (raw/ece/nll) | "
                  f"cedr ECE {row['cedr_ece_raw']}/{row['cedr_ece_Tece']}/{row['cedr_ece_Tnll']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {name}: {type(exc).__name__}: {exc}", flush=True)

    frame = pd.DataFrame(rows).reindex(columns=COLUMNS).sort_values("rugo_ece_raw", ascending=False)
    csv_path = args.output_dir / "calibration_objective_comparison.csv"
    frame.to_csv(csv_path, index=False)
    pd.set_option("display.width", 220)
    for dom, title in (("rugo", "RuGo test"), ("cedr", "CEDR test")):
        cols = ["model", "T_ece", "T_nll",
                f"{dom}_ece_raw", f"{dom}_ece_Tece", f"{dom}_ece_Tnll",
                f"{dom}_kl_raw", f"{dom}_kl_Tece", f"{dom}_kl_Tnll"]
        print(f"\n=== {title}: ECE then KL under raw / T_ece / T_nll ===")
        print(frame[cols].to_string(index=False))
    print(f"\nARTIFACTS {csv_path}")


if __name__ == "__main__":
    main()
