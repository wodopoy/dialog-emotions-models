"""Post-hoc temperature calibration of every trained checkpoint, scored by ECE.

The leaderboards report each model's *raw* (T=1) calibration error. The linear
TF-IDF models — the deploy candidates — are badly overconfident: their softmax
sits near one-hot while top-1 accuracy is ~0.6-0.7, so top-label ECE runs 0.25-0.40.
Temperature scaling (Guo et al., 2017) fixes this with a single scalar T applied to
the logits before softmax. Because it never touches the argmax, accuracy / F1 /
the confusion matrix are unchanged — it is a *free* recalibration.

This script does it honestly and uniformly:

* T is fit per model on the RuGoEmotions *validation* split (held out from
  training), choosing the T that minimizes validation ECE — the metric the thesis
  reports. A second T fit by NLL (the textbook smooth surrogate) is reported
  alongside as a cross-check that the choice is not bin-gaming.
* The before/after numbers are then read off the held-out RuGo *test* split and
  the native-Russian *CEDR* test (cross-domain transfer of the in-domain T).
* For CEDR a separate `cedr_T_oracle` is fit directly on CEDR (an optimistic
  upper bound: it has seen the eval set) to bound the transfer gap.

Everything runs on cached logits, so the T sweep is a handful of softmaxes per
model; the only real cost is loading each checkpoint once.

    python scripts/calibrate_temperature.py                 # all checkpoints
    python scripts/calibrate_temperature.py --skip-heavy     # linear/baseline only
    python scripts/calibrate_temperature.py --only logreg-char,ridge-char,fasttext
"""

from __future__ import annotations

import os

os.environ.setdefault("EMO_SCHEME", "7")  # checkpoints are 7-class; set before import

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

# Heavy checkpoints (transformers / random forest): loaded + inferred last, and
# skippable with --skip-heavy. Everything else is a fast linear / baseline model.
HEAVY = {
    "rubert-tiny2-finetune", "hf-seara-rubert-tiny2",
    "hf-fyaronskiy-deberta", "hf-maxkazak-rubert-base", "tree-rf",
}

COLUMNS = [
    "model", "kind", "T_ece", "T_nll",
    "acc_rugo", "rugo_ece_raw", "rugo_ece_cal", "rugo_kl_raw", "rugo_kl_cal",
    "acc_cedr", "cedr_ece_raw", "cedr_ece_cal", "cedr_kl_raw", "cedr_kl_cal",
    "cedr_T_oracle", "cedr_ece_oracle",
    # Deadband diagnostic (free = no deadband): makes the "ECE rose / 6x-gap" argument
    # reproducible. val_gain_free = val_ece_raw - val_ece_free; transfer sign = test_ece_free vs rugo_ece_raw.
    "T_ece_free", "val_ece_raw", "val_ece_free", "val_gain_free", "test_ece_free",
]


def _logits(model, texts: list[str]) -> np.ndarray:
    # fastText carries its own inference-time temperature; zero it out so the raw
    # logits are genuinely raw and the T we fit here is the whole calibration.
    if hasattr(model, "temperature"):
        model.temperature = 1.0
    return np.asarray(model.predict_logits(texts), dtype=float)


def _at(y: np.ndarray, logits: np.ndarray, temperature: float) -> dict:
    return quality_metrics(y, apply_temperature(logits, temperature))


def main() -> None:
    parser = argparse.ArgumentParser(description="Temperature-calibrate every checkpoint by ECE.")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/datasets/ru_go_emotions/simplified"))
    parser.add_argument("--cedr-dir", type=Path, default=Path("artifacts/datasets/cedr"))
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/calibration"))
    parser.add_argument("--skip-heavy", action="store_true")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated checkpoint names")
    parser.add_argument(
        "--min-improve", type=float, default=0.005,
        help="Deadband: keep T=1 unless val ECE improves by at least this much (default 0.005).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cedr_dir.mkdir(parents=True, exist_ok=True)

    val_x, val_y = load_rugoemotions(args.data_dir / "validation.parquet")
    test_x, test_y = load_rugoemotions(args.data_dir / "test.parquet")
    cedr_path = args.cedr_dir / "test.parquet"
    if not cedr_path.exists():
        urlretrieve(CEDR_URL, cedr_path)
    cedr_x, cedr_y = load_cedr(cedr_path)
    print(f"val={len(val_x)} test={len(test_x)} cedr={len(cedr_x)}", flush=True)

    checkpoints = available_checkpoints(args.models_dir)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        checkpoints = {n: p for n, p in checkpoints.items() if n in wanted}
    elif args.skip_heavy:
        checkpoints = {n: p for n, p in checkpoints.items() if n not in HEAVY}
    # Light models first, heavy last, so --skip-heavy and partial runs are useful.
    order = sorted(checkpoints, key=lambda n: (n in HEAVY, n))
    print(f"checkpoints: {', '.join(order)}", flush=True)

    rows: list[dict] = []
    for name in order:
        print(f"RUN {name}", flush=True)
        try:
            model = load_checkpoint(name, models_dir=args.models_dir)
            val_logits = _logits(model, val_x)
            test_logits = _logits(model, test_x)
            cedr_logits = _logits(model, cedr_x)
            del model
            gc.collect()

            # T fit on RuGo validation: primary by ECE (with a deadband so already-
            # calibrated models keep T=1 instead of chasing ECE noise), NLL as cross-check.
            t_ece, _ = best_temperature(val_logits, val_y, objective="ece", min_improve=args.min_improve)
            t_nll, _ = best_temperature(val_logits, val_y, objective="nll")
            # Free (no-deadband) ECE fit: exposes the noise gain the deadband suppresses.
            t_free, _ = best_temperature(val_logits, val_y, objective="ece", min_improve=0.0)
            val_ece_raw = _at(val_y, val_logits, 1.0)["ece"]
            val_ece_free = _at(val_y, val_logits, t_free)["ece"]
            test_ece_free = _at(test_y, test_logits, t_free)["ece"]
            # Optimistic CEDR ceiling: T fit directly on the CEDR eval set.
            t_cedr, _ = best_temperature(cedr_logits, cedr_y, objective="ece")

            rugo_raw = _at(test_y, test_logits, 1.0)
            rugo_cal = _at(test_y, test_logits, t_ece)
            cedr_raw = _at(cedr_y, cedr_logits, 1.0)
            cedr_cal = _at(cedr_y, cedr_logits, t_ece)
            cedr_orc = _at(cedr_y, cedr_logits, t_cedr)

            rows.append({
                "model": name, "kind": "heavy" if name in HEAVY else "light",
                "T_ece": round(t_ece, 3), "T_nll": round(t_nll, 3),
                "acc_rugo": round(rugo_raw["primary_accuracy"], 4),
                "rugo_ece_raw": round(rugo_raw["ece"], 4), "rugo_ece_cal": round(rugo_cal["ece"], 4),
                "rugo_kl_raw": round(rugo_raw["kl"], 4), "rugo_kl_cal": round(rugo_cal["kl"], 4),
                "acc_cedr": round(cedr_raw["primary_accuracy"], 4),
                "cedr_ece_raw": round(cedr_raw["ece"], 4), "cedr_ece_cal": round(cedr_cal["ece"], 4),
                "cedr_kl_raw": round(cedr_raw["kl"], 4), "cedr_kl_cal": round(cedr_cal["kl"], 4),
                "cedr_T_oracle": round(t_cedr, 3), "cedr_ece_oracle": round(cedr_orc["ece"], 4),
                "T_ece_free": round(t_free, 3),
                "val_ece_raw": round(val_ece_raw, 4), "val_ece_free": round(val_ece_free, 4),
                "val_gain_free": round(val_ece_raw - val_ece_free, 4),
                "test_ece_free": round(test_ece_free, 4),
            })
            print(f"  T*={t_ece} rugo_ece {rugo_raw['ece']:.3f}->{rugo_cal['ece']:.3f} "
                  f"cedr_ece {cedr_raw['ece']:.3f}->{cedr_cal['ece']:.3f}", flush=True)
        except Exception as exc:  # noqa: BLE001 - one bad checkpoint shouldn't sink the run
            print(f"  SKIP {name}: {type(exc).__name__}: {exc}", flush=True)

    frame = pd.DataFrame(rows).reindex(columns=COLUMNS).sort_values("rugo_ece_raw", ascending=False)
    csv_path = args.output_dir / "temperature_calibration.csv"
    frame.to_csv(csv_path, index=False)
    _write_markdown(frame, args.output_dir / "temperature_calibration.md")
    print("\nTEMPERATURE CALIBRATION (raw ECE descending)")
    print(frame.to_string(index=False))
    print(f"\nARTIFACTS {args.output_dir}")


def _write_markdown(frame: pd.DataFrame, path: Path) -> None:
    headers = ["model", "T*", "RuGo ECE raw", "RuGo ECE cal",
               "CEDR ECE raw", "CEDR ECE cal", "CEDR ECE (oracle T)"]
    keys = ["model", "T_ece", "rugo_ece_raw", "rugo_ece_cal",
            "cedr_ece_raw", "cedr_ece_cal", "cedr_ece_oracle"]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[k]) for k in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
