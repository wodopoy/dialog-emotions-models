"""Reliability diagrams (before / after temperature calibration) for the thesis.

Two figure kinds, both off the same `metrics.reliability_curve` that defines ECE,
so what the picture shows is exactly the number the table reports:

* per-model reliability diagram — accuracy vs confidence per bin, raw (T=1) next to
  calibrated (T*), with the gap shaded and ECE annotated;
* one overview bar chart — RuGo / CEDR ECE before vs after across all models.

T* is re-fit here on RuGo validation with the same deadband as the calibration run,
so the figures stand alone (no dependency on having run the CSV first), while the
overview reads the CSV the calibration script wrote.

    python scripts/plot_reliability.py                      # default curated set + overview
    python scripts/plot_reliability.py --models ridge-char,logreg-char --split cedr
"""

from __future__ import annotations

import os

os.environ.setdefault("EMO_SCHEME", "7")  # checkpoints are 7-class

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dialog_emo_models.checkpoints import load_checkpoint
from dialog_emo_models.datasets import load_cedr, load_rugoemotions
from dialog_emo_models.metrics import (
    apply_temperature,
    best_temperature,
    expected_calibration_error,
    reliability_curve,
)

CEDR_URL = ("https://huggingface.co/datasets/sagteam/cedr_v1/resolve/"
            "refs%2Fconvert%2Fparquet/main/test/0000.parquet")
# Curated for the thesis: the dramatic fix (ridge / maxkazak), a fine-tuned
# transformer, the already-calibrated deploy pick, and the "no help" contrast (fastText).
DEFAULT_MODELS = ["ridge-char", "hf-maxkazak-rubert-base", "hf-fyaronskiy-deberta",
                  "logreg-char", "fasttext"]
BASELINES = {"dummy", "majority", "prior"}  # nothing to calibrate; off the overview


def _reliability_panel(ax, y, probs, *, title: str, color: str) -> None:
    curve = reliability_curve(y, probs)
    centers, conf, acc, counts = curve["centers"], curve["confidence"], curve["accuracy"], curve["counts"]
    seen = counts > 0
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, zorder=1)  # perfect calibration
    # gap = where accuracy departs from confidence (red), accuracy itself (solid).
    ax.bar(centers[seen], acc[seen], width=0.1, color=color, alpha=0.85, label="accuracy", zorder=2)
    gap_bottom = np.minimum(acc[seen], conf[seen])
    gap_height = np.abs(conf[seen] - acc[seen])
    ax.bar(centers[seen], gap_height, bottom=gap_bottom, width=0.1,
           color="crimson", alpha=0.35, label="gap", zorder=3)
    ece = expected_calibration_error(y, probs)
    ax.set_title(f"{title}\nECE={ece:.3f}", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence")
    ax.set_aspect("equal")


def plot_model(name: str, val, eval_split, split_name: str, min_improve: float, out: Path) -> None:
    model = load_checkpoint(name, apply_temperature=False)
    if hasattr(model, "temperature"):
        model.temperature = 1.0
    vx, vy = val
    ex, ey = eval_split
    val_logits = np.asarray(model.predict_logits(vx), dtype=float)
    eval_logits = np.asarray(model.predict_logits(ex), dtype=float)
    t_star, _ = best_temperature(val_logits, vy, objective="ece", min_improve=min_improve)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    _reliability_panel(axes[0], ey, apply_temperature(eval_logits, 1.0),
                       title=f"{name} — raw (T=1)", color="steelblue")
    _reliability_panel(axes[1], ey, apply_temperature(eval_logits, t_star),
                       title=f"{name} — calibrated (T={t_star:.2f})", color="seagreen")
    axes[0].set_ylabel("accuracy")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Reliability on {split_name}", fontsize=11)
    fig.tight_layout()
    path = out / f"reliability_{name}_{split_name}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}  (T*={t_star:.2f})", flush=True)


def plot_overview(csv_path: Path, out: Path) -> None:
    if not csv_path.exists():
        print(f"  no CSV at {csv_path}; run calibrate_temperature.py first — skipping overview")
        return
    frame = pd.read_csv(csv_path)
    frame = frame[~frame["model"].isin(BASELINES)].sort_values("rugo_ece_raw")
    models = frame["model"].tolist()
    ypos = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharey=True)
    for ax, dom, label in ((axes[0], "rugo", "RuGoEmotions test"), (axes[1], "cedr", "CEDR test")):
        ax.barh(ypos - 0.2, frame[f"{dom}_ece_raw"], height=0.38, color="steelblue", label="raw (T=1)")
        ax.barh(ypos + 0.2, frame[f"{dom}_ece_cal"], height=0.38, color="seagreen", label="calibrated")
        ax.set_yticks(ypos)
        ax.set_yticklabels(models, fontsize=8)
        ax.set_xlabel("ECE")
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8)
    fig.suptitle("Top-label ECE before vs after temperature calibration", fontsize=12)
    fig.tight_layout()
    path = out / "ece_overview.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reliability diagrams for temperature calibration.")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/datasets/ru_go_emotions/simplified"))
    parser.add_argument("--cedr-dir", type=Path, default=Path("artifacts/datasets/cedr"))
    parser.add_argument("--csv", type=Path,
                        default=Path("artifacts/experiments/calibration/temperature_calibration.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/img/calibration"))
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--split", choices=["rugo-test", "cedr"], default="rugo-test")
    parser.add_argument("--min-improve", type=float, default=0.005)
    parser.add_argument("--no-overview", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    val = load_rugoemotions(args.data_dir / "validation.parquet")
    if args.split == "cedr":
        cedr_path = args.cedr_dir / "test.parquet"
        if not cedr_path.exists():
            cedr_path.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(CEDR_URL, cedr_path)
        eval_split, split_name = load_cedr(cedr_path), "cedr-test"
    else:
        eval_split, split_name = load_rugoemotions(args.data_dir / "test.parquet"), "rugo-test"

    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"PLOT {name}", flush=True)
        try:
            plot_model(name, val, eval_split, split_name, args.min_improve, args.output_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {name}: {type(exc).__name__}: {exc}", flush=True)

    if not args.no_overview:
        print("PLOT overview", flush=True)
        plot_overview(args.csv, args.output_dir)
    print(f"ARTIFACTS {args.output_dir}")


if __name__ == "__main__":
    main()
