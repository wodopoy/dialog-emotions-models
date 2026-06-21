"""Figures for the ECE-vs-NLL temperature-objective study (for the thesis).

Two figures, both off ``calibration_objective_comparison.csv`` (and, for the
reliability panels, the saved checkpoints):

* objective_ece_overview.png — grouped bars of ECE under raw / ECE-fit / NLL-fit on
  RuGo test and CEDR test, all models. Shows in-domain ECE-fit wins, cross-domain
  NLL often wins, and the fastText / maxkazak blow-ups under NLL.
* objective_reliability_mechanism.png — reliability diagrams contrasting where NLL
  helps (DeBERTa on CEDR: raw -> T_ece -> T_nll hugs the diagonal best) and where it
  breaks (fastText: T_nll over-softens and pulls the bars off the diagonal).

    python scripts/plot_objective_comparison.py
"""

from __future__ import annotations

import os

os.environ.setdefault("EMO_SCHEME", "7")

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dialog_emo_models.checkpoints import load_checkpoint
from dialog_emo_models.datasets import load_cedr, load_rugoemotions
from dialog_emo_models.metrics import apply_temperature, expected_calibration_error, reliability_curve

CEDR_URL = ("https://huggingface.co/datasets/sagteam/cedr_v1/resolve/"
            "refs%2Fconvert%2Fparquet/main/test/0000.parquet")
BASELINES = {"dummy", "majority", "prior"}


def plot_overview(frame: pd.DataFrame, out: Path) -> None:
    core = frame[~frame["model"].isin(BASELINES)].sort_values("rugo_ece_raw")
    models = core["model"].tolist()
    y = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5), sharey=True)
    for ax, dom, title in ((axes[0], "rugo", "RuGoEmotions test (in-domain)"),
                           (axes[1], "cedr", "CEDR test (native, deploy domain)")):
        ax.barh(y - 0.26, core[f"{dom}_ece_raw"], height=0.26, color="#4C72B0", label="raw (T=1)")
        ax.barh(y + 0.00, core[f"{dom}_ece_Tece"], height=0.26, color="#55A868", label="fit by ECE")
        ax.barh(y + 0.26, core[f"{dom}_ece_Tnll"], height=0.26, color="#C44E52", label="fit by NLL")
        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=8)
        ax.set_xlabel("ECE")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Top-label ECE: temperature fit by ECE vs by NLL", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "objective_ece_overview.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out / 'objective_ece_overview.png'}", flush=True)


def _panel(ax, y, probs, *, title, color):
    curve = reliability_curve(y, probs)
    c, conf, acc, counts = curve["centers"], curve["confidence"], curve["accuracy"], curve["counts"]
    seen = counts > 0
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, zorder=1)
    ax.bar(c[seen], acc[seen], width=0.1, color=color, alpha=0.85, zorder=2)
    gap_bottom = np.minimum(acc[seen], conf[seen])
    ax.bar(c[seen], np.abs(conf[seen] - acc[seen]), bottom=gap_bottom, width=0.1,
           color="crimson", alpha=0.35, zorder=3)
    ax.set_title(f"{title}\nECE={expected_calibration_error(y, probs):.3f}", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("confidence", fontsize=8)


def plot_mechanism(frame: pd.DataFrame, data_dir: Path, cedr_dir: Path, out: Path) -> None:
    row = {r["model"]: r for _, r in frame.iterrows()}
    cedr_path = cedr_dir / "test.parquet"
    if not cedr_path.exists():
        cedr_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(CEDR_URL, cedr_path)
    cedr_x, cedr_y = load_cedr(cedr_path)
    rugo_x, rugo_y = load_rugoemotions(data_dir / "test.parquet")

    # (model, eval split, why) — DeBERTa shows NLL helping cross-domain; fastText the pathology.
    cases = [
        ("hf-fyaronskiy-deberta", cedr_x, cedr_y, "DeBERTa, CEDR — NLL помогает"),
        ("fasttext", rugo_x, rugo_y, "fastText, RuGo — NLL ломает"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for r, (name, ex, ey, label) in enumerate(cases):
        model = load_checkpoint(name, apply_temperature=False)
        if hasattr(model, "temperature"):
            model.temperature = 1.0
        logits = np.asarray(model.predict_logits(ex), dtype=float)
        del model
        t_ece, t_nll = float(row[name]["T_ece"]), float(row[name]["T_nll"])
        for c, (t, sub, color) in enumerate([
            (1.0, "raw (T=1)", "#4C72B0"),
            (t_ece, f"fit by ECE (T={t_ece:.2f})", "#55A868"),
            (t_nll, f"fit by NLL (T={t_nll:.2f})", "#C44E52"),
        ]):
            _panel(axes[r, c], ey, apply_temperature(logits, t), title=sub, color=color)
        axes[r, 0].set_ylabel(f"{label}\naccuracy", fontsize=9)
    fig.suptitle("Надёжность под температурой, подобранной по ECE и по NLL", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "objective_reliability_mechanism.png", dpi=140)
    plt.close(fig)
    print(f"  wrote {out / 'objective_reliability_mechanism.png'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Figures for the ECE-vs-NLL objective study.")
    parser.add_argument("--csv", type=Path,
                        default=Path("artifacts/experiments/calibration/calibration_objective_comparison.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/datasets/ru_go_emotions/simplified"))
    parser.add_argument("--cedr-dir", type=Path, default=Path("artifacts/datasets/cedr"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/img/calibration"))
    parser.add_argument("--no-mechanism", action="store_true", help="skip the reliability figure (no model loading)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.csv)
    print("PLOT overview", flush=True)
    plot_overview(frame, args.output_dir)
    if not args.no_mechanism:
        print("PLOT mechanism (loads DeBERTa + fastText)", flush=True)
        plot_mechanism(frame, args.data_dir, args.cedr_dir, args.output_dir)
    print(f"ARTIFACTS {args.output_dir}")


if __name__ == "__main__":
    main()
