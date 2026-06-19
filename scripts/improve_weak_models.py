"""Try to squeeze more quality out of the weak models: fastText and lexicon.

fastText: compare the best grid config vs fastText's own autotune, then sweep a
calibration temperature against validation KL (its main weakness was
overconfidence). Lexicon: compare the hand-written dictionary vs a data-driven
learned lexicon over a few `top_k`.

    python scripts/improve_weak_models.py            # autotune 180s
    python scripts/improve_weak_models.py --autotune-duration 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dialog_emo_models.metrics import measure_latency_ms, model_size_mb, quality_metrics
from dialog_emo_models.models import (
    GOEMOTIONS_GROUPS,
    GOEMOTIONS_LABELS,
    FastTextSupervisedEmotionModel,
    LearnedLexiconEmotionModel,
    LexiconEmotionModel,
)
from dialog_emo_models.schema import EMOTIONS

BEST_FT_PARAMS = {"lr": 0.3, "epoch": 25, "wordNgrams": 2, "dim": 100, "minn": 3, "maxn": 6, "loss": "ova"}
TEMPERATURES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
COLUMNS = ["model", "val_kl", "val_acc", "val_macro_f1", "val_ece",
           "test_kl", "test_acc", "test_macro_f1", "test_ece", "size_mb",
           "latency_p50_ms", "latency_p95_ms", "notes"]


def load_split(path: Path) -> tuple[list[str], np.ndarray]:
    raw = pd.read_parquet(path)
    id_to_name = dict(enumerate(GOEMOTIONS_LABELS))
    label_to_group = {
        label: group for group, labels in GOEMOTIONS_GROUPS.items() for label in labels
    }
    texts: list[str] = []
    labels: list[list[float]] = []
    for _, row in raw.iterrows():
        groups = sorted(
            {label_to_group[id_to_name[int(i)]] for i in row["labels"]
             if id_to_name[int(i)] in label_to_group}
        )
        if not groups:
            continue
        texts.append(str(row["ru_text"]))
        labels.append([(1.0 / len(groups)) if e in groups else 0.0 for e in EMOTIONS])
    return texts, np.asarray(labels, dtype=float)


def evaluate(model, name, splits, *, notes="", tmp=Path("/tmp/_weak.joblib")) -> dict:
    vx, vy, tx, ty = splits
    val = quality_metrics(vy, model.predict_proba(vx))
    test = quality_metrics(ty, model.predict_proba(tx))
    try:
        joblib.dump(model, tmp)
        size = round(model_size_mb(tmp), 3)
        tmp.unlink(missing_ok=True)
    except Exception:
        size = float("nan")
    latency = measure_latency_ms(model, vx, sample=150, repeats=2)
    return {
        "model": name,
        "val_kl": round(val["kl"], 4), "val_acc": round(val["primary_accuracy"], 4),
        "val_macro_f1": round(val["macro_f1"], 4), "val_ece": round(val["ece"], 4),
        "test_kl": round(test["kl"], 4), "test_acc": round(test["primary_accuracy"], 4),
        "test_macro_f1": round(test["macro_f1"], 4), "test_ece": round(test["ece"], 4),
        "size_mb": size,
        "latency_p50_ms": round(latency["latency_p50_ms"], 3),
        "latency_p95_ms": round(latency["latency_p95_ms"], 3),
        "notes": notes,
    }


def best_temperature(model, vx, vy) -> tuple[float, float]:
    best_t, best_kl = 1.0, float("inf")
    for temperature in TEMPERATURES:
        model.temperature = temperature
        kl = quality_metrics(vy, model.predict_proba(vx))["kl"]
        if kl < best_kl:
            best_t, best_kl = temperature, kl
    model.temperature = best_t
    return best_t, best_kl


def main() -> None:
    parser = argparse.ArgumentParser(description="Improve the weak (fastText, lexicon) models.")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("artifacts/datasets/ru_go_emotions/simplified")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/experiments/weak-models")
    )
    parser.add_argument("--autotune-duration", type=int, default=180)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y = load_split(args.data_dir / "train.parquet")
    val_x, val_y = load_split(args.data_dir / "validation.parquet")
    test_x, test_y = load_split(args.data_dir / "test.parquet")
    splits = (val_x, val_y, test_x, test_y)
    print(f"train={len(train_x)} val={len(val_x)} test={len(test_x)}", flush=True)

    rows: list[dict] = []

    # --- fastText -------------------------------------------------------
    print("\n# fastText: grid-best config", flush=True)
    ft_grid = FastTextSupervisedEmotionModel(params=BEST_FT_PARAMS).fit(train_x, train_y)
    rows.append(evaluate(ft_grid, "fasttext-grid (T=1)", splits, notes=json.dumps(BEST_FT_PARAMS)))
    t, _ = best_temperature(ft_grid, val_x, val_y)
    rows.append(evaluate(ft_grid, "fasttext-grid +temp", splits, notes=f"T={t}"))

    print(f"\n# fastText: autotune ({args.autotune_duration}s)", flush=True)
    ft_auto = FastTextSupervisedEmotionModel().fit_autotune(
        train_x, train_y, val_x, val_y, duration=args.autotune_duration
    )
    rows.append(evaluate(ft_auto, "fasttext-autotune (T=1)", splits, notes="autotune"))
    t, _ = best_temperature(ft_auto, val_x, val_y)
    rows.append(evaluate(ft_auto, "fasttext-autotune +temp", splits, notes=f"autotune, T={t}"))

    # --- lexicon --------------------------------------------------------
    print("\n# lexicon: hand-written", flush=True)
    rows.append(evaluate(LexiconEmotionModel(), "lexicon-hand", splits, notes="no training"))
    for top_k in (50, 100, 200, 400):
        print(f"# learned lexicon top_k={top_k}", flush=True)
        learned = LearnedLexiconEmotionModel(top_k=top_k, min_count=5).fit(train_x, train_y)
        rows.append(evaluate(learned, f"lexicon-learned top_k={top_k}", splits, notes="log-odds"))

    frame = pd.DataFrame(rows).reindex(columns=COLUMNS)
    frame.to_csv(args.output_dir / "weak_models.csv", index=False)
    print("\nRESULTS (val KL ascending)")
    print(frame.sort_values("val_kl").to_string(index=False))
    print(f"\nARTIFACTS {args.output_dir}")


if __name__ == "__main__":
    main()
