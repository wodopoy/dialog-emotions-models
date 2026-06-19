"""Learning curve: quality vs train-set size for the best linear models.

Shows whether the linear models are data-starved or near a plateau (i.e. whether
"add more data" would help, or the gap to transformers is model capacity).

    python scripts/learning_curve.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dialog_emo_models.datasets import load_rugoemotions
from dialog_emo_models.metrics import quality_metrics
from dialog_emo_models.models import TfidfLogRegEmotionModel

SIZES = [2000, 5000, 10000, 20000, 39055]
MODELS = {
    "logreg-char": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                        max_features=30000, sublinear_tf=True, C=2.0, class_weight=None),
    "logreg-union": dict(analyzer="word+char", min_df=2, max_features=20000,
                         sublinear_tf=True, C=1.0, class_weight=None),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/datasets/ru_go_emotions/simplified"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/learning-curve"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y = load_rugoemotions(args.data_dir / "train.parquet")
    val_x, val_y = load_rugoemotions(args.data_dir / "validation.parquet")
    print(f"train={len(train_x)} val={len(val_x)}", flush=True)

    rows = []
    for name, params in MODELS.items():
        for size in SIZES:
            n = min(size, len(train_x))
            model = TfidfLogRegEmotionModel(**params).fit(train_x[:n], train_y[:n])
            metrics = quality_metrics(val_y, model.predict_proba(val_x))
            rows.append({"model": name, "train_size": n,
                         "val_acc": round(metrics["primary_accuracy"], 4),
                         "val_macro_f1": round(metrics["macro_f1"], 4),
                         "val_kl": round(metrics["kl"], 4)})
            print(f"  {name} n={n} acc={metrics['primary_accuracy']:.4f} kl={metrics['kl']:.4f}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "learning_curve.csv", index=False)
    print("\nLEARNING CURVE")
    print(frame.to_string(index=False))
    print(f"\nARTIFACTS {args.output_dir}")


if __name__ == "__main__":
    main()
