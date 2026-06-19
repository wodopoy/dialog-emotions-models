"""Per-class metrics, confusion matrices, and misclassified examples.

For the deployable models: per-emotion P/R/F1, the 6x6 confusion matrix (which
emotions get confused), and the most-confident mistakes for inspection.

    python scripts/error_analysis.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dialog_emo_models.datasets import load_rugoemotions
from dialog_emo_models.metrics import confusion_matrix, per_class_f1
from dialog_emo_models.models import LearnedLexiconEmotionModel, TfidfLogRegEmotionModel
from dialog_emo_models.schema import EMOTIONS

MODELS = {
    "logreg-char": lambda: TfidfLogRegEmotionModel(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000,
        sublinear_tf=True, C=2.0, class_weight=None),
    "logreg-union": lambda: TfidfLogRegEmotionModel(
        analyzer="word+char", min_df=2, max_features=20000, sublinear_tf=True,
        C=1.0, class_weight=None),
    "lexicon-learned": lambda: LearnedLexiconEmotionModel(top_k=200, min_count=5),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/datasets/ru_go_emotions/simplified"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/error-analysis"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y = load_rugoemotions(args.data_dir / "train.parquet")
    val_x, val_y = load_rugoemotions(args.data_dir / "validation.parquet")
    true_primary = val_y.argmax(axis=1)

    for name, factory in MODELS.items():
        print(f"\n=== {name} ===", flush=True)
        model = factory().fit(train_x, train_y)
        proba = model.predict_proba(val_x)

        per_class = per_class_f1(val_y, proba)
        pc_frame = pd.DataFrame(per_class).T.reset_index().rename(columns={"index": "emotion"})
        pc_frame.to_csv(args.output_dir / f"per_class_{name}.csv", index=False)

        matrix = confusion_matrix(val_y, proba)
        conf = pd.DataFrame(matrix, index=[f"true_{e}" for e in EMOTIONS], columns=list(EMOTIONS))
        conf.to_csv(args.output_dir / f"confusion_{name}.csv")

        print("per-class F1:", {e: round(per_class[e]["f1"], 3) for e in EMOTIONS})
        print("confusion matrix (rows=true, cols=pred):")
        print(conf.to_string())

        if name == "logreg-char":
            pred = proba.argmax(axis=1)
            conf_score = proba.max(axis=1)
            wrong = np.where(pred != true_primary)[0]
            wrong = wrong[np.argsort(-conf_score[wrong])][:25]
            errors = pd.DataFrame({
                "text": [val_x[i][:140] for i in wrong],
                "true": [EMOTIONS[true_primary[i]] for i in wrong],
                "pred": [EMOTIONS[pred[i]] for i in wrong],
                "confidence": [round(float(conf_score[i]), 3) for i in wrong],
            })
            errors.to_csv(args.output_dir / "errors_logreg-char.csv", index=False)
            print("\nMost confident mistakes (logreg-char):")
            print(errors.head(12).to_string(index=False))

    print(f"\nARTIFACTS {args.output_dir}")


if __name__ == "__main__":
    main()
