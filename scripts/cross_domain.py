"""Cross-domain test: models trained on RuGoEmotions, scored on native-Russian CEDR.

Quantifies the translationese-Reddit -> native-Russian gap. CEDR has no 'warmth'
and drops 'surprise', so it covers 4 of 6 emotions; warmth is never scored here
(a stated limitation). Reports in-domain (RuGoEmotions test) vs cross-domain (CEDR).

    python scripts/cross_domain.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from dialog_emo_models.datasets import load_cedr, load_rugoemotions
from dialog_emo_models.metrics import quality_metrics
from dialog_emo_models.models import (
    LabelPriorEmotionModel,
    LearnedLexiconEmotionModel,
    TfidfLogRegEmotionModel,
)

CEDR_URL = ("https://huggingface.co/datasets/sagteam/cedr_v1/resolve/"
            "refs%2Fconvert%2Fparquet/main/test/0000.parquet")

MODELS = {
    "prior": lambda: LabelPriorEmotionModel(),
    "lexicon-learned": lambda: LearnedLexiconEmotionModel(top_k=200, min_count=5),
    "logreg-char": lambda: TfidfLogRegEmotionModel(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000,
        sublinear_tf=True, C=2.0, class_weight=None),
    "logreg-union": lambda: TfidfLogRegEmotionModel(
        analyzer="word+char", min_df=2, max_features=20000, sublinear_tf=True,
        C=1.0, class_weight=None),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/datasets/ru_go_emotions/simplified"))
    parser.add_argument("--cedr-dir", type=Path, default=Path("artifacts/datasets/cedr"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/cross-domain"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cedr_dir.mkdir(parents=True, exist_ok=True)

    train = load_rugoemotions(args.data_dir / "train.parquet")
    rugo_test = load_rugoemotions(args.data_dir / "test.parquet")
    cedr_path = args.cedr_dir / "test.parquet"
    if not cedr_path.exists():
        urlretrieve(CEDR_URL, cedr_path)
    cedr = load_cedr(cedr_path)
    print(f"train={len(train[0])} rugo_test={len(rugo_test[0])} cedr_test={len(cedr[0])}", flush=True)

    rows = []
    for name, factory in MODELS.items():
        model = factory().fit(*train)
        ind = quality_metrics(rugo_test[1], model.predict_proba(rugo_test[0]))
        out = quality_metrics(cedr[1], model.predict_proba(cedr[0]))
        rows.append({
            "model": name,
            "rugo_acc": round(ind["primary_accuracy"], 4),
            "cedr_acc": round(out["primary_accuracy"], 4),
            "acc_drop": round(ind["primary_accuracy"] - out["primary_accuracy"], 4),
            "rugo_macro_f1": round(ind["macro_f1"], 4),
            "cedr_macro_f1": round(out["macro_f1"], 4),
            "rugo_kl": round(ind["kl"], 4),
            "cedr_kl": round(out["kl"], 4),
        })
        print(f"  {name}: rugo_acc={rows[-1]['rugo_acc']} cedr_acc={rows[-1]['cedr_acc']} "
              f"drop={rows[-1]['acc_drop']}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "cross_domain.csv", index=False)
    print("\nCROSS-DOMAIN (train=RuGoEmotions; CEDR covers 4/6 emotions, no warmth)")
    print(frame.to_string(index=False))
    print(f"\nARTIFACTS {args.output_dir}")


if __name__ == "__main__":
    main()
