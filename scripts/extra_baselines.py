"""Completeness baselines: Naive Bayes and tree ensembles (random forest, HGB).

Same protocol/metrics as the other models. Expectation (and the point of the
experiment): on sparse text these are dominated by the linear models and far
larger — a documented "we tried it" result.

    python scripts/extra_baselines.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from dialog_emo_models.datasets import load_rugoemotions
from dialog_emo_models.metrics import measure_latency_ms, model_size_mb, quality_metrics
from dialog_emo_models.models import TfidfNaiveBayesEmotionModel, TfidfTreeEmotionModel

COLUMNS = ["model", "val_kl", "val_acc", "val_macro_f1", "val_ece",
           "test_kl", "test_acc", "test_macro_f1", "size_mb", "latency_p50_ms"]


def evaluate(model, name, val, test, tmp) -> dict:
    vx, vy = val
    tx, ty = test
    v = quality_metrics(vy, model.predict_proba(vx))
    t = quality_metrics(ty, model.predict_proba(tx))
    joblib.dump(model, tmp)
    size = round(model_size_mb(tmp), 2)
    tmp.unlink(missing_ok=True)
    lat = measure_latency_ms(model, vx, sample=150, repeats=2)
    return {"model": name, "val_kl": round(v["kl"], 4), "val_acc": round(v["primary_accuracy"], 4),
            "val_macro_f1": round(v["macro_f1"], 4), "val_ece": round(v["ece"], 4),
            "test_kl": round(t["kl"], 4), "test_acc": round(t["primary_accuracy"], 4),
            "test_macro_f1": round(t["macro_f1"], 4), "size_mb": size,
            "latency_p50_ms": round(lat["latency_p50_ms"], 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/datasets/ru_go_emotions/simplified"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/extra-baselines"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = load_rugoemotions(args.data_dir / "train.parquet")
    val = load_rugoemotions(args.data_dir / "validation.parquet")
    test = load_rugoemotions(args.data_dir / "test.parquet")
    print(f"train={len(train[0])} val={len(val[0])} test={len(test[0])}", flush=True)

    models = {
        "nb-complement": TfidfNaiveBayesEmotionModel(kind="complement"),
        "nb-multinomial": TfidfNaiveBayesEmotionModel(kind="multinomial"),
        "tree-rf": TfidfTreeEmotionModel(estimator="rf", svd_components=300, n_estimators=300),
        "tree-hgb": TfidfTreeEmotionModel(estimator="hgb", svd_components=300),
    }
    rows = []
    tmp = args.output_dir / "_tmp.joblib"
    for name, model in models.items():
        print(f"RUN {name}", flush=True)
        model.fit(*train)
        rows.append(evaluate(model, name, val, test, tmp))
        print(f"  {rows[-1]}", flush=True)

    frame = pd.DataFrame(rows).reindex(columns=COLUMNS).sort_values("val_kl")
    frame.to_csv(args.output_dir / "extra_baselines.csv", index=False)
    print("\nEXTRA BASELINES (val KL ascending)")
    print(frame.to_string(index=False))
    print(f"\nARTIFACTS {args.output_dir}")


if __name__ == "__main__":
    main()
