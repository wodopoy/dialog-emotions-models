"""Score the heavy transformer models on the SAME protocol as the light ones.

Same val/test splits and the same quality + deployment metrics as
`tune_light_models.py`, so the heavy models can sit in one joint leaderboard:

- `rubert-tiny2-finetune`: fine-tuned here on the train split (argmax targets).
- 3 HF GoEmotions presets: inference only.

    python scripts/evaluate_heavy_models.py
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dialog_emo_models.metrics import measure_latency_ms, quality_metrics
from dialog_emo_models.models import (
    GOEMOTIONS_GROUPS,
    GOEMOTIONS_LABELS,
    FyaronskiyDebertaGoEmotionsModel,
    MaxKazakRuBertBaseGoEmotionsModel,
    RuBertTiny2EmotionModel,
    SearaRuBertTiny2GoEmotionsModel,
)
from dialog_emo_models.schema import EMOTIONS

LIGHT_COLUMNS = [
    "family", "params",
    "val_kl", "val_acc", "val_macro_f1", "val_ece",
    "test_kl", "test_acc", "test_macro_f1", "test_ece",
    "size_mb", "latency_p50_ms", "latency_p95_ms",
]


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
            {
                label_to_group[id_to_name[int(i)]]
                for i in row["labels"]
                if id_to_name[int(i)] in label_to_group
            }
        )
        if not groups:
            continue
        texts.append(str(row["ru_text"]))
        labels.append([(1.0 / len(groups)) if e in groups else 0.0 for e in EMOTIONS])
    return texts, np.asarray(labels, dtype=float)


def _param_size_mb(model) -> float:
    torch_model = getattr(model, "_model", None)
    if torch_model is None or not hasattr(torch_model, "parameters"):
        return float("nan")
    total = sum(p.numel() * p.element_size() for p in torch_model.parameters())
    return total / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score heavy models on the shared protocol.")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("artifacts/datasets/ru_go_emotions/simplified")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/experiments/heavy-leaderboard")
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--latency-sample", type=int, default=80)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_texts, train_labels = load_split(args.data_dir / "train.parquet")
    val_texts, val_labels = load_split(args.data_dir / "validation.parquet")
    test_texts, test_labels = load_split(args.data_dir / "test.parquet")
    print(f"train={len(train_texts)} val={len(val_texts)} test={len(test_texts)}", flush=True)

    jobs = [
        ("rubert-tiny2-finetune", "train",
         lambda: RuBertTiny2EmotionModel(epochs=args.epochs),
         {"base": "cointegrated/rubert-tiny2", "epochs": args.epochs}),
        ("hf-seara-rubert-tiny2", "infer", SearaRuBertTiny2GoEmotionsModel,
         {"model_id": "seara/rubert-tiny2-russian-emotion-detection-ru-go-emotions"}),
        ("hf-fyaronskiy-deberta", "infer", FyaronskiyDebertaGoEmotionsModel,
         {"model_id": "fyaronskiy/deberta-v1-base-russian-go-emotions"}),
        ("hf-maxkazak-rubert-base", "infer", MaxKazakRuBertBaseGoEmotionsModel,
         {"model_id": "MaxKazak/ruBert-base-russian-emotion-detection"}),
    ]

    rows: list[dict] = []
    for name, mode, factory, params in jobs:
        print(f"\n=== {name} ({mode}) ===", flush=True)
        try:
            model = factory()
            if mode == "train":
                model.fit(train_texts, train_labels)
            val_metrics = quality_metrics(val_labels, model.predict_proba(val_texts, show_progress=True))
            test_metrics = quality_metrics(test_labels, model.predict_proba(test_texts, show_progress=True))
            latency = measure_latency_ms(model, val_texts, sample=args.latency_sample, repeats=2)
            rows.append({
                "family": name,
                "params": json.dumps(params, ensure_ascii=False),
                "val_kl": round(val_metrics["kl"], 4),
                "val_acc": round(val_metrics["primary_accuracy"], 4),
                "val_macro_f1": round(val_metrics["macro_f1"], 4),
                "val_ece": round(val_metrics["ece"], 4),
                "test_kl": round(test_metrics["kl"], 4),
                "test_acc": round(test_metrics["primary_accuracy"], 4),
                "test_macro_f1": round(test_metrics["macro_f1"], 4),
                "test_ece": round(test_metrics["ece"], 4),
                "size_mb": round(_param_size_mb(model), 1),
                "latency_p50_ms": round(latency["latency_p50_ms"], 3),
                "latency_p95_ms": round(latency["latency_p95_ms"], 3),
            })
            print(f"  val: kl={val_metrics['kl']:.4f} acc={val_metrics['primary_accuracy']:.4f} "
                  f"macroF1={val_metrics['macro_f1']:.4f} ece={val_metrics['ece']:.4f}", flush=True)
            del model
            gc.collect()
        except Exception as exc:
            print(f"  SKIP {name}: {type(exc).__name__}: {exc}", flush=True)

    heavy = pd.DataFrame(rows).reindex(columns=LIGHT_COLUMNS)
    heavy.to_csv(args.output_dir / "heavy_metrics.csv", index=False)

    light_path = Path("artifacts/experiments/light-tuning/tuning_best.csv")
    if light_path.exists():
        light = pd.read_csv(light_path)
        combined = pd.concat([light, heavy], ignore_index=True).sort_values("val_kl")
        combined.to_csv(args.output_dir / "combined_leaderboard.csv", index=False)
        print("\nCOMBINED LEADERBOARD (val, sorted by KL)")
        print(combined.to_string(index=False))
    else:
        print("\nHEAVY MODELS")
        print(heavy.to_string(index=False))
    print(f"\nARTIFACTS {args.output_dir}")


if __name__ == "__main__":
    main()
