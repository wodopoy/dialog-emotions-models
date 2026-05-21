from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from dialog_emo_models.models import (
    GOEMOTIONS_GROUPS,
    GOEMOTIONS_LABELS,
    TfidfLogRegEmotionModel,
    TfidfRidgeEmotionModel,
)
from dialog_emo_models.models.dummy import DummyEmotionModel
from dialog_emo_models.schema import EMOTIONS
from dialog_emo_models.training import train_from_full_frame

SPLIT_URLS = {
    "train": "https://huggingface.co/datasets/seara/ru_go_emotions/resolve/main/"
    "simplified/train-00000-of-00001-46692d7e0c0147a4.parquet",
    "validation": "https://huggingface.co/datasets/seara/ru_go_emotions/resolve/main/"
    "simplified/validation-00000-of-00001-9e6cdf9c1f2a20a4.parquet",
    "test": "https://huggingface.co/datasets/seara/ru_go_emotions/resolve/main/"
    "simplified/test-00000-of-00001-0acb4be83ca6567e.parquet",
}

MODEL_FACTORIES = {
    "dummy": DummyEmotionModel,
    "ridge-tfidf": TfidfRidgeEmotionModel,
    "logreg-tfidf": TfidfLogRegEmotionModel,
    "ridge-word-tfidf": lambda: TfidfRidgeEmotionModel(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50_000
    ),
    "logreg-word-tfidf": lambda: TfidfLogRegEmotionModel(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50_000, max_iter=500
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate small baselines on ru_go_emotions.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("artifacts/datasets/ru_go_emotions/simplified"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments/ru-go-emotions-tfidf"),
    )
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _download_missing_splits(args.data_dir)

    splits: dict[str, pd.DataFrame] = {}
    stats: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        splits[split], stats[split] = _convert_split(args.data_dir / f"{split}.parquet")
        splits[split].to_csv(args.output_dir / f"{split}.full.csv", index=False)

    results = _evaluate(splits, args.output_dir)
    stats_frame = pd.DataFrame(stats).T
    stats_frame.to_csv(args.output_dir / "dataset_stats.csv")
    results.to_csv(args.output_dir / "metrics.csv", index=False)

    print("DATASET_STATS")
    print(stats_frame.to_string())
    print("\nCLASS_DISTRIBUTION")
    for split_name, frame in splits.items():
        counts = (frame.loc[:, EMOTIONS] > 0).sum(axis=0).astype(int).to_dict()
        print(split_name, counts)
    print("\nRESULTS")
    print(results.sort_values(["split", "kl", "mae"]).to_string(index=False))
    print(f"\nARTIFACTS {args.output_dir}")


def _download_missing_splits(data_dir: Path) -> None:
    for split, url in SPLIT_URLS.items():
        path = data_dir / f"{split}.parquet"
        if not path.exists():
            urlretrieve(url, path)


def _convert_split(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_parquet(path)
    label_id_to_name = dict(enumerate(GOEMOTIONS_LABELS))
    label_to_group = {
        label: group for group, labels in GOEMOTIONS_GROUPS.items() for label in labels
    }

    rows: list[dict[str, object]] = []
    dropped_unmapped = 0
    multi_group = 0
    for _, row in raw.iterrows():
        groups = sorted(
            {
                label_to_group[label_id_to_name[int(label_id)]]
                for label_id in row["labels"]
                if label_id_to_name[int(label_id)] in label_to_group
            }
        )
        if not groups:
            dropped_unmapped += 1
            continue
        if len(groups) > 1:
            multi_group += 1

        item: dict[str, object] = {
            "turn_index": len(rows),
            "timestamp": "",
            "sender": "ru_go_emotions",
            "text": str(row["ru_text"]),
        }
        for emotion in EMOTIONS:
            item[emotion] = (1.0 / len(groups)) if emotion in groups else 0.0
        rows.append(item)

    frame = pd.DataFrame(rows)
    stats = {
        "raw_rows": int(len(raw)),
        "kept_rows": int(len(frame)),
        "dropped_unmapped_rows": int(dropped_unmapped),
        "multi_group_rows": int(multi_group),
    }
    return frame, stats


def _evaluate(splits: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    runs: list[dict[str, object]] = []
    train = splits["train"]
    for model_name, factory in MODEL_FACTORIES.items():
        print(f"TRAIN {model_name}", flush=True)
        model = factory()
        if model_name != "dummy":
            model = train_from_full_frame(train, model)
            joblib.dump(model, output_dir / f"{model_name}.joblib")

        for split_name in ("validation", "test"):
            frame = splits[split_name]
            proba = model.predict_proba(frame["text"].astype(str).tolist())
            row: dict[str, object] = {
                "model": model_name,
                "split": split_name,
                "n": int(len(frame)),
            }
            row.update(_metrics(_y_matrix(frame), proba))
            runs.append(row)

    results = pd.DataFrame(runs)
    for column in ["top1_hit", "top1_primary_acc", "macro_f1_top1", "mae", "mse", "kl"]:
        results[column] = results[column].round(4)
    return results


def _y_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, EMOTIONS].to_numpy(dtype=float)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1.0)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    true_positive = y_true > 0
    pred_top = y_pred.argmax(axis=1)
    true_primary = y_true.argmax(axis=1)
    pred_onehot = np.zeros_like(y_true, dtype=int)
    pred_onehot[np.arange(len(pred_top)), pred_top] = 1
    return {
        "top1_hit": float(np.mean(true_positive[np.arange(len(pred_top)), pred_top])),
        "top1_primary_acc": float(np.mean(pred_top == true_primary)),
        "macro_f1_top1": float(
            f1_score((y_true > 0).astype(int), pred_onehot, average="macro", zero_division=0)
        ),
        "mae": float(np.abs(y_true - y_pred).mean()),
        "mse": float(((y_true - y_pred) ** 2).mean()),
        "kl": float(
            (y_true * (np.log(np.clip(y_true, eps, 1.0)) - np.log(y_pred)))
            .sum(axis=1)
            .mean()
        ),
    }


if __name__ == "__main__":
    main()
