from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import pandas as pd
from tqdm.auto import tqdm

from dialog_emo_models.metrics import (
    measure_latency_ms,
    measure_load_time_ms,
    model_size_mb,
    quality_metrics,
)
from dialog_emo_models.models import (
    GOEMOTIONS_GROUPS,
    GOEMOTIONS_LABELS,
    DummyEmotionModel,
    FastTextSupervisedEmotionModel,
    LabelPriorEmotionModel,
    LexiconEmotionModel,
    MajorityClassEmotionModel,
    TfidfLogRegEmotionModel,
    TfidfRidgeEmotionModel,
)
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

# All light, deployable models + reference floors. Heavy transformers live
# elsewhere; this harness is the comparable leaderboard for the light roster.
MODEL_FACTORIES = {
    "dummy": DummyEmotionModel,
    "majority": MajorityClassEmotionModel,
    "prior": LabelPriorEmotionModel,
    "lexicon": LexiconEmotionModel,
    "ridge-tfidf": TfidfRidgeEmotionModel,
    "logreg-tfidf": TfidfLogRegEmotionModel,
    "ridge-word-tfidf": lambda: TfidfRidgeEmotionModel(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50_000
    ),
    "logreg-word-tfidf": lambda: TfidfLogRegEmotionModel(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50_000, max_iter=500
    ),
    "ridge-word-char-tfidf": lambda: TfidfRidgeEmotionModel(analyzer="word+char"),
    "logreg-word-char-tfidf": lambda: TfidfLogRegEmotionModel(analyzer="word+char"),
    "fasttext-supervised": FastTextSupervisedEmotionModel,
}

LEADERBOARD_COLUMNS = [
    "model", "split", "n",
    "primary_accuracy", "macro_f1", "weighted_f1", "micro_f1", "top1_hit",
    "kl", "js", "mae", "mse", "ece",
    "size_mb", "load_ms", "latency_p50_ms", "latency_p95_ms",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark light models on ru_go_emotions.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("artifacts/datasets/ru_go_emotions/simplified"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments/light-leaderboard"),
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Subsample the train split for a quick smoke run.",
    )
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _download_missing_splits(args.data_dir)

    splits: dict[str, pd.DataFrame] = {}
    stats: dict[str, dict[str, int]] = {}
    for split in tqdm(("train", "validation", "test"), desc="convert splits", unit="split"):
        splits[split], stats[split] = _convert_split(args.data_dir / f"{split}.parquet")

    results = _evaluate(splits, args.output_dir, max_train_rows=args.max_train_rows)
    results = results.reindex(columns=LEADERBOARD_COLUMNS)
    pd.DataFrame(stats).T.to_csv(args.output_dir / "dataset_stats.csv")
    results.to_csv(args.output_dir / "metrics.csv", index=False)

    print("\nLEADERBOARD (validation, sorted by KL)")
    view = results[results["split"] == "validation"].sort_values("kl")
    print(view.to_string(index=False))
    print(f"\nARTIFACTS {args.output_dir}")


def _download_missing_splits(data_dir: Path) -> None:
    for split, url in tqdm(SPLIT_URLS.items(), desc="download splits", unit="split"):
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


def _evaluate(
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    max_train_rows: int | None = None,
) -> pd.DataFrame:
    train = splits["train"]
    if max_train_rows:
        train = train.head(max_train_rows)
    val_texts = splits["validation"]["text"].astype(str).tolist()

    rows: list[dict[str, object]] = []
    for model_name, factory in tqdm(MODEL_FACTORIES.items(), desc="models", unit="model"):
        print(f"RUN {model_name}", flush=True)
        try:
            model = factory()
            if model_name != "dummy":
                model = train_from_full_frame(train, model, show_progress=False)
            model_path = output_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path)

            deploy: dict[str, float] = {
                "size_mb": round(model_size_mb(model_path), 3),
                "load_ms": round(measure_load_time_ms(lambda p=model_path: joblib.load(p)), 2),
            }
            deploy.update(
                {key: round(value, 3) for key, value in measure_latency_ms(model, val_texts).items()}
            )

            for split_name in ("validation", "test"):
                frame = splits[split_name]
                proba = model.predict_proba(frame["text"].astype(str).tolist())
                metrics = quality_metrics(frame.loc[:, EMOTIONS].to_numpy(dtype=float), proba)
                row: dict[str, object] = {
                    "model": model_name,
                    "split": split_name,
                    "n": int(len(frame)),
                }
                row.update({key: round(value, 4) for key, value in metrics.items()})
                row.update(deploy)
                rows.append(row)
        except Exception as exc:  # keep the leaderboard partial-safe
            print(f"SKIP {model_name}: {type(exc).__name__}: {exc}", flush=True)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
