"""Hyperparameter search for the light deployable models on RuGoEmotions.

Each model family is searched over a small grid, trained on the train split and
selected on validation by KL (tie-break MAE). Per-config size is recorded so the
size<->quality trade-off is visible; the per-family winners are then scored once
on the test split for the final leaderboard.

    python scripts/tune_light_models.py            # full search (~30-45 min)
    python scripts/tune_light_models.py --quick     # tiny grid smoke run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from dialog_emo_models.metrics import measure_latency_ms, model_size_mb, quality_metrics
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

BEST_COLUMNS = [
    "family", "params",
    "val_kl", "val_acc", "val_macro_f1", "val_ece",
    "test_kl", "test_acc", "test_macro_f1", "test_ece",
    "size_mb", "latency_p50_ms", "latency_p95_ms",
]


def build(spec: dict) -> object:
    params = spec["params"]
    builders = {
        "dummy": lambda: DummyEmotionModel(),
        "majority": lambda: MajorityClassEmotionModel(),
        "prior": lambda: LabelPriorEmotionModel(),
        "lexicon": lambda: LexiconEmotionModel(),
        "ridge": lambda: TfidfRidgeEmotionModel(**params),
        "logreg": lambda: TfidfLogRegEmotionModel(**params),
        "fasttext": lambda: FastTextSupervisedEmotionModel(params=params, thread=1),
    }
    return builders[spec["builder"]]()


def specs(quick: bool) -> list[dict]:
    out: list[dict] = []
    for family in ("dummy", "majority", "prior", "lexicon"):
        out.append({"family": family, "builder": family, "params": {}})

    mf = [50_000] if quick else [30_000, 50_000, 100_000]
    umf = [50_000] if quick else [20_000, 50_000, 100_000]

    for ngram in [(3, 5)] if quick else [(3, 5), (2, 6)]:
        for max_features in mf:
            for alpha in [1.0] if quick else [0.3, 1.0, 3.0]:
                out.append({"family": "ridge-char", "builder": "ridge", "params": dict(
                    analyzer="char_wb", ngram_range=ngram, min_df=2,
                    max_features=max_features, sublinear_tf=True, alpha=alpha)})

    for max_features in mf:
        for c in [1.0] if quick else [1.0, 2.0]:
            for cw in ["balanced"] if quick else [None, "balanced"]:
                out.append({"family": "logreg-char", "builder": "logreg", "params": dict(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                    max_features=max_features, sublinear_tf=True, C=c, class_weight=cw)})

    for max_features in umf:
        for alpha in [1.0] if quick else [1.0, 3.0]:
            out.append({"family": "ridge-union", "builder": "ridge", "params": dict(
                analyzer="word+char", min_df=2, max_features=max_features,
                sublinear_tf=True, alpha=alpha)})

    for max_features in umf:
        for cw in ["balanced"] if quick else [None, "balanced"]:
            out.append({"family": "logreg-union", "builder": "logreg", "params": dict(
                analyzer="word+char", min_df=2, max_features=max_features,
                sublinear_tf=True, C=1.0, class_weight=cw)})

    for max_features in [50_000] if quick else [30_000, 50_000]:
        for alpha in [1.0] if quick else [1.0, 3.0]:
            out.append({"family": "ridge-word", "builder": "ridge", "params": dict(
                analyzer="word", ngram_range=(1, 2), min_df=2,
                max_features=max_features, sublinear_tf=True, alpha=alpha)})

    for max_features in [50_000] if quick else [30_000, 50_000]:
        for cw in ["balanced"] if quick else [None, "balanced"]:
            out.append({"family": "logreg-word", "builder": "logreg", "params": dict(
                analyzer="word", ngram_range=(1, 2), min_df=2,
                max_features=max_features, sublinear_tf=True, C=1.0, class_weight=cw)})

    for lr in [0.5] if quick else [0.3, 0.5, 1.0]:
        for epoch in [25] if quick else [25, 50]:
            for loss in ["softmax"] if quick else ["softmax", "ova"]:
                out.append({"family": "fasttext", "builder": "fasttext", "params": dict(
                    lr=lr, epoch=epoch, wordNgrams=2, dim=100, minn=3, maxn=6, loss=loss)})

    return out


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune light models on ru_go_emotions.")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("artifacts/datasets/ru_go_emotions/simplified")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/experiments/light-tuning")
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_texts, train_labels = load_split(args.data_dir / "train.parquet")
    val_texts, val_labels = load_split(args.data_dir / "validation.parquet")
    test_texts, test_labels = load_split(args.data_dir / "test.parquet")
    if args.max_train_rows:
        train_texts = train_texts[: args.max_train_rows]
        train_labels = train_labels[: args.max_train_rows]
    print(f"train={len(train_texts)} val={len(val_texts)} test={len(test_texts)}", flush=True)

    all_specs = specs(args.quick)
    print(f"{len(all_specs)} configs", flush=True)
    tmp_path = args.output_dir / "_tmp.joblib"
    all_rows: list[dict] = []
    best: dict[str, dict] = {}

    for spec in tqdm(all_specs, desc="search", unit="cfg"):
        family, params = spec["family"], spec["params"]
        try:
            model = build(spec)
            model.fit(train_texts, train_labels)
            metrics = quality_metrics(val_labels, model.predict_proba(val_texts))
            joblib.dump(model, tmp_path)
            size = model_size_mb(tmp_path)
            all_rows.append({
                "family": family,
                "params": json.dumps(params, ensure_ascii=False),
                **{k: round(v, 4) for k, v in metrics.items()},
                "size_mb": round(size, 3),
            })
            if family not in best or metrics["kl"] < best[family]["val"]["kl"]:
                best[family] = {"model": model, "params": params, "val": metrics, "size_mb": size}
            print(f"  {family} kl={metrics['kl']:.4f} acc={metrics['primary_accuracy']:.4f} "
                  f"size={size:.1f}MB {json.dumps(params, ensure_ascii=False)}", flush=True)
        except Exception as exc:
            print(f"  SKIP {family} {params}: {type(exc).__name__}: {exc}", flush=True)
    tmp_path.unlink(missing_ok=True)

    best_rows: list[dict] = []
    for family, info in best.items():
        model = info["model"]
        test_metrics = quality_metrics(test_labels, model.predict_proba(test_texts))
        latency = measure_latency_ms(model, val_texts)
        joblib.dump(model, args.output_dir / f"best__{family}.joblib")
        best_rows.append({
            "family": family,
            "params": json.dumps(info["params"], ensure_ascii=False),
            "val_kl": round(info["val"]["kl"], 4),
            "val_acc": round(info["val"]["primary_accuracy"], 4),
            "val_macro_f1": round(info["val"]["macro_f1"], 4),
            "val_ece": round(info["val"]["ece"], 4),
            "test_kl": round(test_metrics["kl"], 4),
            "test_acc": round(test_metrics["primary_accuracy"], 4),
            "test_macro_f1": round(test_metrics["macro_f1"], 4),
            "test_ece": round(test_metrics["ece"], 4),
            "size_mb": round(info["size_mb"], 3),
            **{k: round(v, 3) for k, v in latency.items()},
        })

    pd.DataFrame(all_rows).to_csv(args.output_dir / "tuning_all_configs.csv", index=False)
    best_frame = pd.DataFrame(best_rows).reindex(columns=BEST_COLUMNS).sort_values("val_kl")
    best_frame.to_csv(args.output_dir / "tuning_best.csv", index=False)

    print("\nBEST PER FAMILY (selected by val KL)")
    print(best_frame.to_string(index=False))
    print(f"\nARTIFACTS {args.output_dir}")


if __name__ == "__main__":
    main()
