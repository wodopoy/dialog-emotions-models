"""One full leaderboard: every model on RuGoEmotions val/test AND native CEDR.

Single consistent pass so val/test/cedr and quality+deployment metrics are all
computed the same way. Heavy models are retrained/loaded here too.

    python scripts/full_leaderboard.py                 # everything (~30-50 min)
    python scripts/full_leaderboard.py --skip-heavy     # light models only
    python scripts/full_leaderboard.py --max-train-rows 2000 --skip-heavy  # smoke
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import pandas as pd

from dialog_emo_models.datasets import load_cedr, load_rugoemotions
from dialog_emo_models.metrics import measure_latency_ms, model_size_mb, quality_metrics
from dialog_emo_models.models import (
    DummyEmotionModel,
    FastTextSupervisedEmotionModel,
    FyaronskiyDebertaGoEmotionsModel,
    LabelPriorEmotionModel,
    LearnedLexiconEmotionModel,
    LexiconEmotionModel,
    MajorityClassEmotionModel,
    MaxKazakRuBertBaseGoEmotionsModel,
    RuBertTiny2EmotionModel,
    SearaRuBertTiny2GoEmotionsModel,
    TfidfLogRegEmotionModel,
    TfidfNaiveBayesEmotionModel,
    TfidfRidgeEmotionModel,
    TfidfTreeEmotionModel,
)

CEDR_URL = ("https://huggingface.co/datasets/sagteam/cedr_v1/resolve/"
            "refs%2Fconvert%2Fparquet/main/test/0000.parquet")

COLUMNS = ["model", "kind",
           "val_acc", "val_kl", "val_macro_f1", "val_ece",
           "test_acc", "test_kl", "test_macro_f1",
           "cedr_acc", "cedr_kl", "cedr_macro_f1",
           "size_mb", "latency_p50_ms", "latency_p95_ms"]

# name, kind(train|infer), factory
LIGHT = [
    ("dummy", "train", DummyEmotionModel),
    ("majority", "train", MajorityClassEmotionModel),
    ("prior", "train", LabelPriorEmotionModel),
    ("lexicon-hand", "train", LexiconEmotionModel),
    ("lexicon-learned", "train", lambda: LearnedLexiconEmotionModel(top_k=200, min_count=5)),
    ("ridge-char", "train", lambda: TfidfRidgeEmotionModel(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=100000, sublinear_tf=True, alpha=3.0)),
    ("logreg-char", "train", lambda: TfidfLogRegEmotionModel(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True, C=2.0, class_weight=None)),
    ("ridge-word", "train", lambda: TfidfRidgeEmotionModel(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True, alpha=3.0)),
    ("logreg-word", "train", lambda: TfidfLogRegEmotionModel(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True, C=1.0, class_weight=None)),
    ("ridge-union", "train", lambda: TfidfRidgeEmotionModel(
        analyzer="word+char", min_df=2, max_features=100000, sublinear_tf=True, alpha=3.0)),
    ("logreg-union", "train", lambda: TfidfLogRegEmotionModel(
        analyzer="word+char", min_df=2, max_features=20000, sublinear_tf=True, C=1.0, class_weight=None)),
    ("fasttext", "train", lambda: FastTextSupervisedEmotionModel(
        params={"lr": 0.3, "epoch": 25, "wordNgrams": 2, "dim": 100, "minn": 3, "maxn": 6, "loss": "ova"})),
    ("nb-complement", "train", lambda: TfidfNaiveBayesEmotionModel(kind="complement")),
    ("nb-multinomial", "train", lambda: TfidfNaiveBayesEmotionModel(kind="multinomial")),
    ("tree-hgb", "train", lambda: TfidfTreeEmotionModel(estimator="hgb", svd_components=300)),
    ("tree-rf", "train", lambda: TfidfTreeEmotionModel(estimator="rf", svd_components=300, n_estimators=300)),
]
HEAVY = [
    ("rubert-tiny2-finetune", "train", lambda: RuBertTiny2EmotionModel(epochs=2)),
    ("hf-seara-rubert-tiny2", "infer", SearaRuBertTiny2GoEmotionsModel),
    ("hf-fyaronskiy-deberta", "infer", FyaronskiyDebertaGoEmotionsModel),
    ("hf-maxkazak-rubert-base", "infer", MaxKazakRuBertBaseGoEmotionsModel),
]


def _size_mb(model, tmp: Path) -> float:
    torch_model = getattr(model, "_model", None)
    if torch_model is not None and hasattr(torch_model, "parameters"):
        return sum(p.numel() * p.element_size() for p in torch_model.parameters()) / (1024 * 1024)
    try:
        joblib.dump(model, tmp)
        size = model_size_mb(tmp)
        tmp.unlink(missing_ok=True)
        return size
    except Exception:
        return float("nan")


def _q(y, proba, prefix: str) -> dict:
    m = quality_metrics(y, proba)
    return {f"{prefix}_acc": round(m["primary_accuracy"], 4), f"{prefix}_kl": round(m["kl"], 4),
            f"{prefix}_macro_f1": round(m["macro_f1"], 4), f"{prefix}_ece": round(m["ece"], 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/datasets/ru_go_emotions/simplified"))
    parser.add_argument("--cedr-dir", type=Path, default=Path("artifacts/datasets/cedr"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/full-leaderboard"))
    parser.add_argument("--skip-heavy", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cedr_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y = load_rugoemotions(args.data_dir / "train.parquet")
    val_x, val_y = load_rugoemotions(args.data_dir / "validation.parquet")
    test_x, test_y = load_rugoemotions(args.data_dir / "test.parquet")
    if args.max_train_rows:
        train_x, train_y = train_x[: args.max_train_rows], train_y[: args.max_train_rows]
    cedr_path = args.cedr_dir / "test.parquet"
    if not cedr_path.exists():
        urlretrieve(CEDR_URL, cedr_path)
    cedr_x, cedr_y = load_cedr(cedr_path)
    print(f"train={len(train_x)} val={len(val_x)} test={len(test_x)} cedr={len(cedr_x)}", flush=True)

    jobs = LIGHT if args.skip_heavy else LIGHT + HEAVY
    heavy_names = {name for name, _, _ in HEAVY}
    tmp = args.output_dir / "_tmp.joblib"
    rows = []
    for name, kind, factory in jobs:
        print(f"RUN {name} ({kind})", flush=True)
        try:
            model = factory()
            if kind == "train":
                model.fit(train_x, train_y)
            if hasattr(model, "temperature"):  # calibrate fastText on val
                best_t, best_kl = 1.0, float("inf")
                for t in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
                    model.temperature = t
                    kl = quality_metrics(val_y, model.predict_proba(val_x))["kl"]
                    if kl < best_kl:
                        best_t, best_kl = t, kl
                model.temperature = best_t
            sample = 60 if kind == "infer" or name.startswith("rubert") else 150
            row = {"model": name, "kind": "heavy" if name in heavy_names else "light"}
            row.update(_q(val_y, model.predict_proba(val_x), "val"))
            row.update(_q(test_y, model.predict_proba(test_x), "test"))
            row.update(_q(cedr_y, model.predict_proba(cedr_x), "cedr"))
            row["size_mb"] = round(_size_mb(model, tmp), 2)
            lat = measure_latency_ms(model, val_x, sample=sample, repeats=2)
            row["latency_p50_ms"] = round(lat["latency_p50_ms"], 3)
            row["latency_p95_ms"] = round(lat["latency_p95_ms"], 3)
            rows.append(row)
            print(f"  val_acc={row['val_acc']} cedr_acc={row['cedr_acc']} size={row['size_mb']}MB", flush=True)
            del model
            gc.collect()
        except Exception as exc:
            print(f"  SKIP {name}: {type(exc).__name__}: {exc}", flush=True)

    frame = pd.DataFrame(rows).reindex(columns=COLUMNS).sort_values("val_kl")
    frame.to_csv(args.output_dir / "full_leaderboard.csv", index=False)
    print("\nFULL LEADERBOARD (val KL ascending)")
    print(frame.to_string(index=False))
    print(f"\nARTIFACTS {args.output_dir}")


if __name__ == "__main__":
    main()
