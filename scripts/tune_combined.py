"""Large randomized HPO for the LIGHT models on RuGo vs RuGo+CEDR.

Heavy transformers are deliberately excluded: they are slow, insensitive to these
sweeps, and near their ceiling. Light models get a wide randomized search over
both training sets, selected by a balanced (RuGo + CEDR) validation KL, so we can
see whether the optimal params shift once native CEDR data is mixed in.

    EMO_SCHEME=7 python scripts/tune_combined.py                 # full (hours)
    EMO_SCHEME=7 python scripts/tune_combined.py --quick --max-train-rows 1500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from dialog_emo_models.datasets import load_cedr, load_rugoemotions
from dialog_emo_models.metrics import measure_latency_ms, model_size_mb, quality_metrics
from dialog_emo_models.models import (
    FastTextSupervisedEmotionModel,
    LearnedLexiconEmotionModel,
    TfidfLogRegEmotionModel,
    TfidfNaiveBayesEmotionModel,
    TfidfRidgeEmotionModel,
)

CEDR_BASE = ("https://huggingface.co/datasets/sagteam/cedr_v1/resolve/"
             "refs%2Fconvert%2Fparquet/main/{split}/0000.parquet")

CHAR_NGRAMS = [(2, 5), (3, 5), (2, 6), (3, 6)]
WORD_NGRAMS = [(1, 1), (1, 2), (1, 3)]

# family -> (builder, K configs, search space)
FAMILIES = {
    "logreg-char": ("logreg", 45, {
        "analyzer": ["char_wb"], "ngram_range": CHAR_NGRAMS, "min_df": [1, 2, 5],
        "max_features": [10000, 20000, 30000, 50000, 100000], "sublinear_tf": [True, False],
        "C": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0], "class_weight": [None, "balanced"]}),
    "logreg-union": ("logreg", 35, {
        "analyzer": ["word+char"], "min_df": [1, 2, 5],
        "max_features": [10000, 20000, 30000, 50000, 100000], "sublinear_tf": [True, False],
        "C": [0.25, 0.5, 1.0, 2.0, 4.0], "class_weight": [None, "balanced"]}),
    "logreg-word": ("logreg", 25, {
        "analyzer": ["word"], "ngram_range": WORD_NGRAMS, "min_df": [1, 2, 5],
        "max_features": [20000, 30000, 50000, 100000], "sublinear_tf": [True, False],
        "C": [0.25, 0.5, 1.0, 2.0, 4.0], "class_weight": [None, "balanced"]}),
    "ridge-char": ("ridge", 55, {
        "analyzer": ["char_wb"], "ngram_range": CHAR_NGRAMS, "min_df": [1, 2, 5],
        "max_features": [20000, 30000, 50000, 100000], "sublinear_tf": [True, False],
        "alpha": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]}),
    "ridge-union": ("ridge", 35, {
        "analyzer": ["word+char"], "min_df": [1, 2, 5],
        "max_features": [10000, 20000, 50000, 100000], "sublinear_tf": [True, False],
        "alpha": [0.3, 1.0, 3.0, 10.0, 30.0]}),
    "ridge-word": ("ridge", 25, {
        "analyzer": ["word"], "ngram_range": WORD_NGRAMS, "min_df": [1, 2, 5],
        "max_features": [20000, 50000, 100000], "sublinear_tf": [True, False],
        "alpha": [0.3, 1.0, 3.0, 10.0, 30.0]}),
    "fasttext": ("fasttext", 45, {
        "lr": [0.1, 0.2, 0.3, 0.5, 0.8, 1.0], "epoch": [15, 25, 50, 75], "dim": [50, 100, 200],
        "wordNgrams": [1, 2, 3], "minn_maxn": [(2, 5), (3, 6), (2, 6)],
        "loss": ["softmax", "ova", "hs"]}),
    "nb": ("nb", 45, {
        "kind": ["complement", "multinomial"], "analyzer": ["char_wb", "word", "word+char"],
        "ngram_range": [(2, 5), (3, 5), (3, 6)], "min_df": [1, 2, 5],
        "max_features": [20000, 50000, 100000], "sublinear_tf": [True, False],
        "alpha": [0.1, 0.3, 0.5, 1.0]}),
    "lexicon-learned": ("lexicon", 18, {
        "top_k": [50, 100, 200, 400, 800, 1500], "min_count": [3, 5, 10],
        "alpha": [0.5, 1.0, 2.0]}),
}


def build(builder: str, p: dict):
    if builder == "logreg":
        return TfidfLogRegEmotionModel(
            analyzer=p["analyzer"], ngram_range=p.get("ngram_range", (3, 5)), min_df=p["min_df"],
            max_features=p["max_features"], sublinear_tf=p["sublinear_tf"], C=p["C"],
            class_weight=p["class_weight"])
    if builder == "ridge":
        return TfidfRidgeEmotionModel(
            analyzer=p["analyzer"], ngram_range=p.get("ngram_range", (3, 5)), min_df=p["min_df"],
            max_features=p["max_features"], sublinear_tf=p["sublinear_tf"], alpha=p["alpha"])
    if builder == "fasttext":
        return FastTextSupervisedEmotionModel(thread=1, params={
            "lr": p["lr"], "epoch": p["epoch"], "dim": p["dim"], "wordNgrams": p["wordNgrams"],
            "minn": p["minn"], "maxn": p["maxn"], "loss": p["loss"]})
    if builder == "nb":
        return TfidfNaiveBayesEmotionModel(
            kind=p["kind"], analyzer=p["analyzer"], ngram_range=p.get("ngram_range", (3, 5)),
            min_df=p["min_df"], max_features=p["max_features"], sublinear_tf=p["sublinear_tf"],
            alpha=p["alpha"])
    return LearnedLexiconEmotionModel(top_k=p["top_k"], min_count=p["min_count"], alpha=p["alpha"])


def sample_configs(space: dict, k: int, rng) -> list[dict]:
    keys = list(space)
    seen: set[str] = set()
    out: list[dict] = []
    for _ in range(k * 40):
        if len(out) >= k:
            break
        p = {key: space[key][int(rng.integers(len(space[key])))] for key in keys}
        if "minn_maxn" in p:
            p["minn"], p["maxn"] = p.pop("minn_maxn")
        sig = json.dumps(p, sort_keys=True, default=str)
        if sig not in seen:
            seen.add(sig)
            out.append(p)
    return out


def _fetch_cedr(cedr_dir: Path, split: str) -> Path:
    path = cedr_dir / f"{split}.parquet"
    if not path.exists():
        urlretrieve(CEDR_BASE.format(split=split), path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/datasets/ru_go_emotions/simplified"))
    parser.add_argument("--cedr-dir", type=Path, default=Path("artifacts/datasets/cedr"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/tune-combined"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cedr_dir.mkdir(parents=True, exist_ok=True)

    rugo_tr = load_rugoemotions(args.data_dir / "train.parquet")
    rugo_val = load_rugoemotions(args.data_dir / "validation.parquet")
    rugo_te = load_rugoemotions(args.data_dir / "test.parquet")
    cedr_tr_full = load_cedr(_fetch_cedr(args.cedr_dir, "train"))
    cedr_te = load_cedr(_fetch_cedr(args.cedr_dir, "test"))

    # carve a CEDR validation slice out of CEDR-train (test stays untouched)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(cedr_tr_full[0]))
    n_val = max(1, int(0.15 * len(order)))
    val_idx, fit_idx = order[:n_val], order[n_val:]
    cedr_val = ([cedr_tr_full[0][i] for i in val_idx], cedr_tr_full[1][val_idx])
    cedr_fit = ([cedr_tr_full[0][i] for i in fit_idx], cedr_tr_full[1][fit_idx])

    if args.max_train_rows:
        n = args.max_train_rows
        rugo_tr = (rugo_tr[0][:n], rugo_tr[1][:n])
        cedr_fit = (cedr_fit[0][:n], cedr_fit[1][:n])

    train_sets = {
        "rugo": rugo_tr,
        "rugo+cedr": (rugo_tr[0] + cedr_fit[0], np.vstack([rugo_tr[1], cedr_fit[1]])),
    }
    print({k: len(v[0]) for k, v in train_sets.items()},
          f"| rugo_val {len(rugo_val[0])} cedr_val {len(cedr_val[0])}"
          f" | rugo_test {len(rugo_te[0])} cedr_test {len(cedr_te[0])}", flush=True)

    all_rows: list[dict] = []
    best: dict[tuple, dict] = {}
    for family, (builder, k, space) in FAMILIES.items():
        configs = sample_configs(space, 3 if args.quick else k, rng)
        for ts_name, (tx, ty) in train_sets.items():
            for p in tqdm(configs, desc=f"{family}/{ts_name}", unit="cfg"):
                try:
                    model = build(builder, p).fit(tx, ty)
                    rv = quality_metrics(rugo_val[1], model.predict_proba(rugo_val[0]))
                    cv = quality_metrics(cedr_val[1], model.predict_proba(cedr_val[0]))
                    balanced = 0.5 * (rv["kl"] + cv["kl"])
                    row = {"family": family, "train_set": ts_name,
                           "params": json.dumps(p, sort_keys=True, default=str),
                           "rugo_val_kl": round(rv["kl"], 4), "rugo_val_acc": round(rv["primary_accuracy"], 4),
                           "cedr_val_kl": round(cv["kl"], 4), "cedr_val_acc": round(cv["primary_accuracy"], 4),
                           "balanced_kl": round(balanced, 4)}
                    all_rows.append(row)
                    key = (family, ts_name)
                    if key not in best or balanced < best[key]["balanced"]:
                        best[key] = {"model": model, "p": p, "balanced": balanced, "row": row}
                except Exception as exc:
                    print(f"  SKIP {family}/{ts_name} {p}: {type(exc).__name__}: {exc}", flush=True)
        pd.DataFrame(all_rows).to_csv(args.output_dir / "all_configs.csv", index=False)  # checkpoint
        print(f"DONE {family}: {len(configs)} configs x2 train-sets", flush=True)

    tmp = args.output_dir / "_tmp.joblib"
    best_rows: list[dict] = []
    for (family, ts_name), info in best.items():
        model = info["model"]
        rt = quality_metrics(rugo_te[1], model.predict_proba(rugo_te[0]))
        ct = quality_metrics(cedr_te[1], model.predict_proba(cedr_te[0]))
        joblib.dump(model, tmp)
        size = round(model_size_mb(tmp), 2)
        tmp.unlink(missing_ok=True)
        lat = measure_latency_ms(model, rugo_val[0], sample=150, repeats=2)
        best_rows.append({
            "family": family, "train_set": ts_name, "params": json.dumps(info["p"], sort_keys=True, default=str),
            "balanced_val_kl": round(info["balanced"], 4),
            "rugo_test_acc": round(rt["primary_accuracy"], 4), "rugo_test_kl": round(rt["kl"], 4),
            "cedr_test_acc": round(ct["primary_accuracy"], 4), "cedr_test_kl": round(ct["kl"], 4),
            "size_mb": size, "latency_p50_ms": round(lat["latency_p50_ms"], 3)})

    best_frame = pd.DataFrame(best_rows).sort_values(["family", "train_set"])
    best_frame.to_csv(args.output_dir / "best_per_family.csv", index=False)
    print("\nBEST PER FAMILY x TRAIN-SET (selected by balanced RuGo+CEDR val KL)")
    print(best_frame.to_string(index=False))
    print(f"\nARTIFACTS {args.output_dir}")


if __name__ == "__main__":
    main()
