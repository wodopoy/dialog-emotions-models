"""Large randomized HPO for the LIGHT models on RuGo+CEDR, parallel across cores.

Heavy transformers are deliberately excluded (slow, insensitive, near ceiling).
Configs are independent, so they run across a process pool — one BLAS thread per
worker to avoid oversubscription. Selected by a balanced (RuGo + CEDR) validation
KL; the per-family winners are scored on both held-out test sets.

    EMO_SCHEME=7 python scripts/tune_combined.py --k-scale 2.5
    EMO_SCHEME=7 python scripts/tune_combined.py --quick --max-train-rows 1500 --workers 4
"""

from __future__ import annotations

import os

# Limit BLAS threads per process BEFORE numpy/sklearn import so N workers ~= N cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402
from urllib.request import urlretrieve  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from dialog_emo_models.datasets import load_cedr, load_rugoemotions  # noqa: E402
from dialog_emo_models.metrics import measure_latency_ms, model_size_mb, quality_metrics  # noqa: E402
from dialog_emo_models.models import (  # noqa: E402
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

_DATA: dict = {}  # per-worker globals


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
    for _ in range(k * 60):
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


def _carve_cedr(cedr_full):
    rng = np.random.default_rng(42)
    order = rng.permutation(len(cedr_full[0]))
    n_val = max(1, int(0.15 * len(order)))
    vi, fi = order[:n_val], order[n_val:]
    cedr_val = ([cedr_full[0][i] for i in vi], cedr_full[1][vi])
    cedr_fit = ([cedr_full[0][i] for i in fi], cedr_full[1][fi])
    return cedr_fit, cedr_val


def _init_worker(data_dir: str, cedr_dir: str, max_train_rows: int | None) -> None:
    rugo_tr = load_rugoemotions(Path(data_dir) / "train.parquet")
    rugo_val = load_rugoemotions(Path(data_dir) / "validation.parquet")
    cedr_fit, cedr_val = _carve_cedr(load_cedr(Path(cedr_dir) / "train.parquet"))
    if max_train_rows:
        rugo_tr = (rugo_tr[0][:max_train_rows], rugo_tr[1][:max_train_rows])
        cedr_fit = (cedr_fit[0][:max_train_rows], cedr_fit[1][:max_train_rows])
    _DATA["train"] = (rugo_tr[0] + cedr_fit[0], np.vstack([rugo_tr[1], cedr_fit[1]]))
    _DATA["rugo_val"] = rugo_val
    _DATA["cedr_val"] = cedr_val


def _run_config(task: tuple) -> dict:
    family, builder, p = task
    base = {"family": family, "params": json.dumps(p, sort_keys=True, default=str)}
    try:
        model = build(builder, p).fit(*_DATA["train"])
        rv = quality_metrics(_DATA["rugo_val"][1], model.predict_proba(_DATA["rugo_val"][0]))
        cv = quality_metrics(_DATA["cedr_val"][1], model.predict_proba(_DATA["cedr_val"][0]))
        base.update({
            "rugo_val_kl": round(rv["kl"], 4), "rugo_val_acc": round(rv["primary_accuracy"], 4),
            "cedr_val_kl": round(cv["kl"], 4), "cedr_val_acc": round(cv["primary_accuracy"], 4),
            "balanced_kl": round(0.5 * (rv["kl"] + cv["kl"]), 4)})
    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
    return base


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
    parser.add_argument("--k-scale", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    parser.add_argument("--max-train-rows", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cedr_dir.mkdir(parents=True, exist_ok=True)

    # ensure data is downloaded once (workers read from disk)
    _fetch_cedr(args.cedr_dir, "train")
    _fetch_cedr(args.cedr_dir, "test")
    rugo_te = load_rugoemotions(args.data_dir / "test.parquet")
    cedr_te = load_cedr(args.cedr_dir / "test.parquet")

    rng = np.random.default_rng(123)
    configs: list[tuple] = []
    for family, (builder, k, space) in FAMILIES.items():
        for p in sample_configs(space, 3 if args.quick else int(round(k * args.k_scale)), rng):
            configs.append((family, builder, p))
    print(f"{len(configs)} configs | {args.workers} workers", flush=True)

    results: list[dict] = []
    ckpt = args.output_dir / "all_configs.csv"
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker,
        initargs=(str(args.data_dir), str(args.cedr_dir), args.max_train_rows),
    ) as ex:
        for i, row in enumerate(tqdm(ex.map(_run_config, configs), total=len(configs), unit="cfg")):
            results.append(row)
            if (i + 1) % 100 == 0:
                pd.DataFrame(results).to_csv(ckpt, index=False)
    pd.DataFrame(results).to_csv(ckpt, index=False)

    # winners per family (refit in parent for test metrics)
    valid = [r for r in results if "balanced_kl" in r]
    by_family: dict[str, dict] = {}
    cfg_lookup = {(f, json.dumps(p, sort_keys=True, default=str)): (b, p) for f, b, p in configs}
    for r in valid:
        f = r["family"]
        if f not in by_family or r["balanced_kl"] < by_family[f]["balanced_kl"]:
            by_family[f] = r

    cedr_fit, _ = _carve_cedr(load_cedr(args.cedr_dir / "train.parquet"))
    rugo_tr = load_rugoemotions(args.data_dir / "train.parquet")
    train = (rugo_tr[0] + cedr_fit[0], np.vstack([rugo_tr[1], cedr_fit[1]]))
    tmp = args.output_dir / "_tmp.joblib"
    best_rows = []
    for family, r in by_family.items():
        builder, p = cfg_lookup[(family, r["params"])]
        model = build(builder, p).fit(*train)
        rt = quality_metrics(rugo_te[1], model.predict_proba(rugo_te[0]))
        ct = quality_metrics(cedr_te[1], model.predict_proba(cedr_te[0]))
        joblib.dump(model, tmp)
        size = round(model_size_mb(tmp), 2)
        tmp.unlink(missing_ok=True)
        lat = measure_latency_ms(model, rugo_te[0], sample=150, repeats=2)
        best_rows.append({
            "family": family, "params": r["params"], "balanced_val_kl": r["balanced_kl"],
            "rugo_test_acc": round(rt["primary_accuracy"], 4), "rugo_test_kl": round(rt["kl"], 4),
            "cedr_test_acc": round(ct["primary_accuracy"], 4), "cedr_test_kl": round(ct["kl"], 4),
            "size_mb": size, "latency_p50_ms": round(lat["latency_p50_ms"], 3)})

    best_frame = pd.DataFrame(best_rows).sort_values("cedr_test_acc", ascending=False)
    best_frame.to_csv(args.output_dir / "best_per_family.csv", index=False)
    print("\nBEST PER FAMILY on rugo+cedr (sorted by CEDR test acc)")
    print(best_frame.to_string(index=False))
    print(f"\nARTIFACTS {args.output_dir}")


if __name__ == "__main__":
    main()
