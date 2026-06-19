"""Domain adaptation: train on RuGoEmotions / CEDR / both, test on BOTH.

Three training regimes x two held-out test sets, so we see what adding native
CEDR data does to both domains (not just CEDR). Trainable models run all three
regimes; frozen HF presets are inference-only reference rows. Run under the
7-class scheme so CEDR maps in fully:

    EMO_SCHEME=7 python scripts/domain_adaptation.py
    EMO_SCHEME=7 python scripts/domain_adaptation.py --light-only --max-train-rows 1500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_leaderboard import HEAVY, LIGHT, _size_mb  # noqa: E402

from dialog_emo_models.datasets import load_cedr, load_rugoemotions  # noqa: E402
from dialog_emo_models.metrics import quality_metrics  # noqa: E402

CEDR_BASE = ("https://huggingface.co/datasets/sagteam/cedr_v1/resolve/"
             "refs%2Fconvert%2Fparquet/main/{split}/0000.parquet")
COLUMNS = ["model", "regime", "train_n",
           "rugo_acc", "rugo_macro_f1", "rugo_kl",
           "cedr_acc", "cedr_macro_f1", "cedr_kl", "size_mb"]


def _fetch_cedr(cedr_dir: Path, split: str) -> Path:
    path = cedr_dir / f"{split}.parquet"
    if not path.exists():
        urlretrieve(CEDR_BASE.format(split=split), path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/datasets/ru_go_emotions/simplified"))
    parser.add_argument("--cedr-dir", type=Path, default=Path("artifacts/datasets/cedr"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/domain-adaptation"))
    parser.add_argument("--light-only", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cedr_dir.mkdir(parents=True, exist_ok=True)

    rugo_train = load_rugoemotions(args.data_dir / "train.parquet")
    rugo_test = load_rugoemotions(args.data_dir / "test.parquet")
    cedr_train = load_cedr(_fetch_cedr(args.cedr_dir, "train"))
    cedr_test = load_cedr(_fetch_cedr(args.cedr_dir, "test"))

    if args.max_train_rows:
        n = args.max_train_rows
        rugo_train = (rugo_train[0][:n], rugo_train[1][:n])
        cedr_train = (cedr_train[0][:n], cedr_train[1][:n])

    regimes = {
        "A:rugo": rugo_train,
        "B:rugo+cedr": (rugo_train[0] + cedr_train[0],
                        np.vstack([rugo_train[1], cedr_train[1]])),
        "C:cedr": cedr_train,
    }
    print({k: len(v[0]) for k, v in regimes.items()},
          "| rugo_test", len(rugo_test[0]), "cedr_test", len(cedr_test[0]), flush=True)

    trainable = list(LIGHT) + [h for h in HEAVY if h[1] == "train"]
    frozen = [] if args.light_only else [h for h in HEAVY if h[1] == "infer"]
    tmp = args.output_dir / "_tmp.joblib"
    rows: list[dict] = []

    def evaluate(model, name, regime, train_n):
        rg = quality_metrics(rugo_test[1], model.predict_proba(rugo_test[0]))
        cd = quality_metrics(cedr_test[1], model.predict_proba(cedr_test[0]))
        return {"model": name, "regime": regime, "train_n": train_n,
                "rugo_acc": round(rg["primary_accuracy"], 4), "rugo_macro_f1": round(rg["macro_f1"], 4),
                "rugo_kl": round(rg["kl"], 4), "cedr_acc": round(cd["primary_accuracy"], 4),
                "cedr_macro_f1": round(cd["macro_f1"], 4), "cedr_kl": round(cd["kl"], 4),
                "size_mb": round(_size_mb(model, tmp), 2)}

    for name, _kind, factory in trainable:
        if args.light_only and name == "rubert-tiny2-finetune":
            continue
        for regime, (tx, ty) in regimes.items():
            print(f"RUN {name} [{regime}] n={len(tx)}", flush=True)
            try:
                model = factory().fit(tx, ty)
                rows.append(evaluate(model, name, regime, len(tx)))
                print(f"  rugo_acc={rows[-1]['rugo_acc']} cedr_acc={rows[-1]['cedr_acc']}", flush=True)
            except Exception as exc:
                print(f"  SKIP {name} [{regime}]: {type(exc).__name__}: {exc}", flush=True)

    for name, _kind, factory in frozen:
        print(f"RUN {name} [frozen]", flush=True)
        try:
            model = factory()
            rows.append(evaluate(model, name, "frozen", 0))
            print(f"  rugo_acc={rows[-1]['rugo_acc']} cedr_acc={rows[-1]['cedr_acc']}", flush=True)
        except Exception as exc:
            print(f"  SKIP {name} [frozen]: {type(exc).__name__}: {exc}", flush=True)

    frame = pd.DataFrame(rows).reindex(columns=COLUMNS)
    frame.to_csv(args.output_dir / "domain_adaptation.csv", index=False)
    print("\nDOMAIN ADAPTATION (train regime x test domain)")
    print(frame.to_string(index=False))
    print(f"\nARTIFACTS {args.output_dir}")


if __name__ == "__main__":
    main()
