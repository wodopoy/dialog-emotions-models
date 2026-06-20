"""Save the FULL roster as 7-class checkpoints trained on rugo+cedr.

The 9 HPO winners are already saved by hand; this fills in the rest of the big
leaderboard: floors, both NB variants, tree ensembles, the HF presets (lazy
wrappers — weights come from the HF cache), and the fine-tuned RuBERT-tiny2.

    EMO_SCHEME=7 python scripts/save_all_models.py --with-rubert
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_combined import build  # noqa: E402

from dialog_emo_models.datasets import load_cedr, load_rugoemotions  # noqa: E402
from dialog_emo_models.models import (  # noqa: E402
    DummyEmotionModel,
    FyaronskiyDebertaGoEmotionsModel,
    LabelPriorEmotionModel,
    LexiconEmotionModel,
    MajorityClassEmotionModel,
    MaxKazakRuBertBaseGoEmotionsModel,
    RuBertTiny2EmotionModel,
    SearaRuBertTiny2GoEmotionsModel,
    TfidfTreeEmotionModel,
)
from dialog_emo_models.schema import EMOTIONS  # noqa: E402

OUT = Path("artifacts/models")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-rubert", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    dd = Path("artifacts/datasets/ru_go_emotions/simplified")
    cd = Path("artifacts/datasets/cedr")
    rugo = load_rugoemotions(dd / "train.parquet")
    cedr = load_cedr(cd / "train.parquet")
    X = rugo[0] + cedr[0]
    Y = np.vstack([rugo[1], cedr[1]])
    print(f"схема: {len(EMOTIONS)} классов; train {len(X)} строк\n", flush=True)

    def dump(name: str, model, fit: bool = True) -> None:
        if fit:
            model.fit(X, Y)
        path = OUT / f"{name}-7class-rugo-cedr.joblib"
        joblib.dump(model, path)
        print(f"  saved {name:24} -> {path.name} ({path.stat().st_size / 1e6:.2f} MB)", flush=True)

    # floors
    dump("dummy", DummyEmotionModel(), fit=False)
    dump("majority", MajorityClassEmotionModel())
    dump("prior", LabelPriorEmotionModel())
    dump("lexicon-hand", LexiconEmotionModel(), fit=False)

    # both NB variants (best config per kind from the HPO sweep)
    nb = pd.read_csv("artifacts/experiments/tune-combined/all_configs.csv")
    nb = nb[(nb["family"] == "nb") & nb["balanced_kl"].notna()].copy()
    nb["kind"] = nb["params"].apply(lambda s: json.loads(s).get("kind"))
    for kind in ("complement", "multinomial"):
        best = nb[nb["kind"] == kind].sort_values("balanced_kl").iloc[0]
        p = json.loads(best["params"])
        if "ngram_range" in p:
            p["ngram_range"] = tuple(p["ngram_range"])
        dump(f"nb-{kind}", build("nb", p))

    # tree ensembles (not HPO-tuned; standard config)
    dump("tree-hgb", TfidfTreeEmotionModel(estimator="hgb", svd_components=300))
    dump("tree-rf", TfidfTreeEmotionModel(estimator="rf", svd_components=300, n_estimators=300))

    # HF presets — lazy wrappers (weights load from the HF cache on use)
    dump("hf-seara-rubert-tiny2", SearaRuBertTiny2GoEmotionsModel(), fit=False)
    dump("hf-fyaronskiy-deberta", FyaronskiyDebertaGoEmotionsModel(), fit=False)
    dump("hf-maxkazak-rubert-base", MaxKazakRuBertBaseGoEmotionsModel(), fit=False)

    # heavy: fine-tune RuBERT-tiny2 (dir-based save_pretrained)
    if args.with_rubert:
        print("\n  training rubert-tiny2-finetune (2 epochs on rugo+cedr)...", flush=True)
        rubert = RuBertTiny2EmotionModel(epochs=2).fit(X, Y)
        rubert_dir = OUT / "rubert-tiny2-finetune-7class-rugo-cedr"
        rubert.save(rubert_dir)
        print(f"  saved rubert-tiny2-finetune -> {rubert_dir}/ (dir)", flush=True)

    print(f"\nГотово. Все в {OUT}/", flush=True)


if __name__ == "__main__":
    main()
