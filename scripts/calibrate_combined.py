"""Full dev-set comparison: fit T on {rugo_val, cedr_val, rugo_val+cedr_val},
report ECE on {rugo_test, cedr_test, rugo_test+cedr_test} for every checkpoint.

CEDR ships no val split (only train, used for the weights, and test), so cedr_val
is carved as one half of CEDR test and cedr_test is the other half — disjoint, and
both held out from weight training. Every dev-fit T uses the SAME deadband as the
main pipeline (ε=0.005), so all columns share one policy: a model already calibrated
on a dev set stays at T=1 there (its @T == raw) instead of chasing ECE noise. raw =
T=1 baseline.

    python scripts/calibrate_combined.py
    python scripts/calibrate_combined.py --only logreg-char,ridge-char,fasttext --skip-heavy
"""

from __future__ import annotations

import os

os.environ.setdefault("EMO_SCHEME", "7")  # checkpoints are 7-class

import argparse
import gc
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from dialog_emo_models.checkpoints import available_checkpoints, load_checkpoint
from dialog_emo_models.datasets import load_cedr, load_rugoemotions
from dialog_emo_models.metrics import apply_temperature, best_temperature, expected_calibration_error

CEDR_URL = ("https://huggingface.co/datasets/sagteam/cedr_v1/resolve/"
            "refs%2Fconvert%2Fparquet/main/test/0000.parquet")
HEAVY = {
    "rubert-tiny2-finetune", "hf-seara-rubert-tiny2",
    "hf-fyaronskiy-deberta", "hf-maxkazak-rubert-base", "tree-rf",
}
# Per test domain: raw + ECE under each of the three dev-fit temperatures.
TESTS = ("rugo", "cedr", "all")
DEVS = ("raw", "Trugo", "Tcedr", "Tcomb")
COLUMNS = ["model", "kind", "T_rugoval", "T_cedrval", "T_comb"] + [
    f"{t}_{d}" for t in TESTS for d in DEVS
]


def _ece(logits: np.ndarray, y: np.ndarray, t: float) -> float:
    return round(expected_calibration_error(y, apply_temperature(logits, t)), 4)


def _split(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    return order[: n // 2], order[n // 2:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare calibration dev-set choices (3x3 matrix).")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/datasets/ru_go_emotions/simplified"))
    parser.add_argument("--cedr-dir", type=Path, default=Path("artifacts/datasets/cedr"))
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/calibration"))
    parser.add_argument("--skip-heavy", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--min-improve", type=float, default=0.005,
                        help="Deadband on each dev-fit T (same as main pipeline; 0 = free fit).")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cedr_dir.mkdir(parents=True, exist_ok=True)

    rval_x, rval_y = load_rugoemotions(args.data_dir / "validation.parquet")
    rtest_x, rtest_y = load_rugoemotions(args.data_dir / "test.parquet")
    cedr_path = args.cedr_dir / "test.parquet"
    if not cedr_path.exists():
        urlretrieve(CEDR_URL, cedr_path)
    cedr_x, cedr_y = load_cedr(cedr_path)
    cval_idx, ctest_idx = _split(len(cedr_x))
    cval_x, cval_y = [cedr_x[i] for i in cval_idx], cedr_y[cval_idx]
    ctest_x, ctest_y = [cedr_x[i] for i in ctest_idx], cedr_y[ctest_idx]
    print(f"DEV  rugo_val={len(rval_x)}  cedr_val={len(cval_x)}  (rugo+cedr={len(rval_x)+len(cval_x)})", flush=True)
    print(f"TEST rugo_test={len(rtest_x)}  cedr_test={len(ctest_x)}  (all={len(rtest_x)+len(ctest_x)})", flush=True)

    checkpoints = available_checkpoints(args.models_dir)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        checkpoints = {n: p for n, p in checkpoints.items() if n in wanted}
    elif args.skip_heavy:
        checkpoints = {n: p for n, p in checkpoints.items() if n not in HEAVY}
    order = sorted(checkpoints, key=lambda n: (n in HEAVY, n))
    print(f"checkpoints: {', '.join(order)}", flush=True)

    rows: list[dict] = []
    for name in order:
        print(f"RUN {name}", flush=True)
        try:
            model = load_checkpoint(name, models_dir=args.models_dir, apply_temperature=False)
            if hasattr(model, "temperature"):
                model.temperature = 1.0
            rval_l = np.asarray(model.predict_logits(rval_x), dtype=float)
            rtest_l = np.asarray(model.predict_logits(rtest_x), dtype=float)
            cval_l = np.asarray(model.predict_logits(cval_x), dtype=float)
            ctest_l = np.asarray(model.predict_logits(ctest_x), dtype=float)
            del model
            gc.collect()

            # Three dev-fit temperatures, each with the main-pipeline deadband so all
            # columns share one policy (no mixing raw=deadband-on with @T=deadband-off).
            mi = args.min_improve
            t_rugo, _ = best_temperature(rval_l, rval_y, objective="ece", min_improve=mi)
            t_cedr, _ = best_temperature(cval_l, cval_y, objective="ece", min_improve=mi)
            t_comb, _ = best_temperature(
                np.vstack([rval_l, cval_l]), np.vstack([rval_y, cval_y]), objective="ece", min_improve=mi)
            temps = {"raw": 1.0, "Trugo": t_rugo, "Tcedr": t_cedr, "Tcomb": t_comb}

            # Three test domains, including the combined (rugo_test + cedr_test).
            test_logits = {
                "rugo": (rtest_l, rtest_y),
                "cedr": (ctest_l, ctest_y),
                "all": (np.vstack([rtest_l, ctest_l]), np.vstack([rtest_y, ctest_y])),
            }
            row = {"model": name, "kind": "heavy" if name in HEAVY else "light",
                   "T_rugoval": round(t_rugo, 3), "T_cedrval": round(t_cedr, 3), "T_comb": round(t_comb, 3)}
            for tname, (lg, y) in test_logits.items():
                for dname, t in temps.items():
                    row[f"{tname}_{dname}"] = _ece(lg, y, t)
            rows.append(row)
            print(f"  T rugo/cedr/comb = {t_rugo:.2f}/{t_cedr:.2f}/{t_comb:.2f} | "
                  f"cedr_test ECE raw {row['cedr_raw']} -> Trugo {row['cedr_Trugo']} "
                  f"Tcomb {row['cedr_Tcomb']} Tcedr {row['cedr_Tcedr']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {name}: {type(exc).__name__}: {exc}", flush=True)

    frame = pd.DataFrame(rows).reindex(columns=COLUMNS).sort_values("cedr_raw", ascending=False)
    csv_path = args.output_dir / "calibration_devset_comparison.csv"
    frame.to_csv(csv_path, index=False)

    pd.set_option("display.width", 200)
    for tname, title in (("rugo", "RuGo test"), ("cedr", "CEDR test"), ("all", "Combined test (rugo+cedr)")):
        cols = ["model", "T_rugoval", "T_cedrval", "T_comb"] + [f"{tname}_{d}" for d in DEVS]
        print(f"\n=== {title}: ECE under each dev-fit T ===")
        print(frame[cols].to_string(index=False))
    print(f"\nARTIFACTS {csv_path}")


if __name__ == "__main__":
    main()
