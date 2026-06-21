"""Does the calibration dev-set choice matter? RuGo-only vs CEDR-only vs combined.

The main calibration (`calibrate_temperature.py`) fits T on RuGo validation, because
that is the only labelled split held out from weight training — CEDR ships only
train (used for the weights) and test (no val). But the 7-class checkpoints are
trained on RuGo+CEDR and deployed on native Russian, so calibrating T on the RuGo
(translationese) marginal alone is a domain mismatch — and indeed the RuGo-fit T
transfers imperfectly to CEDR.

This script answers the question honestly with a held-out CEDR split: CEDR test is
cut 50/50 into a `calib` half (used to fit T) and a `report` half (used to score),
so T-fit and reporting never overlap. For each model it fits three temperatures —

* `T_rugo`     : RuGo val               (in-domain only, the current default)
* `T_cedr`     : CEDR calib half        (target domain only)
* `T_combined` : RuGo val + CEDR calib  (matches the RuGo+CEDR training mixture)

— and reports ECE on held-out RuGo test and the held-out CEDR report half under each.
Fit is free (no deadband) so the dev-set effect is not masked.

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
from dialog_emo_models.metrics import apply_temperature, best_temperature, quality_metrics

CEDR_URL = ("https://huggingface.co/datasets/sagteam/cedr_v1/resolve/"
            "refs%2Fconvert%2Fparquet/main/test/0000.parquet")
HEAVY = {
    "rubert-tiny2-finetune", "hf-seara-rubert-tiny2",
    "hf-fyaronskiy-deberta", "hf-maxkazak-rubert-base", "tree-rf",
}
COLUMNS = [
    "model", "kind", "T_rugo", "T_cedr", "T_combined",
    "rugo_ece_raw", "rugo_ece_Trugo", "rugo_ece_Tcedr", "rugo_ece_Tcomb",
    "cedr_ece_raw", "cedr_ece_Trugo", "cedr_ece_Tcedr", "cedr_ece_Tcomb",
]


def _ece(y: np.ndarray, logits: np.ndarray, t: float) -> float:
    return quality_metrics(y, apply_temperature(logits, t))["ece"]


def _split_indices(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    half = n // 2
    return order[:half], order[half:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare calibration dev-set choices.")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/datasets/ru_go_emotions/simplified"))
    parser.add_argument("--cedr-dir", type=Path, default=Path("artifacts/datasets/cedr"))
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/calibration"))
    parser.add_argument("--skip-heavy", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cedr_dir.mkdir(parents=True, exist_ok=True)

    val_x, val_y = load_rugoemotions(args.data_dir / "validation.parquet")
    test_x, test_y = load_rugoemotions(args.data_dir / "test.parquet")
    cedr_path = args.cedr_dir / "test.parquet"
    if not cedr_path.exists():
        urlretrieve(CEDR_URL, cedr_path)
    cedr_x, cedr_y = load_cedr(cedr_path)
    calib_idx, report_idx = _split_indices(len(cedr_x))
    cc_x = [cedr_x[i] for i in calib_idx]
    cc_y = cedr_y[calib_idx]
    cr_x = [cedr_x[i] for i in report_idx]
    cr_y = cedr_y[report_idx]
    print(f"rugo val={len(val_x)} test={len(test_x)} | cedr calib={len(cc_x)} report={len(cr_x)}", flush=True)

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
            model = load_checkpoint(name, models_dir=args.models_dir)
            if hasattr(model, "temperature"):
                model.temperature = 1.0
            vl = np.asarray(model.predict_logits(val_x), dtype=float)
            tl = np.asarray(model.predict_logits(test_x), dtype=float)
            ccl = np.asarray(model.predict_logits(cc_x), dtype=float)
            crl = np.asarray(model.predict_logits(cr_x), dtype=float)
            del model
            gc.collect()

            t_rugo, _ = best_temperature(vl, val_y, objective="ece")
            t_cedr, _ = best_temperature(ccl, cc_y, objective="ece")
            # Combined dev = RuGo val + CEDR calib, stacked at natural sizes (≈ training mix).
            comb_logits = np.vstack([vl, ccl])
            comb_y = np.vstack([val_y, cc_y])
            t_comb, _ = best_temperature(comb_logits, comb_y, objective="ece")

            rows.append({
                "model": name, "kind": "heavy" if name in HEAVY else "light",
                "T_rugo": round(t_rugo, 3), "T_cedr": round(t_cedr, 3), "T_combined": round(t_comb, 3),
                "rugo_ece_raw": round(_ece(test_y, tl, 1.0), 4),
                "rugo_ece_Trugo": round(_ece(test_y, tl, t_rugo), 4),
                "rugo_ece_Tcedr": round(_ece(test_y, tl, t_cedr), 4),
                "rugo_ece_Tcomb": round(_ece(test_y, tl, t_comb), 4),
                "cedr_ece_raw": round(_ece(cr_y, crl, 1.0), 4),
                "cedr_ece_Trugo": round(_ece(cr_y, crl, t_rugo), 4),
                "cedr_ece_Tcedr": round(_ece(cr_y, crl, t_cedr), 4),
                "cedr_ece_Tcomb": round(_ece(cr_y, crl, t_comb), 4),
            })
            r = rows[-1]
            print(f"  T rugo/cedr/comb = {t_rugo:.2f}/{t_cedr:.2f}/{t_comb:.2f} | "
                  f"CEDR ECE raw {r['cedr_ece_raw']} -> Trugo {r['cedr_ece_Trugo']} / "
                  f"Tcomb {r['cedr_ece_Tcomb']} / Tcedr {r['cedr_ece_Tcedr']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {name}: {type(exc).__name__}: {exc}", flush=True)

    frame = pd.DataFrame(rows).reindex(columns=COLUMNS).sort_values("cedr_ece_raw", ascending=False)
    csv_path = args.output_dir / "calibration_devset_comparison.csv"
    frame.to_csv(csv_path, index=False)
    print("\nDEV-SET COMPARISON (CEDR report ECE; raw ECE descending)")
    print(frame.to_string(index=False))
    print(f"\nARTIFACTS {csv_path}")


if __name__ == "__main__":
    main()
