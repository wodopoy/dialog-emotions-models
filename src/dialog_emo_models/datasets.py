"""Dataset loaders that map external corpora into the 6-emotion soft-label format.

`load_rugoemotions` is the primary train/val/test source (machine-translated
GoEmotions). `load_cedr` is a native-Russian cross-domain test; it only covers 4
of the 6 emotions (no warmth, surprise dropped), so warmth is never scored on it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dialog_emo_models.models import GOEMOTIONS_GROUPS, GOEMOTIONS_LABELS
from dialog_emo_models.schema import EMOTIONS

# CEDR (sagteam/cedr_v1) label ids -> our emotions. In the 6-class scheme
# 'surprise' has no target (dropped); in the 7-class scheme it maps in fully.
CEDR_LABELS = ("joy", "sadness", "surprise", "fear", "anger")
CEDR_TO_EMOTION = {"joy": "joy", "sadness": "sadness", "fear": "anxiety", "anger": "anger"}
if os.environ.get("EMO_SCHEME") == "7":
    CEDR_TO_EMOTION["surprise"] = "surprise"


def _soft_row(groups: list[str]) -> list[float]:
    return [(1.0 / len(groups)) if e in groups else 0.0 for e in EMOTIONS]


def load_rugoemotions(path: str | Path) -> tuple[list[str], NDArray[np.float64]]:
    raw = pd.read_parquet(path)
    id_to_name = dict(enumerate(GOEMOTIONS_LABELS))
    label_to_group = {
        label: group for group, labels in GOEMOTIONS_GROUPS.items() for label in labels
    }
    texts: list[str] = []
    labels: list[list[float]] = []
    for _, row in raw.iterrows():
        groups = sorted(
            {label_to_group[id_to_name[int(i)]] for i in row["labels"]
             if id_to_name[int(i)] in label_to_group}
        )
        if not groups:
            continue
        texts.append(str(row["ru_text"]))
        labels.append(_soft_row(groups))
    return texts, np.asarray(labels, dtype=float)


def load_cedr(path: str | Path) -> tuple[list[str], NDArray[np.float64]]:
    """Native-Russian cross-domain test. Empty labels -> neutral; surprise-only rows dropped."""
    raw = pd.read_parquet(path)
    texts: list[str] = []
    labels: list[list[float]] = []
    for _, row in raw.iterrows():
        ids = list(row["labels"])
        if not ids:
            groups = ["neutral"]
        else:
            groups = sorted(
                {CEDR_TO_EMOTION[CEDR_LABELS[int(i)]] for i in ids
                 if CEDR_LABELS[int(i)] in CEDR_TO_EMOTION}
            )
        if not groups:  # surprise-only -> unmappable
            continue
        texts.append(str(row["text"]))
        labels.append(_soft_row(groups))
    return texts, np.asarray(labels, dtype=float)
