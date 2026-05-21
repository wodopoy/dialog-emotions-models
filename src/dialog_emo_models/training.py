from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dialog_emo_models.models import (
    EmotionModel,
    TfidfLogRegEmotionModel,
    TfidfRidgeEmotionModel,
)
from dialog_emo_models.schema import EMOTIONS, load_full_csv, validate_full_frame

TrainableModelFactory = Callable[[], EmotionModel]

TRAINABLE_MODEL_REGISTRY: dict[str, TrainableModelFactory] = {
    "logreg-tfidf": TfidfLogRegEmotionModel,
    "ridge-tfidf": TfidfRidgeEmotionModel,
}


def available_trainable_model_names() -> list[str]:
    return sorted(TRAINABLE_MODEL_REGISTRY)


def create_trainable_model(name: str) -> EmotionModel:
    try:
        return TRAINABLE_MODEL_REGISTRY[name]()
    except KeyError as exc:
        available = ", ".join(available_trainable_model_names())
        raise ValueError(
            f"Unknown trainable model {name!r}. Available models: {available}"
        ) from exc


def labels_from_full_frame(frame: pd.DataFrame) -> NDArray[np.float64]:
    validated = validate_full_frame(frame)
    return validated.loc[:, EMOTIONS].to_numpy(dtype=float)


def train_from_full_frame(frame: pd.DataFrame, model: EmotionModel) -> EmotionModel:
    validated = validate_full_frame(frame)
    texts = validated["text"].tolist()
    labels = labels_from_full_frame(validated)
    return model.fit(texts, labels)


def train_from_full_csv(
    input_path: str | Path,
    *,
    model_name: str,
    output_path: str | Path,
) -> EmotionModel:
    frame = load_full_csv(input_path)
    model = train_from_full_frame(frame, create_trainable_model(model_name))
    model.save(output_path)
    return model
