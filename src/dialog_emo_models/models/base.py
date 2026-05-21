from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray

from dialog_emo_models.schema import EMOTIONS

ModelT = TypeVar("ModelT", bound="EmotionModel")


class ModelOutputError(ValueError):
    """Raised when a model does not return logits in the expected shape."""


class EmotionModel(ABC):
    """Base contract for all dialogue emotion models.

    Implementations receive only message texts. The surrounding pipeline keeps
    dialogue metadata intact and converts logits into CSV probabilities.
    """

    def fit(
        self,
        texts: Sequence[str],
        labels: NDArray[np.float64] | None = None,
    ) -> EmotionModel:
        return self

    @abstractmethod
    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Return an array shaped `(len(texts), len(EMOTIONS))`."""

    def predict_proba(
        self,
        texts: Sequence[str],
        *,
        show_progress: bool = False,
    ) -> NDArray[np.float64]:
        _ = show_progress
        logits = validate_logits(self.predict_logits(texts), expected_rows=len(texts))
        return logits_to_probabilities(logits)

    def save(self, path: str | Path) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement save()")

    @classmethod
    def load(cls: type[ModelT], path: str | Path) -> ModelT:
        raise NotImplementedError(f"{cls.__name__} does not implement load()")


def validate_logits(
    logits: NDArray[np.float64] | Sequence[Sequence[float]],
    *,
    expected_rows: int,
) -> NDArray[np.float64]:
    array = np.asarray(logits, dtype=float)
    expected_shape = (expected_rows, len(EMOTIONS))
    if array.shape != expected_shape:
        raise ModelOutputError(
            f"Model logits must have shape {expected_shape}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ModelOutputError("Model logits must contain only finite values")
    return array


def logits_to_probabilities(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    if logits.shape[0] == 0:
        return logits.copy()
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)
