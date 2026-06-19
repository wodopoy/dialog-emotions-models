from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from dialog_emo_models.models.base import EmotionModel, validate_logits
from dialog_emo_models.schema import EMOTIONS

_LOG_FLOOR = 1e-9


class LabelPriorEmotionModel(EmotionModel):
    """Predicts the training label-prior distribution for every input.

    This is the honest probabilistic floor: the lowest KL / MAE a text-blind
    model can reach (the uniform `dummy` is not that floor). Stores the mean of
    the training soft-label matrix.
    """

    def __init__(self) -> None:
        self.prior: NDArray[np.float64] | None = None

    def fit(
        self,
        texts: Sequence[str],
        labels: NDArray[np.float64] | None = None,
    ) -> LabelPriorEmotionModel:
        if labels is None:
            raise ValueError("LabelPriorEmotionModel.fit() requires labels")
        matrix = validate_logits(labels, expected_rows=len(texts))
        prior = matrix.mean(axis=0)
        total = float(prior.sum())
        self.prior = prior / total if total > 0 else np.full(len(EMOTIONS), 1 / len(EMOTIONS))
        return self

    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        if self.prior is None:
            raise RuntimeError("LabelPriorEmotionModel must be fitted before prediction")
        row = np.log(np.clip(self.prior, _LOG_FLOOR, 1.0))
        return np.tile(row, (len(texts), 1))


class MajorityClassEmotionModel(EmotionModel):
    """Always predicts the single most frequent primary emotion (accuracy floor)."""

    def __init__(self) -> None:
        self.majority_index: int | None = None

    def fit(
        self,
        texts: Sequence[str],
        labels: NDArray[np.float64] | None = None,
    ) -> MajorityClassEmotionModel:
        if labels is None:
            raise ValueError("MajorityClassEmotionModel.fit() requires labels")
        matrix = validate_logits(labels, expected_rows=len(texts))
        counts = np.bincount(matrix.argmax(axis=1), minlength=len(EMOTIONS))
        self.majority_index = int(counts.argmax())
        return self

    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        if self.majority_index is None:
            raise RuntimeError("MajorityClassEmotionModel must be fitted before prediction")
        logits = np.full((len(texts), len(EMOTIONS)), -30.0, dtype=float)
        logits[:, self.majority_index] = 30.0
        return logits
