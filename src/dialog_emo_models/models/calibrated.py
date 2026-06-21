"""Temperature-calibration wrapper around any :class:`EmotionModel`.

Temperature scaling is post-hoc: it divides the model's pre-softmax logits by a
scalar ``T`` before the pipeline softmaxes them. This wrapper carries a fitted ``T``
alongside any inner model so that scoring a dialogue applies the calibrated
temperature automatically — the deployed checkpoints don't have to be re-pickled,
and callers that want the raw model can unwrap via ``.model``.

``temperature`` is a plain mutable attribute (same convention as
:class:`FastTextSupervisedEmotionModel`), so analysis code that re-fits T can reset
it to ``1.0`` to recover raw logits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from dialog_emo_models.models.base import EmotionModel


class CalibratedEmotionModel(EmotionModel):
    """Wrap an :class:`EmotionModel`, scaling its logits by a fitted temperature."""

    def __init__(self, model: EmotionModel, temperature: float = 1.0) -> None:
        self.model = model
        self.temperature = float(temperature)

    def fit(self, texts: Sequence[str], labels: NDArray[np.float64] | None = None) -> "CalibratedEmotionModel":
        self.model.fit(texts, labels)
        return self

    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        logits = np.asarray(self.model.predict_logits(texts), dtype=float)
        return logits / self.temperature

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "CalibratedEmotionModel":
        import joblib

        return joblib.load(path)

    def __repr__(self) -> str:
        return f"CalibratedEmotionModel({self.model!r}, temperature={self.temperature})"
