from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from dialog_emo_models.schema import EMOTIONS
from dialog_emo_models.models.base import EmotionModel


class DummyEmotionModel(EmotionModel):
    """Simple baseline model that produces uniform emotion probabilities."""

    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        return np.zeros((len(texts), len(EMOTIONS)), dtype=float)
