from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from numpy.typing import NDArray

from dialog_emo_models.models.base import EmotionModel, ModelT, validate_logits
from dialog_emo_models.schema import EMOTIONS
from dialog_emo_models.text import normalize_text

DEFAULT_PARAMS = {
    "lr": 0.5,
    "epoch": 25,
    "wordNgrams": 2,
    "dim": 100,
    "minn": 3,
    "maxn": 6,
    "loss": "softmax",
}
_LOGIT_FLOOR = 1e-9


def _import_fasttext():
    try:
        import fasttext
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise RuntimeError(
            "The fastText model requires the optional 'fasttext' dependency. "
            "Install it with: uv add fasttext-wheel  (or `uv sync --extra fasttext`)."
        ) from exc
    return fasttext


class FastTextSupervisedEmotionModel(EmotionModel):
    """Native fastText supervised classifier carried over from the thesis.

    Trains on the dominant emotion per row (argmax of the soft labels) using
    subword features, then quantizes to a compact `.ftz`. Per-label
    probabilities are returned as logits in the project's emotion order; the
    pipeline softmax leaves the distribution effectively unchanged.
    """

    def __init__(
        self,
        *,
        thread: int = 4,
        verbose: int = 0,
        params: dict[str, object] | None = None,
    ) -> None:
        self.thread = thread
        self.verbose = verbose
        self.params = dict(params) if params is not None else dict(DEFAULT_PARAMS)
        self._model = None

    def fit(
        self,
        texts: Sequence[str],
        labels: NDArray[np.float64] | None = None,
    ) -> FastTextSupervisedEmotionModel:
        if labels is None:
            raise ValueError("FastTextSupervisedEmotionModel.fit() requires labels")
        fasttext = _import_fasttext()
        target = validate_logits(labels, expected_rows=len(texts)).argmax(axis=1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            train_path = Path(tmp_dir) / "train.txt"
            with train_path.open("w", encoding="utf-8", newline="\n") as handle:
                for text, label_index in zip(texts, target):
                    line = normalize_text(text).replace("\n", " ").strip()
                    handle.write(f"__label__{EMOTIONS[int(label_index)]} {line}\n")
            model = fasttext.train_supervised(
                input=str(train_path),
                thread=self.thread,
                verbose=self.verbose,
                **self.params,
            )
            try:
                model.quantize(input=str(train_path), retrain=True, cutoff=100_000)
            except Exception:  # pragma: no cover - quantization is best effort
                pass
        self._model = model
        return self

    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        if self._model is None:
            raise RuntimeError("FastTextSupervisedEmotionModel must be fitted before prediction")
        logits = np.full((len(texts), len(EMOTIONS)), np.log(_LOGIT_FLOOR), dtype=float)
        index_by_emotion = {emotion: column for column, emotion in enumerate(EMOTIONS)}
        for row, text in enumerate(texts):
            clean = normalize_text(text).replace("\n", " ").strip()
            raw_labels, raw_probs = self._model.predict(clean, k=len(EMOTIONS))
            for raw_label, probability in zip(raw_labels, raw_probs):
                emotion = str(raw_label).replace("__label__", "", 1)
                column = index_by_emotion.get(emotion)
                if column is not None:
                    logits[row, column] = float(np.log(max(probability, _LOGIT_FLOOR)))
        return logits

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output)

    @classmethod
    def load(cls: type[ModelT], path: str | Path) -> ModelT:
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model).__name__}")
        return model

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        model = state.pop("_model", None)
        state["_model_bytes"] = None
        if model is not None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir) / "model.ftz"
                model.save_model(str(tmp))
                state["_model_bytes"] = tmp.read_bytes()
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        blob = state.pop("_model_bytes", None)
        self.__dict__.update(state)
        self._model = None
        if blob:
            fasttext = _import_fasttext()
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir) / "model.ftz"
                tmp.write_bytes(blob)
                self._model = fasttext.load_model(str(tmp))
