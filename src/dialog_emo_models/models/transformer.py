from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from dialog_emo_models.models.base import EmotionModel, ModelT, validate_logits
from dialog_emo_models.schema import EMOTIONS
from dialog_emo_models.text import normalize_text

DEFAULT_BASE_MODEL = "cointegrated/rubert-tiny2"


def _select_device(preference: str | None):
    import torch

    if preference is not None:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _reorder_logits_to_emotions(
    logits: NDArray[np.float64],
    id2label: dict[int | str, str],
) -> NDArray[np.float64]:
    """Permute classifier columns into the project's emotion order.

    If the model's labels match the six emotions (case-insensitive), columns are
    reordered to EMOTIONS. Otherwise the logits are returned unchanged, assuming
    they are already aligned.
    """
    logits = np.asarray(logits, dtype=float)
    labels = [str(id2label[key]).strip().lower() for key in sorted(id2label, key=int)]
    if logits.shape[1] != len(EMOTIONS) or set(labels) != {e.lower() for e in EMOTIONS}:
        return logits
    column_by_label = {label: column for column, label in enumerate(labels)}
    order = [column_by_label[emotion.lower()] for emotion in EMOTIONS]
    return logits[:, order]


class RuBertTiny2EmotionModel(EmotionModel):
    """RuBERT-tiny2 fine-tuned classifier, carried over from the thesis.

    Fine-tunes on the dominant emotion per row (argmax of the soft labels). This
    is the heavy trainable counterpart to the linear and fastText baselines.
    """

    def __init__(
        self,
        *,
        base_model: str = DEFAULT_BASE_MODEL,
        max_length: int = 96,
        epochs: int = 2,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        device: str | None = None,
    ) -> None:
        self.base_model = base_model
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.history: list[dict[str, float]] = []
        self._tokenizer = None
        self._model = None

    def fit(
        self,
        texts: Sequence[str],
        labels: NDArray[np.float64] | None = None,
    ) -> RuBertTiny2EmotionModel:
        if labels is None:
            raise ValueError("RuBertTiny2EmotionModel.fit() requires labels")

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        target = validate_logits(labels, expected_rows=len(texts)).argmax(axis=1)
        id2label = dict(enumerate(EMOTIONS))
        label2id = {emotion: index for index, emotion in enumerate(EMOTIONS)}

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=len(EMOTIONS),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        device = _select_device(self.device)
        model.to(device)

        encoded = tokenizer(
            [normalize_text(text) for text in texts],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        dataset = TensorDataset(
            encoded["input_ids"],
            encoded["attention_mask"],
            torch.tensor(target, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)

        self.history = []
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            for input_ids, attention_mask, batch_labels in loader:
                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    labels=batch_labels.to(device),
                )
                outputs.loss.backward()
                optimizer.step()
                total_loss += float(outputs.loss.item())
            self.history.append(
                {"epoch": epoch + 1, "loss": total_loss / max(1, len(loader))}
            )

        model.to("cpu")
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        return self

    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("RuBertTiny2EmotionModel must be fitted or loaded first")
        if not texts:
            return np.zeros((0, len(EMOTIONS)), dtype=float)

        import torch

        device = next(self._model.parameters()).device
        chunks = []
        normalized = [normalize_text(text) for text in texts]
        with torch.no_grad():
            for start in range(0, len(normalized), self.batch_size):
                batch = self._tokenizer(
                    normalized[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch = {key: value.to(device) for key, value in batch.items()}
                chunks.append(self._model(**batch).logits.detach().cpu().numpy())

        logits = np.concatenate(chunks, axis=0)
        return _reorder_logits_to_emotions(logits, self._model.config.id2label)

    def save(self, path: str | Path) -> None:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Cannot save an unfitted RuBertTiny2EmotionModel")
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(output)
        self._tokenizer.save_pretrained(output)
        (output / "metadata.json").write_text(
            json.dumps(
                {
                    "kind": "rubert_tiny2_finetune",
                    "base_model": self.base_model,
                    "labels": list(EMOTIONS),
                    "max_length": self.max_length,
                    "history": self.history,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls: type[ModelT], path: str | Path) -> ModelT:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path)
        metadata = {}
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = cls(
            base_model=metadata.get("base_model", DEFAULT_BASE_MODEL),
            max_length=metadata.get("max_length", 96),
        )
        model._tokenizer = AutoTokenizer.from_pretrained(path)
        model._model = AutoModelForSequenceClassification.from_pretrained(path)
        model._model.eval()
        model.history = metadata.get("history", [])
        return model
