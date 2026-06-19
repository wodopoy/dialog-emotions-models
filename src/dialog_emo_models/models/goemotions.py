from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from dialog_emo_models.models.base import EmotionModel
from dialog_emo_models.schema import EMOTIONS

if os.environ.get("EMO_SCHEME") == "7":
    # 7-class scheme: adds `surprise` (recovers realization/curiosity/confusion
    # from the dropped set) and folds `relief` into joy. Makes CEDR map in fully.
    GOEMOTIONS_GROUPS: dict[str, list[str]] = {
        "joy": [
            "admiration", "amusement", "approval", "excitement",
            "joy", "optimism", "pride", "relief",
        ],
        "warmth": ["caring", "gratitude", "love"],
        "sadness": ["disappointment", "grief", "remorse", "sadness"],
        "anger": ["anger", "annoyance", "disapproval", "disgust"],
        "anxiety": ["fear", "nervousness", "embarrassment"],
        "surprise": ["surprise", "realization", "curiosity", "confusion"],
        "neutral": ["neutral"],
    }
else:
    GOEMOTIONS_GROUPS: dict[str, list[str]] = {
        "anxiety": ["fear", "nervousness", "embarrassment"],
        "anger": ["anger", "annoyance", "disapproval", "disgust"],
        "sadness": ["disappointment", "grief", "remorse", "sadness"],
        "warmth": ["caring", "gratitude", "love"],
        "joy": [
            "admiration",
            "amusement",
            "approval",
            "excitement",
            "joy",
            "optimism",
            "pride",
        ],
        "neutral": ["neutral"],
    }

GOEMOTIONS_LABELS = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)

SEARA_RUBERT_TINY2_GOEMOTIONS = (
    "seara/rubert-tiny2-russian-emotion-detection-ru-go-emotions"
)
FYARONSKIY_DEBERTA_GOEMOTIONS = "fyaronskiy/deberta-v1-base-russian-go-emotions"
MAXKAZAK_RUBERT_BASE_GOEMOTIONS = "MaxKazak/ruBert-base-russian-emotion-detection"


@dataclass(frozen=True)
class GoEmotionsPreset:
    name: str
    model_id: str
    labels: tuple[str, ...] = GOEMOTIONS_LABELS


GOEMOTIONS_PRESETS = {
    "hf-seara-rubert-tiny2-goemotions": GoEmotionsPreset(
        name="hf-seara-rubert-tiny2-goemotions",
        model_id=SEARA_RUBERT_TINY2_GOEMOTIONS,
    ),
    "hf-fyaronskiy-deberta-goemotions": GoEmotionsPreset(
        name="hf-fyaronskiy-deberta-goemotions",
        model_id=FYARONSKIY_DEBERTA_GOEMOTIONS,
    ),
    "hf-maxkazak-rubert-base-goemotions": GoEmotionsPreset(
        name="hf-maxkazak-rubert-base-goemotions",
        model_id=MAXKAZAK_RUBERT_BASE_GOEMOTIONS,
    ),
}


class GoEmotionsHFModel(EmotionModel):
    """Aggregate a Russian GoEmotions HF classifier into the six project emotions."""

    def __init__(
        self,
        model_id: str,
        *,
        labels: Sequence[str] | None = GOEMOTIONS_LABELS,
        max_length: int = 128,
        batch_size: int = 64,
        device: str | None = None,
        missing_group_logit: float = -30.0,
    ) -> None:
        self.model_id = model_id
        self.fallback_labels = tuple(labels) if labels is not None else None
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.missing_group_logit = missing_group_logit
        self._tokenizer = None
        self._model = None

    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        return self._predict_logits(texts, show_progress=False)

    def predict_proba(
        self,
        texts: Sequence[str],
        *,
        show_progress: bool = False,
    ) -> NDArray[np.float64]:
        from dialog_emo_models.models.base import logits_to_probabilities, validate_logits

        logits = validate_logits(
            self._predict_logits(texts, show_progress=show_progress),
            expected_rows=len(texts),
        )
        return logits_to_probabilities(logits)

    def _predict_logits(
        self,
        texts: Sequence[str],
        *,
        show_progress: bool,
    ) -> NDArray[np.float64]:
        if not texts:
            return np.zeros((0, len(EMOTIONS)), dtype=float)

        tokenizer, model = self._load_backend()

        import torch

        model_device = next(model.parameters()).device
        label_logits_chunks = []
        text_list = list(texts)
        starts = range(0, len(text_list), self.batch_size)
        iterator = tqdm(
            starts,
            total=(len(text_list) + self.batch_size - 1) // self.batch_size,
            desc=f"scoring {self.model_id}",
            unit="batch",
            disable=not show_progress,
        )
        with torch.no_grad():
            for start in iterator:
                batch = text_list[start : start + self.batch_size]
                inputs = tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(model_device) for key, value in inputs.items()}
                output = model(**inputs)
                label_logits_chunks.append(output.logits.detach().cpu().numpy())

        label_logits = np.concatenate(label_logits_chunks, axis=0)
        labels = self._labels_for_model(model, label_logits.shape[1])
        return aggregate_goemotions_logits(
            label_logits,
            labels,
            missing_group_logit=self.missing_group_logit,
        )

    def _load_backend(self):
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HF GoEmotions models require optional dependencies. "
                "Install them with: uv add transformers torch"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        target_device = self.device or (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        model.to(target_device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model

    def _labels_for_model(self, model, label_count: int) -> tuple[str, ...]:
        config_labels = _labels_from_config(model.config.id2label)
        if _looks_like_real_labels(config_labels) and len(config_labels) == label_count:
            return config_labels
        if self.fallback_labels is not None and len(self.fallback_labels) == label_count:
            return self.fallback_labels
        return config_labels


class SearaRuBertTiny2GoEmotionsModel(GoEmotionsHFModel):
    def __init__(self) -> None:
        super().__init__(SEARA_RUBERT_TINY2_GOEMOTIONS)


class FyaronskiyDebertaGoEmotionsModel(GoEmotionsHFModel):
    def __init__(self) -> None:
        super().__init__(FYARONSKIY_DEBERTA_GOEMOTIONS)


class MaxKazakRuBertBaseGoEmotionsModel(GoEmotionsHFModel):
    def __init__(self) -> None:
        super().__init__(MAXKAZAK_RUBERT_BASE_GOEMOTIONS)


def aggregate_goemotions_logits(
    label_logits: NDArray[np.float64] | Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    missing_group_logit: float = -30.0,
) -> NDArray[np.float64]:
    logits = np.asarray(label_logits, dtype=float)
    if logits.ndim != 2:
        raise ValueError(f"GoEmotions logits must be 2D, got shape {logits.shape}")
    if logits.shape[1] != len(labels):
        raise ValueError(
            f"Expected {len(labels)} labels for logits, got {logits.shape[1]}"
        )

    label_to_index = {_normalize_label(label): index for index, label in enumerate(labels)}
    grouped = np.full((logits.shape[0], len(EMOTIONS)), missing_group_logit, dtype=float)
    for group_index, group in enumerate(EMOTIONS):
        indexes = [
            label_to_index[label]
            for label in GOEMOTIONS_GROUPS[group]
            if label in label_to_index
        ]
        if indexes:
            grouped[:, group_index] = _logsumexp(logits[:, indexes], axis=1)
    return grouped


def missing_goemotions_labels(labels: Sequence[str]) -> dict[str, list[str]]:
    known = {_normalize_label(label) for label in labels}
    return {
        group: [label for label in group_labels if label not in known]
        for group, group_labels in GOEMOTIONS_GROUPS.items()
        if any(label not in known for label in group_labels)
    }


def _labels_from_config(id2label: dict[int | str, str]) -> tuple[str, ...]:
    return tuple(
        _normalize_label(id2label[key])
        for key in sorted(id2label, key=lambda value: int(value))
    )


def _looks_like_real_labels(labels: Sequence[str]) -> bool:
    return any(not label.startswith("label_") for label in labels)


def _normalize_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


def _logsumexp(values: NDArray[np.float64], *, axis: int) -> NDArray[np.float64]:
    shift = values.max(axis=axis, keepdims=True)
    return (shift + np.log(np.exp(values - shift).sum(axis=axis, keepdims=True))).squeeze(
        axis=axis
    )
