from __future__ import annotations

from collections.abc import Callable

from dialog_emo_models.models import DummyEmotionModel, EmotionModel

ModelFactory = Callable[[], EmotionModel]

MODEL_REGISTRY: dict[str, ModelFactory] = {
    "dummy": DummyEmotionModel,
}


def available_model_names() -> list[str]:
    return sorted(MODEL_REGISTRY)


def create_model(name: str) -> EmotionModel:
    try:
        return MODEL_REGISTRY[name]()
    except KeyError as exc:
        available = ", ".join(available_model_names())
        raise ValueError(f"Unknown model {name!r}. Available models: {available}") from exc


def register_model(name: str, factory: ModelFactory) -> None:
    if not name:
        raise ValueError("Model name must not be empty")
    MODEL_REGISTRY[name] = factory
