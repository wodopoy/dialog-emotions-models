from __future__ import annotations

import numpy as np
import pytest

from dialog_emo_models.models import (
    GOEMOTIONS_LABELS,
    ModelOutputError,
    aggregate_goemotions_logits,
    logits_to_probabilities,
    missing_goemotions_labels,
    validate_logits,
)
from dialog_emo_models.registry import available_model_names, create_model
from dialog_emo_models.schema import EMOTIONS


def test_registry_contains_all_goemotions_presets() -> None:
    names = available_model_names()

    assert "hf-seara-rubert-tiny2-goemotions" in names
    assert "hf-fyaronskiy-deberta-goemotions" in names
    assert "hf-maxkazak-rubert-base-goemotions" in names
    assert create_model("hf-seara-rubert-tiny2-goemotions").model_id.startswith("seara/")


def test_aggregate_goemotions_logits_maps_labels_into_project_groups() -> None:
    labels = ["fear", "anger", "grief", "love", "joy", "neutral"]
    label_logits = np.array([[2.0, 3.0, 4.0, 5.0, 6.0, 1.0]])

    grouped = aggregate_goemotions_logits(label_logits, labels)
    probabilities = logits_to_probabilities(
        validate_logits(grouped, expected_rows=1)
    )

    assert grouped.shape == (1, len(EMOTIONS))
    assert probabilities.sum() == pytest.approx(1.0)
    assert grouped[0, EMOTIONS.index("joy")] == pytest.approx(6.0)
    assert grouped[0, EMOTIONS.index("warmth")] == pytest.approx(5.0)
    assert grouped[0, EMOTIONS.index("sadness")] == pytest.approx(4.0)
    assert grouped[0, EMOTIONS.index("anger")] == pytest.approx(3.0)
    assert grouped[0, EMOTIONS.index("anxiety")] == pytest.approx(2.0)
    assert grouped[0, EMOTIONS.index("neutral")] == pytest.approx(1.0)


def test_aggregate_goemotions_logits_uses_logsumexp_for_multiple_group_labels() -> None:
    labels = ["anger", "annoyance", "neutral"]
    grouped = aggregate_goemotions_logits([[1.0, 1.0, 0.0]], labels)

    anger = grouped[0, EMOTIONS.index("anger")]

    assert anger == pytest.approx(1.0 + np.log(2.0))


def test_aggregate_goemotions_logits_rejects_label_count_mismatch() -> None:
    with pytest.raises(ValueError, match="Expected 2 labels"):
        aggregate_goemotions_logits([[0.0, 1.0, 2.0]], ["joy", "neutral"])


def test_missing_goemotions_labels_reports_partial_models() -> None:
    missing = missing_goemotions_labels(["joy", "neutral"])

    assert missing["warmth"] == ["caring", "gratitude", "love"]
    assert "joy" in missing


def test_full_goemotions_labels_cover_project_groups() -> None:
    missing = missing_goemotions_labels(GOEMOTIONS_LABELS)

    assert missing == {}


def test_empty_logits_round_trip() -> None:
    grouped = aggregate_goemotions_logits(np.zeros((0, len(GOEMOTIONS_LABELS))), GOEMOTIONS_LABELS)

    assert validate_logits(grouped, expected_rows=0).shape == (0, len(EMOTIONS))
    assert logits_to_probabilities(grouped).shape == (0, len(EMOTIONS))


def test_validate_logits_still_rejects_bad_group_shape() -> None:
    with pytest.raises(ModelOutputError, match="shape"):
        validate_logits([[0.0, 1.0]], expected_rows=1)
