from __future__ import annotations

import json

import joblib
import numpy as np
import pytest

from dialog_emo_models.checkpoints import (
    CHECKPOINT_SUFFIX,
    TEMPERATURES_FILE,
    CheckpointError,
    _is_pathlike,
    _resolve_path,
    available_checkpoints,
    load_checkpoint,
)
from dialog_emo_models.models import CalibratedEmotionModel, DummyEmotionModel


def _make_models_dir(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / f"logreg-union{CHECKPOINT_SUFFIX}.joblib").touch()
    (models / f"prior{CHECKPOINT_SUFFIX}.joblib").touch()

    hf = models / "hf-seara-rubert-tiny2"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps({"id2label": {}}), encoding="utf-8")

    rubert = models / f"rubert-tiny2-finetune{CHECKPOINT_SUFFIX}"
    rubert.mkdir()
    (rubert / "metadata.json").write_text(json.dumps({"kind": "rubert"}), encoding="utf-8")

    (models / ".DS_Store").touch()
    (models / "notes.txt").write_text("ignore", encoding="utf-8")
    (models / "scratch").mkdir()  # no config/metadata -> not a model dir
    return models


def test_available_checkpoints_discovers_and_cleans(tmp_path):
    models = _make_models_dir(tmp_path)
    found = available_checkpoints(models)
    assert set(found) == {"logreg-union", "prior", "hf-seara-rubert-tiny2", "rubert-tiny2-finetune"}
    assert found["logreg-union"].suffix == ".joblib"
    assert found["hf-seara-rubert-tiny2"].is_dir()


def test_available_checkpoints_missing_dir(tmp_path):
    assert available_checkpoints(tmp_path / "nope") == {}


def test_is_pathlike():
    assert not _is_pathlike("prior")
    assert not _is_pathlike("logreg-union")
    assert _is_pathlike("prior.joblib")
    assert _is_pathlike("sub/prior")
    assert _is_pathlike("/abs/prior")


def test_resolve_bare_name_uses_models_dir_not_cwd(tmp_path, monkeypatch):
    models = _make_models_dir(tmp_path)
    # A file named exactly like a checkpoint sits in the CWD; it must NOT shadow
    # the real checkpoint resolved from models_dir.
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "prior").write_text("decoy", encoding="utf-8")
    monkeypatch.chdir(cwd)

    resolved = _resolve_path("prior", models)
    assert resolved == models / f"prior{CHECKPOINT_SUFFIX}.joblib"


def test_resolve_unknown_name_raises(tmp_path):
    models = _make_models_dir(tmp_path)
    with pytest.raises(CheckpointError, match="Unknown checkpoint"):
        _resolve_path("does-not-exist", models)


def _toy_models_dir(tmp_path, temperature):
    joblib.dump(DummyEmotionModel(), tmp_path / "toy.joblib")
    (tmp_path / TEMPERATURES_FILE).write_text(json.dumps({"toy": temperature}), encoding="utf-8")
    return tmp_path


def test_calibrated_wrapper_scales_logits():
    inner = DummyEmotionModel()
    texts = ["привет", "пока"]
    assert np.allclose(
        CalibratedEmotionModel(inner, 2.0).predict_logits(texts), inner.predict_logits(texts) / 2.0
    )
    assert np.allclose(
        CalibratedEmotionModel(inner, 1.0).predict_logits(texts), inner.predict_logits(texts)
    )


def test_load_checkpoint_applies_baked_temperature(tmp_path):
    models = _toy_models_dir(tmp_path, temperature=2.5)
    calibrated = load_checkpoint("toy", models_dir=models)
    assert isinstance(calibrated, CalibratedEmotionModel) and calibrated.temperature == 2.5

    raw = load_checkpoint("toy", models_dir=models, apply_temperature=False)
    assert isinstance(raw, DummyEmotionModel)

    texts = ["раз", "два", "три"]
    assert np.allclose(calibrated.predict_logits(texts), raw.predict_logits(texts) / 2.5)


def test_load_checkpoint_skips_wrap_when_temperature_is_one(tmp_path):
    models = _toy_models_dir(tmp_path, temperature=1.0)
    assert isinstance(load_checkpoint("toy", models_dir=models), DummyEmotionModel)


def test_load_checkpoint_without_sidecar_is_raw(tmp_path):
    joblib.dump(DummyEmotionModel(), tmp_path / "toy.joblib")
    assert isinstance(load_checkpoint("toy", models_dir=tmp_path), DummyEmotionModel)
