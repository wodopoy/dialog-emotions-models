from __future__ import annotations

import json

import pytest

from dialog_emo_models.checkpoints import (
    CHECKPOINT_SUFFIX,
    CheckpointError,
    _is_pathlike,
    _resolve_path,
    available_checkpoints,
)


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
