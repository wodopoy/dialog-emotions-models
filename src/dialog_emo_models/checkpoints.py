"""Unified loader for the trained checkpoints under ``artifacts/models/``.

The training scripts here save a whole roster of models in three on-disk forms:

* ``*.joblib`` pickles — baselines, lexicons, sklearn TF-IDF models, fastText;
* a HuggingFace export directory (``config.json`` + ``model.safetensors``) — the
  GoEmotions presets (seara / fyaronskiy / maxkazak), whose 28- or 9-label heads
  are aggregated into the project emotions by :class:`GoEmotionsHFModel`; and
* a native save directory (``metadata.json``) — the fine-tuned RuBERT-tiny2,
  restored via :meth:`RuBertTiny2EmotionModel.load`.

``load_checkpoint`` is the single entry point that loads *any* of them by clean
name (or path), dispatching on the on-disk format and returning a ready
:class:`~dialog_emo_models.models.EmotionModel`. The checkpoints are 7-class
artifacts, so the package must be running under ``EMO_SCHEME=7`` — this is checked
explicitly with a precise error rather than failing deep inside scoring.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dialog_emo_models.models import CalibratedEmotionModel, EmotionModel
from dialog_emo_models.schema import EMOTIONS

# ``{stem}{CHECKPOINT_SUFFIX}.joblib`` registers as ``{stem}``, e.g.
# ``logreg-union-7class-rugo-cedr.joblib`` -> ``logreg-union``.
CHECKPOINT_SUFFIX = "-7class-rugo-cedr"

# Sidecar in the models dir: ``{clean name: calibrated temperature}``. Written by
# ``scripts/bake_temperatures.py`` and applied automatically on load, so scoring a
# dialogue uses the fitted T without anyone passing it by hand. Reversible: delete
# the file (or pass ``apply_temperature=False``) to get raw, uncalibrated logits.
TEMPERATURES_FILE = "temperatures.json"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be located or loaded."""


def _default_models_dir() -> Path:
    env = os.environ.get("DIALOG_EMO_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    # src/dialog_emo_models/checkpoints.py -> repo root is three parents up.
    return Path(__file__).resolve().parents[2] / "artifacts" / "models"


def _check_scheme_for(path: Path) -> None:
    # The artifacts are named '*-7class-*'. If one is loaded while the package is
    # running under a different scheme, fail early with a precise message instead
    # of deep inside scoring with a bare shape mismatch. Only fires for 7-class
    # artifacts, so plain --model-path of a non-7-class joblib stays unaffected.
    if "7class" in path.name and (len(EMOTIONS) != 7 or "surprise" not in EMOTIONS):
        raise CheckpointError(
            f"{path.name} is a 7-class checkpoint but the package is running with "
            f"EMOTIONS={EMOTIONS!r}. Set EMO_SCHEME=7 before importing dialog_emo_models."
        )


def _clean_name(raw: str) -> str:
    return raw[: -len(CHECKPOINT_SUFFIX)] if raw.endswith(CHECKPOINT_SUFFIX) else raw


def _looks_like_model_dir(path: Path) -> bool:
    return (path / "metadata.json").exists() or (path / "config.json").exists()


def available_checkpoints(models_dir: str | Path | None = None) -> dict[str, Path]:
    """Map clean checkpoint name -> path for every checkpoint in ``models_dir``.

    Cheap: only lists the directory. Returns ``{}`` when the directory is absent.
    """
    base = Path(models_dir).expanduser() if models_dir else _default_models_dir()
    if not base.is_dir():
        return {}

    found: dict[str, Path] = {}
    for entry in sorted(base.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_file() and entry.suffix == ".joblib":
            found[_clean_name(entry.stem)] = entry
        elif entry.is_dir() and _looks_like_model_dir(entry):
            found[_clean_name(entry.name)] = entry
    return found


def _is_pathlike(name_or_path: str | Path) -> bool:
    # Treat the argument as a direct filesystem path only when it actually looks
    # like one — otherwise a bare name like ``prior`` could be shadowed by a
    # same-named file/dir in the current working directory.
    text = str(name_or_path)
    given = Path(name_or_path)
    return os.sep in text or (os.altsep or "\x00") in text or given.is_absolute() or bool(given.suffix)


def _resolve_path(name_or_path: str | Path, models_dir: str | Path | None) -> Path:
    given = Path(name_or_path)
    if _is_pathlike(name_or_path) and given.exists():
        return given

    base = Path(models_dir).expanduser() if models_dir else _default_models_dir()
    name = str(name_or_path)
    candidates = [
        base / name,
        base / f"{name}.joblib",
        base / f"{name}{CHECKPOINT_SUFFIX}.joblib",
        base / f"{name}{CHECKPOINT_SUFFIX}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    known = ", ".join(sorted(available_checkpoints(base))) or "(none found)"
    raise CheckpointError(f"Unknown checkpoint {name!r} in {base}. Available: {known}")


def _temperature_for(path: Path) -> float:
    """Calibrated temperature for the checkpoint at ``path`` (1.0 if none recorded)."""
    sidecar = path.parent / TEMPERATURES_FILE
    if not sidecar.exists():
        return 1.0
    try:
        table = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 1.0
    return float(table.get(_clean_name(path.stem), 1.0))


def _load_dir(path: Path) -> EmotionModel:
    """Load a directory checkpoint, dispatching on its on-disk format."""
    if (path / "metadata.json").exists():
        from dialog_emo_models.models import RuBertTiny2EmotionModel

        return RuBertTiny2EmotionModel.load(path)

    if (path / "config.json").exists():
        from dialog_emo_models.models import GoEmotionsHFModel

        return GoEmotionsHFModel(str(path))

    raise CheckpointError(
        f"Directory checkpoint {path} has neither metadata.json (native save) "
        "nor config.json (HuggingFace export); cannot determine how to load it."
    )


def load_checkpoint(
    name_or_path: str | Path,
    *,
    models_dir: str | Path | None = None,
    apply_temperature: bool = True,
) -> EmotionModel:
    """Load any saved checkpoint by clean name (or path) into an ``EmotionModel``.

    With ``apply_temperature`` (the default) the calibrated temperature recorded in
    the sidecar ``temperatures.json`` is applied automatically — the returned model
    scores with its fitted T. Pass ``apply_temperature=False`` to get the raw model
    (used by the calibration scripts that re-fit T themselves).
    """
    path = _resolve_path(name_or_path, models_dir)
    _check_scheme_for(path)
    if path.is_dir():
        model = _load_dir(path)
    else:
        import joblib

        model = joblib.load(path)

    if not isinstance(model, EmotionModel):
        raise CheckpointError(
            f"{path} did not load an EmotionModel (got {type(model).__name__})"
        )

    if not apply_temperature:
        # Unwrap if the artifact itself is calibrated, so callers get raw logits.
        return model.model if isinstance(model, CalibratedEmotionModel) else model

    temperature = _temperature_for(path)
    if temperature != 1.0 and not isinstance(model, CalibratedEmotionModel):
        return CalibratedEmotionModel(model, temperature)
    return model


__all__ = [
    "CHECKPOINT_SUFFIX",
    "CheckpointError",
    "TEMPERATURES_FILE",
    "available_checkpoints",
    "load_checkpoint",
]
