# dialog-emo-models

Training, evaluation, and scoring of emotion models for short Russian dialogue
messages — the model side of a fully local ("on-device") emotional-timeline
pipeline. This is the experimental codebase of the bachelor's thesis
«Исследование и разработка методов локального определения эмоциональной окраски
текстовых сообщений в мобильных системах» (МФТИ, 2026). The visualisation side
lives in the sibling
[dialog-emotions-demo](https://github.com/wodopoy/dialog-emotions-demo).

The stable boundary between the two:

```text
Telegram result.json -> parsed CSV -> EmotionModel -> full CSV (emotion columns)
```

The demo app only renders the full CSV; nothing here draws plots, nothing there
loads a model.

## Results at a glance

Seven emotion classes — `joy, warmth, sadness, anger, anxiety, surprise,
neutral` (`surprise` added so the native CEDR corpus maps in fully). About
twenty models — from hand lexicons and TF-IDF linear models to fastText, naive
Bayes, tree ensembles, and fine-tuned/pretrained transformers — are compared on
one protocol: trained on RuGoEmotions (a machine-translated GoEmotions),
evaluated both in-domain and on the native Russian corpus CEDR. The primary
selection metric is the KL divergence between the predicted and target class
distributions (labels are soft); accuracy, macro-F1, ECE, JS, and deploy
metrics (artifact size, CPU latency P50/P95) are reported alongside.

Final comparison on the held-out CEDR test set (training on RuGoEmotions+CEDR,
hyperparameter search for the light families):

| model | acc | macro-F1 | KL | ECE | size, MB | P95, ms |
| --- | --- | --- | --- | --- | --- | --- |
| rubert-tiny2-finetune (quality ceiling) | 0.775 | 0.595 | 0.638 | 0.026 | 111.4 | 1.54 |
| **logreg-char (deploy pick)** | **0.722** | 0.534 | 0.824 | 0.057 | **2.7** | **0.39** |
| logreg-union | 0.698 | 0.519 | 0.878 | 0.042 | 37.2 | 1.89 |
| fastText | 0.674 | 0.527 | 1.029 | 0.106 | 5.8 | 0.09 |
| nb-multinomial | 0.668 | 0.472 | 0.938 | 0.037 | 2.9 | 0.36 |
| lexicon-learned | 0.540 | 0.407 | 1.436 | 0.087 | 0.04 | 0.02 |

Key findings:

- **`logreg-char` is the final deploy choice**: char n-gram TF-IDF (`char_wb`
  2–6, 30k features, sublinear TF) + `LogisticRegression` (C=2.0, no class
  weighting). 2.7 MB and sub-millisecond CPU latency, within ~5 p.p. accuracy
  of a fine-tuned transformer 40× its size — a Pareto compromise, not a single-
  metric winner. The fine-tuned RuBERT-tiny2 stays the quality reference when
  resource limits are relaxed.
- **The native-domain gap is data, not capacity.** Linear models trained only
  on the translated corpus lose ~10 p.p. accuracy on native Russian (part of
  their in-domain edge is fitting translation artifacts). Adding ~7.5k native
  CEDR examples closes most of it — logreg-char goes 0.494 → 0.699 on CEDR at
  ~zero in-domain cost, overtaking a frozen DeBERTa 175× its size. Collecting
  a small native sample beats shipping a transformer.
- **Calibration is part of the contract.** The timeline averages predicted
  distributions, so ECE matters. Per-model temperature scaling is fitted on
  validation by ECE with a noise deadband: the chosen logreg-char is already
  calibrated (T = 1), while overconfident ridge / off-the-shelf HF models drop
  from ECE ≈ 0.26–0.40 to ≈ 0.03 for free (argmax is temperature-invariant, so
  accuracy/F1 are untouched). Fitted temperatures ship next to the checkpoints
  and are applied automatically on load.

Full experiment journals live in `docs/`:
[RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) (per-iteration leaderboards),
[PROJECT_HISTORY.md](docs/PROJECT_HISTORY.md) (narrative of what was tried and
why), [CALIBRATION.md](docs/CALIBRATION.md) and
[CALIBRATION_OBJECTIVE.md](docs/CALIBRATION_OBJECTIVE.md) (temperature
calibration and the ECE-vs-NLL objective study, with reliability diagrams in
`docs/img/`). The experiment runners are in `scripts/`.

## Emotion scheme

The thesis scheme and all shipped checkpoints are 7-class; select it with
`EMO_SCHEME=7` (env var, read at import time). Without it the package falls
back to the earlier 6-class scheme (no `surprise`), kept for the early
experiments. Loading a 7-class checkpoint under the wrong scheme fails early
with a precise error.

## Setup

```bash
uv venv
uv sync
```

Useful checks:

```bash
uv run pytest
uv run ruff check .
```

## CLI

`data/result.json` is a small synthetic Telegram-export fixture (the same
hand-authored dialogue as the demo's bundled sample); the examples below and
the tests use it. No real conversations are stored in this repository.

Parse Telegram Desktop JSON into the model input format:

```bash
uv run dialog-emo parse \
  --input data/result.json \
  --output output/parsed.csv
```

Score an existing parsed CSV:

```bash
uv run dialog-emo score \
  --input output/parsed.csv \
  --output output/scored.csv \
  --model dummy
```

Run both steps:

```bash
uv run dialog-emo run \
  --input data/result.json \
  --parsed-output output/parsed.csv \
  --output output/scored.csv \
  --model dummy
```

`output/`, `runs/`, and `artifacts/` are local-only and ignored by git.

Train a lightweight sklearn baseline from a full CSV:

```bash
uv run dialog-emo train \
  --input data/train.csv \
  --output artifacts/ridge-tfidf.joblib \
  --model ridge-tfidf
```

Use a saved model for scoring:

```bash
uv run dialog-emo score \
  --input output/parsed.csv \
  --output output/scored.csv \
  --model-path artifacts/ridge-tfidf.joblib
```

### Trained checkpoints (unified loader)

Every model saved under `artifacts/models/` — whatever its on-disk form —
loads through one interface (`dialog_emo_models.checkpoints.load_checkpoint`),
which dispatches on the format: `*.joblib` pickles, HuggingFace export
directories (GoEmotions presets, aggregated into the project emotions), and the
native `RuBERT-tiny2` fine-tune directory. Calibrated temperatures recorded in
the sidecar `temperatures.json` (written by `scripts/bake_temperatures.py`) are
applied automatically on load; pass `apply_temperature=False` for raw logits.

```bash
EMO_SCHEME=7 uv run dialog-emo list          # all checkpoints by clean name + kind

# Score with any of them by name (joblib OR directory, transparently):
EMO_SCHEME=7 uv run dialog-emo run \
  --input data/result.json \
  --output output/scored.csv \
  --checkpoint logreg-char                    # or rubert-tiny2-finetune, logreg-union, ...
```

`--checkpoint NAME` resolves against `artifacts/models/` (override with
`--models-dir` or `$DIALOG_EMO_MODELS_DIR`); `--model-path PATH` takes an
explicit file or directory. The checkpoints are 7-class, so set `EMO_SCHEME=7`.
The output CSV is exactly the column contract the `dialog-emotions-demo` UI
reads.

```python
from dialog_emo_models.checkpoints import available_checkpoints, load_checkpoint

available_checkpoints()                       # {name: path}
model = load_checkpoint("logreg-char")        # ready EmotionModel, calibrated T applied
model.predict_proba(["я так рад тебя видеть"])
```

## Data Contracts

Parsed CSV, used as model input:

```text
turn_index,timestamp,sender,text
```

Full CSV, used as scored/training data (7-class scheme):

```text
turn_index,timestamp,sender,text,joy,warmth,sadness,anger,anxiety,surprise,neutral
```

(Under the legacy 6-class scheme the `surprise` column is absent.)

Rules:

- `turn_index` must be dense `0..N-1`.
- Emotion columns are probabilities in `[0, 1]`.
- The emotion probabilities must sum to `1 +/- 1e-3` per row.
- `timestamp`, `sender`, and `text` are preserved as metadata; models receive
  only `text`.

## Model Interface

Every model subclasses `EmotionModel` and implements batched logits:

```python
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from dialog_emo_models.models import EmotionModel
from dialog_emo_models.schema import EMOTIONS


class MyModel(EmotionModel):
    def predict_logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        return np.zeros((len(texts), len(EMOTIONS)), dtype=float)
```

The pipeline softmaxes logits and writes probabilities to the full CSV. This
keeps models interchangeable: sklearn/fastText wrappers vectorize `texts`
internally, HF GoEmotions wrappers map provider labels into the project
emotions, and supervised training reads full-CSV emotion columns as the label
matrix. Before training and inference every model sees the same
`normalize_text` pass (case/whitespace/link normalisation, emoji mapped to
emotion-word markers).

Register a model lazily:

```python
from dialog_emo_models.registry import register_model

register_model("my-model", lambda: MyModel())
```

The `dummy` model returns zero logits and therefore uniform probabilities.

## Trainable Baselines

These models train through `dialog-emo train` and share the same
`fit(texts, labels)` / `predict_logits(texts)` contract, so they can all be
trained on one full CSV and compared on one validation set:

| training name | model | target handling |
| --- | --- | --- |
| `ridge-tfidf` | char n-gram TF-IDF + `Ridge` | learns log-probabilities from the soft labels |
| `logreg-tfidf` | char n-gram TF-IDF + `LogisticRegression` | trains on the strongest emotion per row |
| `ridge-word-char-tfidf` | word + char TF-IDF union + `Ridge` | soft labels |
| `logreg-word-char-tfidf` | word + char TF-IDF union + `LogisticRegression` | strongest emotion per row |
| `fasttext-supervised` | native fastText with subword features | strongest emotion per row |
| `rubert-tiny2-finetune` | RuBERT-tiny2 fine-tuned classifier | strongest emotion per row |

On the benchmark the `logreg` variants dominate `ridge` on the metrics that
matter for the timeline (KL and calibration: ridge is overconfident, ECE ≈
0.24–0.26 vs ≈ 0.03 for tuned logreg), which is why the thesis deploy pick is
the char-n-gram logistic regression. Ridge remains a useful soft-label
baseline; fastText is the fastest candidate but weaker as a distribution.

## Lexicon Baseline

`lexicon` is a keyword-count dictionary baseline (no training). It is registered
for inference and runs through the normal CLI:

```bash
uv run dialog-emo score --input output/parsed.csv --output output/scored.csv --model lexicon
```

## HF GoEmotions Presets

Off-the-shelf Russian emotion classifiers used as reference points:

| registry name | HF model | notes |
| --- | --- | --- |
| `hf-seara-rubert-tiny2-goemotions` | `seara/rubert-tiny2-russian-emotion-detection-ru-go-emotions` | lightest first try |
| `hf-fyaronskiy-deberta-goemotions` | `fyaronskiy/deberta-v1-base-russian-go-emotions` | strongest off-the-shelf, 475 MB |
| `hf-maxkazak-rubert-base-goemotions` | `MaxKazak/ruBert-base-russian-emotion-detection` | trained on a different label scheme; excluded from the fair KL comparison |

Run any preset through the normal CLI:

```bash
EMO_SCHEME=7 uv run dialog-emo run \
  --input data/result.json \
  --parsed-output output/parsed.csv \
  --output output/scored.csv \
  --model hf-seara-rubert-tiny2-goemotions
```

The wrappers aggregate GoEmotions labels into the project emotions (7-class
mapping shown; the rare `desire` label has no clean target and is dropped):

```python
GOEMOTIONS_GROUPS = {
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
```

Aggregation uses `logsumexp` over available label logits for each group, then
the pipeline softmaxes the final group logits. If a model exposes only a
partial label set, missing labels are ignored; a completely missing group gets
a very low fallback logit.
