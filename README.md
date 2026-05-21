# dialog-emo-models

Training and scoring contracts for dialogue emotion models.

This repository is intentionally model-focused: no UI, no Gradio app, no
plotting layer. The stable boundary is:

```text
Telegram result.json -> parsed CSV -> EmotionModel -> full CSV
```

The current v1 ships the data contracts, Telegram parser, dummy model, CLI, and
tests. Real training code can be added behind the same model interface later.

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

## Data Contracts

Parsed CSV, used as model input:

```text
turn_index,timestamp,sender,text
```

Full CSV, used as scored/training data:

```text
turn_index,timestamp,sender,text,joy,warmth,sadness,anger,anxiety,neutral
```

Rules:

- `turn_index` must be dense `0..N-1`.
- Emotion columns are probabilities in `[0, 1]`.
- The six emotion probabilities must sum to `1 +/- 1e-3` per row.
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
keeps future models interchangeable:

- sklearn/CatBoost wrappers can vectorize `texts` internally and return
  `(n_texts, 6)` scores.
- HF GoEmotions wrappers map provider labels into the same six emotions.
- Supervised training can read full CSV emotion columns as the label matrix.

Register a model lazily:

```python
from dialog_emo_models.registry import register_model

register_model("my-model", lambda: MyModel())
```

The `dummy` model returns zero logits and therefore uniform `1/6`
probabilities.

## Trainable Baselines

Two sklearn baselines are available through `dialog-emo train`:

| training name | model | target handling | when to use |
| --- | --- | --- | --- |
| `ridge-tfidf` | char n-gram TF-IDF + `Ridge` | learns log-probabilities from the six soft labels | first default for full CSV labels |
| `logreg-tfidf` | char n-gram TF-IDF + `LogisticRegression` | trains on the strongest emotion per row | quick hard-label comparison |

`ridge-tfidf` is the recommended first training baseline. It preserves the idea
that labels are distributions, trains fast on small data, saves as `.joblib`,
and still satisfies the same `EmotionModel` contract at inference time.

CatBoost is a good next experiment, but it is not in the base dependency set
yet. The intended shape is the same: train from full CSV, save under
`artifacts/`, and expose `predict_logits(texts)` so the normal scorer can use
it.

## HF GoEmotions Presets

HF models are optional so the base environment stays small. Install the runtime
deps only when you want to run them:

```bash
uv add transformers torch
```

Registered presets:

| registry name | HF model | notes |
| --- | --- | --- |
| `hf-seara-rubert-tiny2-goemotions` | `seara/rubert-tiny2-russian-emotion-detection-ru-go-emotions` | lightest first try |
| `hf-fyaronskiy-deberta-goemotions` | `fyaronskiy/deberta-v1-base-russian-go-emotions` | stronger, still medium-sized |
| `hf-maxkazak-rubert-base-goemotions` | `MaxKazak/ruBert-base-russian-emotion-detection` | RuBERT-base comparison point |

Run any preset through the normal CLI:

```bash
uv run dialog-emo run \
  --input data/result.json \
  --parsed-output output/parsed.csv \
  --output output/scored.csv \
  --model hf-seara-rubert-tiny2-goemotions
```

The wrappers aggregate GoEmotions labels into the project emotions:

```python
GOEMOTIONS_GROUPS = {
    "anxiety": ["fear", "nervousness", "embarrassment"],
    "anger": ["anger", "annoyance", "disapproval", "disgust"],
    "sadness": ["disappointment", "grief", "remorse", "sadness"],
    "warmth": ["caring", "gratitude", "love"],
    "joy": [
        "admiration", "amusement", "approval", "excitement",
        "joy", "optimism", "pride",
    ],
    "neutral": ["neutral"],
}
```

Aggregation uses `logsumexp` over available label logits for each group, then
the pipeline softmaxes the final six group logits. If a model exposes only a
partial label set, missing labels are ignored; a completely missing group gets
a very low fallback logit.
