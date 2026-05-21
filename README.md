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
- HF zero-shot wrappers can map provider labels into the same six emotions.
- Supervised training can read full CSV emotion columns as the label matrix.

Register a model lazily:

```python
from dialog_emo_models.registry import register_model

register_model("my-model", lambda: MyModel())
```

The v1 registry contains only `dummy`, which returns zero logits and therefore
uniform `1/6` probabilities.
