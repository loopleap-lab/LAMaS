"""The frozen sentence encoder shared by the controller and the learned stop gate.

The paper uses ``all-MiniLM-L6-v2`` (384 dimensions) for both the query and the operator
descriptions. It runs comfortably on a CPU. Loaded once per process, on first use.
"""

from __future__ import annotations

import os
import threading
from typing import Optional, List, Sequence

MODEL_NAME = os.environ.get("LAMAS_ENCODER", "sentence-transformers/all-MiniLM-L6-v2")
DIM = 384
MAX_CHARS = 1000

_lock = threading.Lock()
_model = None


def _load():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(MODEL_NAME, device=os.environ.get("LAMAS_DEVICE", "cpu"))
    return _model


def embed(texts: Sequence[str]):
    """Embed a batch of texts. Returns a ``(n, 384)`` float tensor on the CPU."""
    import torch

    vectors = _load().encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
    return torch.from_numpy(vectors).float()


def embed_one(text: str, max_chars: Optional[int] = MAX_CHARS):
    """Embed one text. Returns a ``(384,)`` float tensor.

    The stop gate was trained on the first ``MAX_CHARS`` characters of the query; the controller
    on the whole query (``max_chars=None``; the encoder itself cuts at 256 tokens).
    """
    return embed([text if max_chars is None else text[:max_chars]])[0]


def warm_up() -> None:
    """Load the model now, so the first query does not pay for it."""
    embed_one("warm up")
