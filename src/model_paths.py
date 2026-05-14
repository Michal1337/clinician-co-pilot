"""Resolve a HuggingFace model id to a local path when ``CLINICIAN_MODELS_DIR``
is set.

Layout assumed under that directory:

    $CLINICIAN_MODELS_DIR/
        gemma-4-26B-A4B-it/
        medsiglip-448/
        medasr/
        Bio_ClinicalBERT/

(i.e. one folder per model, named after the model's basename — the part
after the ``org/`` slash). If the resolved folder doesn't exist, falls
back to the original HF id so transformers will hit the hub / cache as
before. This keeps the codebase usable both with a curated local mirror
and on a fresh box.
"""

import os
from pathlib import Path

_MODELS_DIR_ENV = "CLINICIAN_MODELS_DIR"


def models_dir() -> str:
    return os.environ.get(_MODELS_DIR_ENV, "").strip()


def resolve_model(model_id_or_path: str) -> str:
    """Map a HuggingFace id like ``google/gemma-4-26B-A4B-it`` to
    ``$CLINICIAN_MODELS_DIR/gemma-4-26B-A4B-it`` if that folder exists.
    Otherwise return the input unchanged."""
    if not model_id_or_path:
        return model_id_or_path
    # If the caller already passed an absolute path or a clearly-local
    # relative path, just trust it.
    if os.path.isabs(model_id_or_path) or model_id_or_path.startswith((".", "~")):
        return os.path.expanduser(model_id_or_path)

    base = models_dir()
    if not base:
        return model_id_or_path

    candidate = Path(base) / model_id_or_path.split("/")[-1]
    if candidate.exists():
        return str(candidate)
    return model_id_or_path
