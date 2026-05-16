import os
from pathlib import Path

_MODELS_DIR_ENV = "CLINICIAN_MODELS_DIR"


def models_dir() -> str:
    return os.environ.get(_MODELS_DIR_ENV, "").strip()


def resolve_model(model_id_or_path: str) -> str:
    if not model_id_or_path:
        return model_id_or_path
    if os.path.isabs(model_id_or_path) or model_id_or_path.startswith((".", "~")):
        return os.path.expanduser(model_id_or_path)
    base = models_dir()
    if not base:
        return model_id_or_path
    candidate = Path(base) / model_id_or_path.split("/")[-1]
    if candidate.exists():
        return str(candidate)
    return model_id_or_path
