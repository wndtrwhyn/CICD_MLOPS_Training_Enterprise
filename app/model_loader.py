# app/model_loader.py

import os
from functools import lru_cache
from pathlib import Path

import joblib
from google.cloud import storage

# Path model lokal (default: models/model.pkl)
LOCAL_MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))

# Contoh: gs://bucket-name/path/to/model.pkl
GCS_MODEL_URI = os.getenv("GCS_MODEL_URI", "")


def _download_from_gcs(gcs_uri: str, local_path: Path) -> None:
    """Download model dari Google Cloud Storage ke path lokal."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError("GCS_MODEL_URI must start with gs://")

    # Parse bucket & blob name dari URI
    _, path_part = gcs_uri.split("gs://", 1)
    bucket_name, blob_name = path_part.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Pastikan folder lokal exist
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Download file
    blob.download_to_filename(str(local_path))


@lru_cache(maxsize=1)
def load_model():
    """
    Prioritas loading model:

    1. Kalau GCS_MODEL_URI di-set → download dari GCS ke LOCAL_MODEL_PATH.
    2. Kalau tidak → langsung load dari LOCAL_MODEL_PATH (sudah dibawa di Docker image).
    """
    # Kalau env GCS_MODEL_URI ada → download dulu
    if GCS_MODEL_URI:
        print(f"[model_loader] Downloading model from {GCS_MODEL_URI}")
        _download_from_gcs(GCS_MODEL_URI, LOCAL_MODEL_PATH)

    # Pastikan file model exist
    if not LOCAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {LOCAL_MODEL_PATH}")

    print(f"[model_loader] Loading model from {LOCAL_MODEL_PATH}")
    return joblib.load(LOCAL_MODEL_PATH)
