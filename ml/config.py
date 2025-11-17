# ml/config.py

from dataclasses import dataclass
from pathlib import Path

# BASE_DIR akan menunjuk ke root project kamu (folder utama)
BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class TrainingConfig:
    random_state: int = 42
    test_size: float = 0.2
    n_estimators: int = 100  # contoh untuk skenario tuning: 50 → 100
    
    model_dir: Path = BASE_DIR / "models"
    model_path: Path = BASE_DIR / "models" / "model.pkl"
    metrics_path: Path = BASE_DIR / "models" / "metrics.json"

# instantiate config
config = TrainingConfig()
