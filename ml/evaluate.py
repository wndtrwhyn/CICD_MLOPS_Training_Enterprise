# ml/evaluate.py

import json
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .config import config


def evaluate():
    # Load trained model
    model = joblib.load(config.model_path)

    # Load data (di sini pakai Iris dataset sebagai contoh)
    data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=data.target,
    )

    # Predict
    y_pred = model.predict(X_test)

    # Hitung akurasi
    acc = accuracy_score(y_test, y_pred)

    # Simpan metrics ke file JSON
    metrics = {
        "accuracy": acc,
        "n_estimators": config.n_estimators,
    }

    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config.metrics_path, "w") as f:
        json.dump(metrics, f)

    print(f"Re-evaluated metrics: {metrics}")

    # Simple guardrail: kalau akurasi terlalu rendah → raise error
    if acc < 0.9:
        raise ValueError(f"Accuracy too low: {acc:.3f}")


if __name__ == "__main__":
    evaluate()
