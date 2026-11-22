import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

MODEL_FILES = {
    "Random Forest": "rf_behavioural.joblib",
    "Logistic Regression": "logreg_behavioural.joblib",
    "Linear SVM": "svm_behavioural.joblib",
}


def load_metadata():
    """Read the metrics and settings saved at training time."""
    path = MODELS_DIR / "model_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run notebooks/09_train_final.py first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_model(name):
    """Load one saved model by its display name."""
    if name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {name}")
    path = MODELS_DIR / MODEL_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run notebooks/09_train_final.py first.")
    return joblib.load(path)


def load_importance():
    """Read the saved feature-importance table."""
    path = MODELS_DIR / "feature_importance.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run notebooks/09_train_final.py first.")
    return pd.read_csv(path)