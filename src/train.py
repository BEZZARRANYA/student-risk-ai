from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data import DatasetConfig, add_labels, load_raw, split_feature_sets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    model_name: str = "logreg_early"
    use_late_features: bool = False  # early-warning by default


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def train_and_eval(X: pd.DataFrame, y: pd.Series, cfg: TrainConfig) -> Tuple[Pipeline, Dict[str, float]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    preprocessor = build_preprocessor(X)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", clf),
        ]
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "risk_rate_test": float(np.mean(y_test)),
    }

    return model, metrics


def main() -> None:
    cfg_data = DatasetConfig()
    df = load_raw(cfg_data)
    df = add_labels(df, cfg_data)
    X_early, X_late, y = split_feature_sets(df)

    cfg = TrainConfig()
    X = X_late if cfg.use_late_features else X_early

    model, metrics = train_and_eval(X, y, cfg)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{cfg.model_name}.joblib"
    metrics_path = REPORTS_DIR / f"metrics_{cfg.model_name}.json"

    joblib.dump(model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "metrics": metrics}, f, indent=2)

    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()

