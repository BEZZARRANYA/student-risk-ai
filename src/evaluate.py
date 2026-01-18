from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import DatasetConfig, add_labels, load_raw, split_feature_sets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"


def load_model(name: str = "logreg_early"):
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing model at {path}. Run src/train.py first.")
    return joblib.load(path)


def get_test_split() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    cfg = DatasetConfig()
    df = add_labels(load_raw(cfg), cfg)
    X_early, _, y = split_feature_sets(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_early, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test, y_train


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Evaluate both models
    for model_name in ["logreg_early", "rf_early"]:
        model = load_model(model_name)
        X_test, y_test, _ = get_test_split()

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        # 1) Confusion matrix
        cm_path = FIG_DIR / f"confusion_matrix_{model_name}.png"
        ConfusionMatrixDisplay.from_predictions(y_test, pred)
        plt.title(f"Confusion Matrix — {model_name}")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=200)
        plt.close()

        # 2) ROC curve
        roc_path = FIG_DIR / f"roc_{model_name}.png"
        RocCurveDisplay.from_predictions(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.title(f"ROC Curve — {model_name} (AUC={auc:.3f})")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=200)
        plt.close()

        # 3) Calibration curve
        cal_path = FIG_DIR / f"calibration_{model_name}.png"
        prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
        plt.plot(prob_pred, prob_true, marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"Calibration Curve — {model_name}")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.tight_layout()
        plt.savefig(cal_path, dpi=200)
        plt.close()

        # Save evaluation summary
        summary_path = REPORTS_DIR / f"evaluation_{model_name}.json"
        summary = {
            "model_name": model_name,
            "auc": float(auc),
            "confusion_matrix_fig": str(cm_path.relative_to(PROJECT_ROOT)),
            "roc_fig": str(roc_path.relative_to(PROJECT_ROOT)),
            "calibration_fig": str(cal_path.relative_to(PROJECT_ROOT)),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[{model_name}] Saved figures + summary.")


if __name__ == "__main__":
    main()

