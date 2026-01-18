from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import DatasetConfig, add_labels, load_raw, split_feature_sets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"


def load_model(name: str = "logreg_early"):
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing model at {path}. Run training first.")
    return joblib.load(path)


def get_feature_names(pipeline) -> List[str]:
    """
    Extract expanded feature names after preprocessing (numeric + one-hot categorical).
    Works for ColumnTransformer + OneHotEncoder.
    """
    prep = pipeline.named_steps["prep"]
    num_cols = prep.transformers_[0][2]
    cat_pipe = prep.transformers_[1][1]
    cat_cols = prep.transformers_[1][2]
    ohe = cat_pipe.named_steps["onehot"]

    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    return list(num_cols) + cat_feature_names


def global_explanation(pipeline, top_n: int = 15) -> pd.DataFrame:
    clf = pipeline.named_steps["clf"]
    coef = clf.coef_.ravel()
    names = get_feature_names(pipeline)

    df = pd.DataFrame({"feature": names, "coef": coef})
    df["abs_coef"] = df["coef"].abs()
    df = df.sort_values("abs_coef", ascending=False).reset_index(drop=True)

    top = df.head(top_n).copy()
    return top


def plot_global(top_df: pd.DataFrame, out_path: Path) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.barh(top_df["feature"][::-1], top_df["coef"][::-1])
    plt.title("Global Explanation — Top Features (LogReg Early Warning)")
    plt.xlabel("Coefficient (positive increases risk)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def local_explanation(pipeline, X: pd.DataFrame, idx: int) -> pd.DataFrame:
    """
    Local explanation for one instance using linear contribution:
      contribution_j = x_j * coef_j  (after preprocessing)
    """
    clf = pipeline.named_steps["clf"]
    coef = clf.coef_.ravel()
    names = get_feature_names(pipeline)

    # Transform one row through preprocessing to get the numeric vector
    x_row = X.iloc[[idx]]
    x_vec = pipeline.named_steps["prep"].transform(x_row)
    x_vec = np.asarray(x_vec).ravel()

    contrib = x_vec * coef
    df = pd.DataFrame({"feature": names, "value": x_vec, "contribution": contrib})
    df["abs_contribution"] = df["contribution"].abs()
    df = df.sort_values("abs_contribution", ascending=False).reset_index(drop=True)
    return df


def main() -> None:
    model = load_model("logreg_early")

    cfg = DatasetConfig()
    df = add_labels(load_raw(cfg), cfg)
    X_early, _, y = split_feature_sets(df)

    # Pick one example student (highest predicted risk)
    proba = model.predict_proba(X_early)[:, 1]
    idx = int(np.argmax(proba))

    top_global = global_explanation(model, top_n=15)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    top_global.to_csv(REPORTS_DIR / "global_top_features_logreg_early.csv", index=False)

    plot_global(top_global, FIG_DIR / "global_top_features_logreg_early.png")

    local_df = local_explanation(model, X_early, idx=idx)
    local_df.head(15).to_csv(REPORTS_DIR / "local_explanation_example_logreg_early.csv", index=False)

    print("Saved global explanation CSV + plot.")
    print("Saved local explanation CSV for example idx:", idx)
    print("Example predicted risk probability:", float(proba[idx]))


if __name__ == "__main__":
    main()

