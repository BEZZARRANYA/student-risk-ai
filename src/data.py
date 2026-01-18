from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class DatasetConfig:
    csv_name: str = "student-mat.csv"  # default: Math dataset
    sep: str = ";"
    risk_threshold: int = 10  # at_risk = 1 if G3 < 10


def load_raw(cfg: DatasetConfig) -> pd.DataFrame:
    """Load raw UCI student data."""
    path = RAW_DIR / cfg.csv_name
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}. Did you unzip into data/raw/?")
    df = pd.read_csv(path, sep=cfg.sep)
    return df


def add_labels(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """Create target columns."""
    out = df.copy()

    # Ensure grade columns are numeric (they may be read as strings)
    for col in ["G1", "G2", "G3"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["at_risk"] = (out["G3"] < cfg.risk_threshold).astype(int)
    return out


def split_feature_sets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Return:
      X_early: features excluding G1/G2 (early warning)
      X_late:  features including G1/G2 (late-term)
      y:       at_risk label
    """
    if "at_risk" not in df.columns:
        raise ValueError("Expected column 'at_risk'. Call add_labels() first.")

    y = df["at_risk"]

    # Drop the true outcome grade (G3) from all feature sets (target leakage)
    base_drop = ["G3", "at_risk"]

    # Late-term set includes G1/G2
    X_late = df.drop(columns=base_drop)

    # Early-warning set excludes G1/G2 as well (more realistic for intervention)
    X_early = df.drop(columns=base_drop + ["G1", "G2"])

    return X_early, X_late, y


def save_processed(df: pd.DataFrame, name: str = "students_labeled.csv") -> Path:
    """Save labeled data to data/processed/ for reproducibility."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / name
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    cfg = DatasetConfig()
    df = load_raw(cfg)
    df = add_labels(df, cfg)

    X_early, X_late, y = split_feature_sets(df)
    out_path = save_processed(df)

    print("Loaded:", cfg.csv_name)
    print("Rows, cols:", df.shape)
    print("Saved processed:", out_path)
    print("Risk rate (at_risk=1):", float(y.mean()))
    print("Early features:", X_early.shape[1], "| Late features:", X_late.shape[1])


if __name__ == "__main__":
    main()

