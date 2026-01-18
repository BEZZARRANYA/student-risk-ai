from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_metrics(name: str):
    path = REPORTS_DIR / f"metrics_{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["metrics"]


def md_row(label: str, m: dict) -> str:
    return (
        f"| {label} | {m['roc_auc']:.3f} | {m['f1']:.3f} | {m['accuracy']:.3f} |"
    )


def main() -> None:
    logreg = load_metrics("logreg_early")
    rf = load_metrics("rf_early")

    table = []
    table.append("| Model (Early-warning) | ROC-AUC | F1 | Accuracy |")
    table.append("|---|---:|---:|---:|")
    table.append(md_row("Logistic Regression", logreg))
    table.append(md_row("Random Forest", rf))

    out_path = REPORTS_DIR / "model_comparison.md"
    out_path.write_text("\n".join(table) + "\n", encoding="utf-8")

    print("Saved:", out_path)
    print("\n".join(table))


if __name__ == "__main__":
    main()

