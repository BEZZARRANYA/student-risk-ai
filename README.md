# Explainable Student Risk Prediction System

An explainable machine learning pipeline to predict **academic risk** (at-risk vs not at-risk) using student demographic + behavioral features.
The project focuses on **transparent decision support** for education: not only predicting risk, but also explaining *why*.

## Dataset
- UCI Student Performance dataset (Math course: `student-mat.csv`)
- Grades range from 0–20.

## Target definition (Risk Label)
We define:

- **at_risk = 1 if final grade (G3) < 10, else 0**

This creates a practical binary risk target aligned with common pass/fail thresholds.

## Leakage-aware feature sets (research-oriented)
We support two experimental settings:

1. **Early-warning (primary setting)**  
   Uses only features available before final exams (excludes `G1`, `G2`).  
   This is the most realistic setting for early intervention.

2. **Late-term (secondary / optional)**  
   Includes `G1`, `G2` (earlier period grades), which improves accuracy but can partially “shortcut” prediction.
   We keep this setting to analyze tradeoffs and report transparently.

## Baseline model
- Logistic Regression with:
  - categorical one-hot encoding
  - missing-value imputation
  - `class_weight="balanced"` for class imbalance

Artifacts are saved for reproducibility:
- Model: `models/logreg_early.joblib`
- Metrics: `reports/metrics_logreg_early.json`

## Evaluation (beyond accuracy)
We generate research-style evaluation plots:

- Confusion Matrix: `reports/figures/confusion_matrix_logreg_early.png`
- ROC Curve: `reports/figures/roc_logreg_early.png`
- Calibration Curve: `reports/figures/calibration_logreg_early.png`

Calibration is included to check whether predicted probabilities are reliable for decision support.

## Model comparison (Early-warning setting)

The dataset is relatively small (395 students), so simpler models can sometimes generalize as well as or better than more complex ones.

**Results:**
| Model (Early-warning) | ROC-AUC | F1 | Accuracy |
|---|---:|---:|---:|
| Logistic Regression | ... | ... | ... |
| Random Forest | ... | ... | ... |
**Plots (both models):**
- Logistic Regression: `reports/figures/roc_logreg_early.png`, `reports/figures/calibration_logreg_early.png`
- Random Forest: `reports/figures/roc_rf_early.png`, `reports/figures/calibration_rf_early.png`

## Explainability
We provide:

### Global explanation
Top features by coefficient magnitude:
- CSV: `reports/global_top_features_logreg_early.csv`
- Plot: `reports/figures/global_top_features_logreg_early.png`

### Local explanation (example student)
Feature-level contributions for a single high-risk prediction:
- `reports/local_explanation_example_logreg_early.csv`

## Ethical considerations
This project is intended for **decision support**, not automated punishment.
Predictions should be used to allocate supportive resources (tutoring, outreach), and explanations should be reviewed by educators.
Models may reflect socioeconomic or demographic correlations, so subgroup evaluation is recommended before real deployment.

## How to run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.data
python -m src.train
python -m src.evaluate
python -m src.explain

