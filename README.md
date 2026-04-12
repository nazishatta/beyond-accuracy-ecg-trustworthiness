# Beyond Accuracy: Toward Trustworthy and Interpretable AI for Heart Rhythm Detection from ECG Signals

**Nazish Atta** | Department of Computer Science | George Washington University | Washington, D.C.  
*Presented at the 2026 CCAS Research Showcase, George Washington University*

---

## Overview

This repository contains the complete implementation of a **trustworthiness-oriented evaluation framework** for machine learning models applied to ECG arrhythmia detection. Rather than optimizing for accuracy alone, this study evaluates models across four clinical trustworthiness dimensions: **sensitivity**, **precision-recall performance**, **probability calibration**, and **SHAP-based interpretability**.

The central finding: the most accurate model (XGBoost, 88.6%) missed **186 arrhythmia patients**, while the less accurate Logistic Regression (82.1%) missed only **155** — demonstrating that accuracy alone is insufficient for safe clinical AI deployment.

---

## Results Summary

| Model | Accuracy | Recall | Specificity | AUC-ROC | Brier Score | False Negatives |
|-------|----------|--------|-------------|---------|-------------|-----------------|
| Logistic Regression | 82.06% | **72.81%** ★ | 85.34% | 0.8603 | 0.1352 | **155** ★ |
| Random Forest | 88.12% | 66.32% | 95.84% | 0.8845 | 0.1079 | 192 |
| XGBoost | **88.62%** ★ | 67.37% | **96.15%** ★ | **0.8867** ★ | **0.0960** ★ | 186 |

> ★ = best value for that metric. For False Negatives and Brier Score, lower is better.

---

## Key Findings

- **Accuracy paradox**: XGBoost is the most accurate model (88.62%) but misses 186 arrhythmia cases. Logistic Regression, with 6.5 percentage points lower accuracy, misses 37 fewer patients.
- **Error character differs**: Tree-based model misses (RF, XGBoost) are "confident misses" — the model assigns very low arrhythmia probability to these cases, making them harder to recover through threshold adjustment. LR's misses cluster near the decision boundary and are more recoverable.
- **Calibration**: XGBoost achieves the lowest Brier score (0.0960), indicating the best overall probability calibration. Logistic Regression has the highest Brier score (0.1352) despite the common assumption that linear models are inherently well-calibrated.
- **SHAP interpretability**: The top predictive features — RR interval, QRS count, ventricular rate, atrial rate, and P-wave morphology — all correspond to established clinical ECG diagnostic criteria, supporting clinical trust in the model's behavior.

---

## Dataset

- **Source**: [PTB-XL Plus](https://physionet.org/content/ptb-xl-plus/1.0.1/) (version 1.0.1) — Strodthoff et al., *Scientific Data*, 2023
- **Size**: 21,799 clinical 12-lead ECG recordings (21,732 used after excluding 67 with missing 12SL features)
- **Features**: 782 structured ECG features extracted by the 12SL algorithm
- **Label**: Binary — Arrhythmia (any non-normal SCP-ECG rhythm code) vs. Normal
- **Split**: Stratified 10-fold — Folds 1–8 train (17,374), Fold 9 validation (2,178), Fold 10 test (2,180)
- **Class imbalance**: 26.1% arrhythmia prevalence in test set; SMOTE applied to training set only

---

## Methodology

### Models
| Model | Description |
|-------|-------------|
| Logistic Regression | L2-regularized linear classifier; transparent probabilistic baseline |
| Random Forest | 100-tree bootstrap ensemble; captures non-linear feature interactions |
| XGBoost | Gradient-boosted tree ensemble; state-of-the-art tabular performance |

All models use default hyperparameters, fixed random seed (42), and decision threshold 0.50.

### Evaluation Framework (4 Dimensions)
1. **Discriminative performance** — Accuracy, Precision, Recall, F1, Specificity, MCC, AUC-ROC, Average Precision
2. **False-negative analysis** — FN rate, probability distributions of detected vs. missed arrhythmia cases
3. **Probability calibration** — Reliability diagrams (calibration curves) and Brier score
4. **SHAP interpretability** — TreeExplainer on Random Forest; global feature ranking via mean |SHAP| over 500-sample test subset

---

## Repository Structure

```
beyond-accuracy-ecg-trustworthiness/
│
├── Notebooks/                        # Python pipeline scripts + Jupyter notebooks
│   ├── 01_load_data.py               # Data loading, label construction, EDA
│   ├── 02_preprocess.py              # Imputation, scaling, SMOTE
│   ├── 03_train_models.py            # LR, RF, XGBoost training
│   ├── 04_evaluate.py                # ROC, PR, calibration, FN analysis plots
│   ├── 05_shap_analysis.py           # SHAP TreeExplainer, beeswarm + bar plots
│   ├── 06_threshold_sensitivity.py   # Threshold sweep analysis
│   ├── 07_platt_scaling.py           # Post-hoc probability recalibration
│   ├── 08_subtype_fn_analysis.py     # Arrhythmia subtype FN breakdown
│   ├── 09_rebuild_paper1.py          # Full Paper 1 rebuild from saved models
│   ├── 01–09_*.ipynb                 # Jupyter notebooks for each analysis step
│   └── 10_paper1_full_analysis.ipynb # Complete Paper 1 analysis notebook
│
├── Plots/                            # Final figures (poster and paper ready)
│   ├── plot1_roc_curves.png          # ROC curves — LR, RF, XGBoost
│   ├── plot2_pr_curves.png           # Precision-recall curves
│   ├── plot3_confusion_matrices.png  # Confusion matrices (all 3 models)
│   ├── plot4_calibration_curves.png  # Reliability diagrams + Brier scores
│   ├── plot5_false_negative_analysis.png  # Missed vs. detected probability distributions
│   ├── plot6_shap_beeswarm_rf.png    # SHAP beeswarm (500-sample test subset)
│   └── plot7_shap_bar_top15_rf.png   # Top 15 features by mean |SHAP|
│
├── Results/                          # Computed metrics and findings
│   ├── metrics_comparison.csv        # Core classification metrics
│   ├── metrics_full.csv              # Extended metrics (Brier, MCC, Bal. Acc.)
│   ├── model_comparison.csv          # Side-by-side model comparison
│   ├── shap_feature_importance_rf.csv # Full SHAP feature ranking (782 features)
│   └── poster_findings_verified.txt  # Verified findings summary
│
├── Tables/                           # Publication-ready tables
│   ├── table1_classification_performance.csv / .tex
│   ├── table2_full_metrics.csv / .tex
│   ├── table3_confusion_counts.csv / .tex
│   └── table4_shap_top10_rf.csv / .tex
│
└── src/                              # Original source scripts
    ├── evaluate.py
    ├── load_data.py
    ├── preprocess.py
    ├── shap_analysis.py
    └── train_models.py
```

---

## How to Run

### Requirements
```bash
pip install numpy pandas scikit-learn xgboost shap imbalanced-learn matplotlib seaborn joblib
```

### Pipeline (run in order)
```bash
python Notebooks/01_load_data.py
python Notebooks/02_preprocess.py
python Notebooks/03_train_models.py
python Notebooks/04_evaluate.py
python Notebooks/05_shap_analysis.py
```

All outputs (models, plots, results) are saved to `outputs/` automatically.

### Environment
- Python 3.11 | NumPy 1.26 | pandas 2.1 | scikit-learn 1.8 | XGBoost 2.0 | SHAP 0.44
- Fixed random seed: 42

---

## Citation

If you use this code or findings, please cite:

```
Atta, N. (2026). Beyond Accuracy: Toward Trustworthy and Interpretable AI for
Heart Rhythm Detection from ECG Signals. Presented at the 2026 CCAS Research
Showcase, George Washington University, Washington, D.C.
```

**Dataset citation:**
```
Strodthoff, N. et al. (2023). PTB-XL+, a comprehensive electrocardiographic
feature dataset. Scientific Data, 10(1), 279.
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
