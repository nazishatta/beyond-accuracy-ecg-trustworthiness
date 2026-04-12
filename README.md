# Beyond Accuracy: Toward Trustworthy and Interpretable AI for Heart Rhythm Detection from ECG Signals

**Nazish Atta** · Department of Data Science · George Washington University · Washington, D.C.  
*2026 CCAS Research Showcase, George Washington University*

---

## Scanned a QR code? Start here.

This project asks a simple clinical question: **which AI model is safest for detecting dangerous heart rhythms?**

The answer is not the most accurate one.

> XGBoost achieved the highest accuracy (88.6%) — but missed **186** arrhythmia patients.  
> Logistic Regression was 6.5 points less accurate — but missed only **155**.  
> **The most accurate model was not the safest model for clinical screening.**

This repository contains the full code, results, and figures behind that finding.

---

## Key Findings

| Finding | Detail |
|---------|--------|
| **Accuracy ≠ Safety** | XGBoost (88.62% accurate) misses 31 more arrhythmia cases than Logistic Regression (82.06% accurate) |
| **Error character matters** | Tree-model misses are confident "Normal" predictions — harder to recover via threshold adjustment. LR's misses cluster near the decision boundary and are more amenable to correction. |
| **Best calibration: XGBoost** | Lowest Brier score (0.0960). Logistic Regression — despite the common assumption — has the worst calibration (Brier = 0.1352) on this dataset. |
| **Clinically coherent features** | SHAP analysis shows the top predictors are RR interval, QRS count, ventricular rate, atrial rate, and P-wave morphology — all established clinical ECG markers. |

---

## Results Summary

Evaluated on the PTB-XL Plus test set — **2,180 ECGs, 570 true arrhythmia cases** (26.1% prevalence).

| Model | Accuracy | Recall | Specificity | AUC-ROC | Brier Score ↓ | False Negatives ↓ |
|-------|:--------:|:------:|:-----------:|:-------:|:-------------:|:-----------------:|
| Logistic Regression | 82.06% | **72.81%** ★ | 85.34% | 0.8603 | 0.1352 | **155** ★ |
| Random Forest | 88.12% | 66.32% | 95.84% | 0.8845 | 0.1079 | 192 |
| XGBoost | **88.62%** ★ | 67.37% | **96.15%** ★ | **0.8867** ★ | **0.0960** ★ | 186 |

> ★ = best on that metric. ↓ = lower is better (Brier Score, False Negatives).

**Extended metrics** (F1, MCC, Average Precision, Balanced Accuracy) are in [`Results/metrics_full.csv`](Results/metrics_full.csv).

---

## Figures and Tables

All figures are in [`Plots/`](Plots/) and all tables are in [`Tables/`](Tables/).

| File | Description |
|------|-------------|
| [`plot1_roc_curves.png`](Plots/plot1_roc_curves.png) | ROC curves for all three models — AUC-ROC: LR=0.8603, RF=0.8845, XGB=0.8867 |
| [`plot2_pr_curves.png`](Plots/plot2_pr_curves.png) | Precision-recall curves — Average Precision: LR=0.7574, RF=0.8060, XGB=0.8151 |
| [`plot3_confusion_matrices.png`](Plots/plot3_confusion_matrices.png) | Confusion matrices for all three models |
| [`plot4_calibration_curves.png`](Plots/plot4_calibration_curves.png) | Reliability diagrams — shows deviation from perfect calibration for all models |
| [`plot5_false_negative_analysis.png`](Plots/plot5_false_negative_analysis.png) | Predicted probability distributions for detected vs. missed arrhythmia cases |
| [`plot6_shap_beeswarm_rf.png`](Plots/plot6_shap_beeswarm_rf.png) | SHAP beeswarm — per-patient feature contributions (500-sample test subset, RF) |
| [`plot7_shap_bar_top15_rf.png`](Plots/plot7_shap_bar_top15_rf.png) | Top 15 ECG features by mean absolute SHAP value |

| File | Description |
|------|-------------|
| [`table1_classification_performance`](Tables/table1_classification_performance.csv) | Core metrics: Accuracy, Precision, Recall, Specificity, F1, AUC |
| [`table2_full_metrics`](Tables/table2_full_metrics.csv) | Extended metrics: Brier Score, MCC, Average Precision, Balanced Accuracy |
| [`table3_confusion_counts`](Tables/table3_confusion_counts.csv) | TP, FP, FN, TN, FN rate, FP rate for all three models |
| [`table4_shap_top10_rf`](Tables/table4_shap_top10_rf.csv) | Top 10 ECG features by mean \|SHAP\| with clinical interpretation |

Each table is available in both `.csv` (for inspection) and `.tex` (for direct LaTeX inclusion).

---

## Methods Summary

### Dataset

- **PTB-XL Plus** (v1.0.1) — [Strodthoff et al., *Scientific Data*, 2023](https://physionet.org/content/ptb-xl-plus/1.0.1/)
- 21,799 clinical 12-lead ECGs; **21,732 used** (67 excluded due to missing 12SL feature records)
- **782 structured features** per recording, extracted by the 12SL algorithm
- **Binary label**: Arrhythmia (any non-normal SCP-ECG rhythm code) vs. Normal
- **Split**: Folds 1–8 → train (17,374), Fold 9 → validation (2,178), Fold 10 → test (2,180)
- **Class imbalance**: SMOTE applied to training set only; test set retains natural 26.1% prevalence

### Models

| Model | Role |
|-------|------|
| Logistic Regression (LR) | Transparent probabilistic baseline; L2 regularization |
| Random Forest (RF) | 100-tree bootstrap ensemble; non-linear feature interactions |
| XGBoost | Gradient-boosted tree ensemble; strongest aggregate performance |

All models: default hyperparameters, random seed = 42, decision threshold = 0.50.

### Evaluation Framework — Four Dimensions

1. **Discriminative performance** — Accuracy, Precision, Recall, F1, Specificity, MCC, AUC-ROC, Average Precision
2. **False-negative analysis** — FN rate; predicted probability distributions for missed vs. detected cases
3. **Probability calibration** — Reliability diagrams and Brier score
4. **SHAP interpretability** — TreeExplainer on Random Forest; global feature importance via mean |SHAP| on a 500-sample test subset

---

## Repository Structure

```
beyond-accuracy-ecg-trustworthiness/
│
├── Notebooks/                              # Full analysis pipeline
│   │
│   ├── — Python scripts (run in order) —
│   ├── 01_load_data.py                     # Data loading, label construction, EDA
│   ├── 02_preprocess.py                    # Median imputation, StandardScaler, SMOTE
│   ├── 03_train_models.py                  # Train LR, RF, XGBoost; save .pkl files
│   ├── 04_evaluate.py                      # ROC, PR, calibration, FN analysis plots
│   ├── 05_shap_analysis.py                 # SHAP TreeExplainer; beeswarm + bar chart
│   ├── 06_threshold_sensitivity.py         # Threshold sweep across operating points
│   ├── 07_platt_scaling.py                 # Post-hoc Platt scaling recalibration
│   ├── 08_subtype_fn_analysis.py           # FN breakdown by arrhythmia subtype
│   └── 09_rebuild_paper1.py               # Full reproducibility rebuild from saved models
│   │
│   └── — Jupyter notebooks (interactive exploration) —
│       ├── 01_baseline_model_comparison.ipynb
│       ├── 02_false_negative_and_precision_recall_analysis.ipynb
│       ├── 03_calibration_analysis.ipynb
│       ├── 04_threshold_sensitivity_analysis.ipynb
│       ├── 05_platt_scaling_recalibration.ipynb
│       ├── 06_arrhythmia_subtype_false_negative_breakdown.ipynb
│       ├── 07_shap_global_interpretability.ipynb
│       ├── 08_shap_waterfall_case_studies.ipynb
│       ├── 09_final_tables_and_figures.ipynb
│       └── 10_paper1_full_analysis.ipynb
│
├── Plots/                                  # 7 final figures (300 dpi, poster-ready)
│   ├── plot1_roc_curves.png
│   ├── plot2_pr_curves.png
│   ├── plot3_confusion_matrices.png
│   ├── plot4_calibration_curves.png
│   ├── plot5_false_negative_analysis.png
│   ├── plot6_shap_beeswarm_rf.png
│   └── plot7_shap_bar_top15_rf.png
│
├── Results/                                # Computed metrics and verified findings
│   ├── metrics_comparison.csv             # Core classification metrics
│   ├── metrics_full.csv                   # Extended metrics (Brier, MCC, Bal. Acc.)
│   ├── model_comparison.csv               # Side-by-side model comparison
│   ├── shap_feature_importance_rf.csv     # Full SHAP ranking for all 782 features
│   └── poster_findings_verified.txt       # Human-readable verified findings summary
│
├── Tables/                                 # Publication-ready tables (.csv + .tex)
│   ├── table1_classification_performance.{csv,tex}
│   ├── table2_full_metrics.{csv,tex}
│   ├── table3_confusion_counts.{csv,tex}
│   └── table4_shap_top10_rf.{csv,tex}
│
└── src/                                    # Source scripts (original versions)
    ├── evaluate.py
    ├── load_data.py
    ├── preprocess.py
    ├── shap_analysis.py
    └── train_models.py
```

---

## Reproducibility

### Requirements

```bash
pip install numpy==1.26 pandas==2.1 scikit-learn==1.8 xgboost==2.0 \
            shap==0.44 imbalanced-learn matplotlib seaborn joblib
```

### Run the full pipeline

```bash
python Notebooks/01_load_data.py
python Notebooks/02_preprocess.py
python Notebooks/03_train_models.py
python Notebooks/04_evaluate.py
python Notebooks/05_shap_analysis.py
```

All outputs — trained model `.pkl` files, plots, and result CSVs — are written to `outputs/` automatically.

To regenerate all Paper 1 figures and tables from the saved models without retraining:

```bash
python Notebooks/09_rebuild_paper1.py
```

**Environment**: Python 3.11 · random seed 42 · Windows 11 / Linux compatible

> **Note on data**: PTB-XL Plus must be downloaded separately from [PhysioNet](https://physionet.org/content/ptb-xl-plus/1.0.1/) and placed in the project root before running `01_load_data.py`. The dataset is not included in this repository.

---

## Poster and Paper Status

| Item | Status |
|------|--------|
| Research poster | Presented — 2026 CCAS Research Showcase, GWU |
| Manuscript | In preparation |
| Code | Complete and reproducible |
| Data | Publicly available via PhysioNet (not redistributed here) |

---

## Citation

```bibtex
@misc{atta2026beyond,
  author    = {Nazish Atta},
  title     = {Beyond Accuracy: Toward Trustworthy and Interpretable AI
               for Heart Rhythm Detection from {ECG} Signals},
  year      = {2026},
  note      = {Presented at the 2026 CCAS Research Showcase,
               George Washington University, Washington, D.C.}
}
```

**Dataset:**

```bibtex
@article{strodthoff2023ptbxlplus,
  author  = {Strodthoff, Nils and others},
  title   = {{PTB-XL+}, a comprehensive electrocardiographic feature dataset},
  journal = {Scientific Data},
  volume  = {10},
  pages   = {279},
  year    = {2023}
}
```

---

## Contact

**Nazish Atta**  
Department of Data Science, George Washington University  
Washington, D.C., USA  
[nazishatta@gwu.edu](mailto:nazishatta@gwu.edu) · [GitHub](https://github.com/nazishatta)

---

## License

This code is released under the [MIT License](LICENSE).  
The PTB-XL Plus dataset is subject to its own license — see [PhysioNet](https://physionet.org/content/ptb-xl-plus/1.0.1/) for terms.

---

<!--
INTERNAL NOTE — not rendered on GitHub:

Filename inconsistencies to clean up in a future commit:
- Notebooks/ uses conflicting number prefixes: .py scripts (01–09) and .ipynb
  notebooks (01–10) share the same prefix numbers, creating visual ambiguity
  in directory listings. Recommend renaming .ipynb files to a separate series
  (e.g., NB01–NB10) or moving scripts and notebooks into separate subdirectories.
- README_github.md at the repo root is a 3-line stub that duplicates this README.
  Safe to delete.
- Results/ contains untracked subdirectories (baseline_metrics/, calibration/,
  shap/, subtype_fn/, threshold_analysis/) not yet pushed to GitHub.
  Consider either committing them or adding them to .gitignore.
- Tables/ contains untracked drafts/ and final/ subdirectories.
-->
