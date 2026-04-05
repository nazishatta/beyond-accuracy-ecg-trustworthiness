# Beyond Accuracy: Toward Trustworthy and Interpretable AI for Heart Rhythm Detection from ECG Signals

**Nazish Atta** | Department of Data Science | George Washington University | Washington, D.C.  
*Presented at the 2026 CCAS Research Showcase, George Washington University*

---

## Overview

This repository contains the complete implementation of a **trustworthiness-oriented evaluation framework** for machine learning models applied to ECG arrhythmia detection. Rather than optimising for accuracy alone, this study evaluates models across four clinical trustworthiness dimensions: **sensitivity**, **precision-recall performance**, **probability calibration**, and **SHAP-based interpretability**.

The central finding: the most accurate model (XGBoost, 88.6%) missed **186 arrhythmia patients**, while the less accurate Logistic Regression (82.1%) missed only **155** — demonstrating that accuracy alone is insufficient for safe clinical AI deployment.

---

## Results Summary

| Model | Accuracy | Recall | Specificity | AUC-ROC | Brier Score | False Negatives |
|-------|----------|--------|-------------|---------|-------------|-----------------|
| Logistic Regression | 82.06% | **72.81%** ★ | 85.34% | 0.8603 | 0.1352 | **155** ★ |
| Random Forest | 88.12% | 66.32% | 95.84% | 0.8845 | 0.1079 | 192 |
| XGBoost | **88.62%** ★ | 67.37% | **96.15%** ★ | **0.8867** ★ | **0.0960** ★ | 186 |
