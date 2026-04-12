"""
threshold_sensitivity.py
ECG Trustworthiness Framework — Nazish Atta, George Washington University
Step 6: Threshold sensitivity analysis — sweep classification threshold for all three models.

Outputs:
  results/threshold_analysis/threshold_sweep_lr.csv
  results/threshold_analysis/threshold_sweep_rf.csv
  results/threshold_analysis/threshold_sweep_xgb.csv
  figures/drafts/threshold_sweep_lr.png
  figures/drafts/threshold_sweep_rf.png
  figures/drafts/threshold_sweep_xgb.png
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs")
DATA_DIR     = os.path.join(OUTPUT_DIR, "data")
MODELS_DIR   = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "results", "threshold_analysis")
FIGURES_DIR  = os.path.join(BASE_DIR, "figures", "drafts")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 60)
print("STEP 6 — THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 60)

# ── Load test data ─────────────────────────────────────────────
test = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))
X_test = test.drop("arrhythmia", axis=1)
y_test = test["arrhythmia"].values
print(f"Test set: {X_test.shape} | Arrhythmia: {y_test.sum()} ({y_test.mean()*100:.1f}%)")

# ── Load models ────────────────────────────────────────────────
model_files = {
    "Logistic Regression": "logistic_regression",
    "Random Forest":       "random_forest",
    "XGBoost":             "xgboost",
}
models = {name: joblib.load(os.path.join(MODELS_DIR, f"{fname}.pkl"))
          for name, fname in model_files.items()}

COLORS      = {"Logistic Regression": "#4472C4", "Random Forest": "#ED7D31", "XGBoost": "#70AD47"}
FILE_LABELS = {"Logistic Regression": "lr", "Random Forest": "rf", "XGBoost": "xgb"}
THRESHOLDS  = np.round(np.arange(0.05, 0.96, 0.05), 2)

# ── Sweep thresholds for each model ───────────────────────────
print(f"\nSweeping {len(THRESHOLDS)} thresholds: {THRESHOLDS[0]:.2f} to {THRESHOLDS[-1]:.2f}")

for model_name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    label  = FILE_LABELS[model_name]
    color  = COLORS[model_name]

    rows = []
    for thresh in THRESHOLDS:
        y_pred = (y_prob >= thresh).astype(int)
        cm     = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Guard against zero-division at extreme thresholds
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        rows.append({
            "threshold": thresh,
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "tp":        int(tp),
            "fp":        int(fp),
            "fn":        int(fn),
            "tn":        int(tn),
            "fn_rate":   round(fn / max(y_test.sum(), 1), 4),
        })

    df_sweep = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, f"threshold_sweep_{label}.csv")
    df_sweep.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # ── Plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Threshold Sensitivity — {model_name}", fontsize=14, fontweight="bold")

    # Left: Precision, Recall, F1 vs threshold
    ax = axes[0]
    ax.plot(df_sweep["threshold"], df_sweep["precision"], "o-", color="#4472C4",
            lw=2, label="Precision")
    ax.plot(df_sweep["threshold"], df_sweep["recall"],    "s-", color="#ED7D31",
            lw=2, label="Recall (Sensitivity)")
    ax.plot(df_sweep["threshold"], df_sweep["f1"],        "^-", color="#70AD47",
            lw=2, label="F1 Score")
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.6, label="Default threshold (0.5)")

    # Mark optimal F1 threshold
    best_idx   = df_sweep["f1"].idxmax()
    best_thresh = df_sweep.loc[best_idx, "threshold"]
    best_f1     = df_sweep.loc[best_idx, "f1"]
    ax.axvline(best_thresh, color="#70AD47", linestyle=":", lw=1.5,
               label=f"Best F1 threshold ({best_thresh:.2f})")
    ax.scatter([best_thresh], [best_f1], color="#70AD47", s=100, zorder=5)

    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1 vs Threshold", fontweight="bold")
    ax.set_xlim([0.03, 0.97]); ax.set_ylim([-0.02, 1.05])
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Right: FN count vs threshold
    ax2 = axes[1]
    ax2.plot(df_sweep["threshold"], df_sweep["fn"], "o-", color="#A32D2D", lw=2,
             label="False Negatives (missed arrhythmias)")
    ax2.axvline(0.5, color="black", linestyle="--", alpha=0.6, label="Default threshold (0.5)")
    ax2.axvline(best_thresh, color="#70AD47", linestyle=":", lw=1.5,
                label=f"Best F1 threshold ({best_thresh:.2f})")
    ax2.set_xlabel("Classification Threshold", fontsize=12)
    ax2.set_ylabel("Number of False Negatives", fontsize=12)
    ax2.set_title("Missed Arrhythmias (FN) vs Threshold", fontweight="bold")
    ax2.set_xlim([0.03, 0.97])
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"threshold_sweep_{label}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")
    print(f"  Optimal F1 threshold: {best_thresh:.2f} "
          f"(F1={best_f1:.4f}, FN={df_sweep.loc[best_idx, 'fn']})")

print("\n[DONE] Step 6 complete.")
