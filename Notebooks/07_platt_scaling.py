"""
platt_scaling.py
ECG Trustworthiness Framework — Nazish Atta, George Washington University
Step 7: Platt scaling recalibration — fit sigmoid calibrators on validation set,
        compare Brier scores and calibration curves before and after.

Outputs:
  results/calibration/brier_scores_before_after.csv
  results/calibration/calibration_summary.txt
  figures/drafts/calibration_curves_before.png
  figures/drafts/calibration_curves_after_platt.png
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
# Note: manual Platt scaling used (sklearn 1.8 removed cv="prefit" from CalibratedClassifierCV)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
DATA_DIR    = os.path.join(OUTPUT_DIR, "data")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "calibration")
FIGURES_DIR = os.path.join(BASE_DIR, "figures", "drafts")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 60)
print("STEP 7 — PLATT SCALING RECALIBRATION")
print("=" * 60)

# ── Load validation set (used to fit calibrators) ─────────────
val  = pd.read_csv(os.path.join(DATA_DIR, "val_processed.csv"))
X_val  = val.drop("arrhythmia", axis=1)
y_val  = val["arrhythmia"].values
print(f"Validation set: {X_val.shape} | Arrhythmia: {y_val.sum()} ({y_val.mean()*100:.1f}%)")

# ── Load test set (used to evaluate before/after) ─────────────
test = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))
X_test = test.drop("arrhythmia", axis=1)
y_test = test["arrhythmia"].values
print(f"Test set:       {X_test.shape} | Arrhythmia: {y_test.sum()} ({y_test.mean()*100:.1f}%)")

# ── Load base models ───────────────────────────────────────────
model_files = {
    "Logistic Regression": "logistic_regression",
    "Random Forest":       "random_forest",
    "XGBoost":             "xgboost",
}
models = {name: joblib.load(os.path.join(MODELS_DIR, f"{fname}.pkl"))
          for name, fname in model_files.items()}

COLORS = {"Logistic Regression": "#4472C4", "Random Forest": "#ED7D31", "XGBoost": "#70AD47"}

# ── Fit Platt scaling and collect results ──────────────────────
print("\nFitting Platt scaling calibrators on validation set...")

results      = []
probs_before = {}
probs_after  = {}

for name, model in models.items():
    # Raw probabilities on val set (used to fit the sigmoid)
    val_prob_raw  = model.predict_proba(X_val)[:, 1].reshape(-1, 1)
    # Raw probabilities on test set before calibration
    prob_before   = model.predict_proba(X_test)[:, 1]
    brier_before  = brier_score_loss(y_test, prob_before)

    # Fit Platt sigmoid: logistic regression on val-set raw probs vs true labels
    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(val_prob_raw, y_val)

    # Apply sigmoid to test set raw probabilities
    prob_after  = platt.predict_proba(prob_before.reshape(-1, 1))[:, 1]
    brier_after = brier_score_loss(y_test, prob_after)

    probs_before[name] = prob_before
    probs_after[name]  = prob_after

    improvement = brier_before - brier_after
    print(f"\n  {name}:")
    print(f"    Brier before: {brier_before:.4f}")
    print(f"    Brier after:  {brier_after:.4f}  (Delta = {improvement:+.4f})")

    results.append({
        "model":          name,
        "brier_before":   round(brier_before, 4),
        "brier_after":    round(brier_after,  4),
        "brier_delta":    round(improvement,  4),
        "improved":       improvement > 0,
    })

    # Save Platt scaler object
    cal_path = os.path.join(MODELS_DIR, f"{model_files[name]}_platt.pkl")
    joblib.dump(platt, cal_path)
    print(f"    Platt scaler saved: {cal_path}")

# ── Save Brier score comparison CSV ───────────────────────────
results_df = pd.DataFrame(results)
csv_path   = os.path.join(RESULTS_DIR, "brier_scores_before_after.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# ── Write calibration summary text ────────────────────────────
summary_path = os.path.join(RESULTS_DIR, "calibration_summary.txt")
with open(summary_path, "w") as f:
    f.write("CALIBRATION SUMMARY — Platt Scaling Results\n")
    f.write("=" * 50 + "\n\n")
    for row in results:
        direction = "improved" if row["improved"] else "worsened"
        f.write(f"{row['model']}\n")
        f.write(f"  Brier before: {row['brier_before']:.4f}\n")
        f.write(f"  Brier after:  {row['brier_after']:.4f}\n")
        f.write(f"  Delta:        {row['brier_delta']:+.4f} ({direction})\n\n")
    f.write("Note: Lower Brier score = better calibrated.\n")
    f.write("Calibrated models saved to outputs/models/*_platt.pkl\n")
print(f"Saved: {summary_path}")

# ── Plot: Calibration curves BEFORE Platt scaling ─────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
for name in models:
    frac_pos, mean_pred = calibration_curve(y_test, probs_before[name], n_bins=10)
    bs = brier_score_loss(y_test, probs_before[name])
    ax.plot(mean_pred, frac_pos, "o-", color=COLORS[name], lw=2,
            label=f"{name} (Brier={bs:.3f})")
ax.set_xlabel("Mean Predicted Probability", fontsize=12)
ax.set_ylabel("Fraction of Positives (Actual)", fontsize=12)
ax.set_title("Calibration Curves — Before Platt Scaling\n(closer to diagonal = better calibrated)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "calibration_curves_before.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {fig_path}")

# ── Plot: Calibration curves AFTER Platt scaling ──────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
for name in models:
    frac_pos, mean_pred = calibration_curve(y_test, probs_after[name], n_bins=10)
    bs = brier_score_loss(y_test, probs_after[name])
    ax.plot(mean_pred, frac_pos, "o-", color=COLORS[name], lw=2,
            label=f"{name} (Brier={bs:.3f})")
ax.set_xlabel("Mean Predicted Probability", fontsize=12)
ax.set_ylabel("Fraction of Positives (Actual)", fontsize=12)
ax.set_title("Calibration Curves — After Platt Scaling\n(closer to diagonal = better calibrated)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "calibration_curves_after_platt.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {fig_path}")

print("\n[DONE] Step 7 complete.")
