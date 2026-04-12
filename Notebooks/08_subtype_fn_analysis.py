"""
subtype_fn_analysis.py
ECG Trustworthiness Framework — Nazish Atta, George Washington University
Step 8: Arrhythmia subtype false-negative breakdown — for each model, identify
        which arrhythmia subtypes are most frequently missed.

Method:
  - test_processed.csv is saved without ecg_id (index=False in preprocess.py)
  - test_raw.csv retains ecg_id as index in the same row order
  - We use positional alignment: row i of test_processed = row i of test_raw
  - Original metadata (ptbxl_database.csv) provides scp_codes per ecg_id

Outputs:
  results/subtype_fn/subtype_false_negative_counts.csv
  results/subtype_fn/subtype_false_negative_rates.csv
  figures/drafts/subtype_false_negative_barplot.png
"""

import os
import ast
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from collections import defaultdict

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
DATA_DIR    = os.path.join(OUTPUT_DIR, "data")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "subtype_fn")
FIGURES_DIR = os.path.join(BASE_DIR, "figures", "drafts")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 60)
print("STEP 8 — ARRHYTHMIA SUBTYPE FALSE-NEGATIVE ANALYSIS")
print("=" * 60)

# ── Load test processed (features + arrhythmia label, no ecg_id) ──
test_proc = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))
X_test = test_proc.drop("arrhythmia", axis=1)
y_test = test_proc["arrhythmia"].values
feature_cols = X_test.columns.tolist()
print(f"Test set: {X_test.shape} | Arrhythmia cases: {y_test.sum()}")

# ── Load test raw (same rows, unscaled — used for positional matching) ──
test_raw = pd.read_csv(os.path.join(DATA_DIR, "test_raw.csv"))
print(f"Test raw: {test_raw.shape}")

# ── Load full feature set and statements to recover ecg_ids ───
# ptbxl_statements.csv rows are positionally aligned with 12sl_features.csv rows
feat_path  = os.path.join(DATASET_DIR, "features", "12sl_features.csv")
stmt_path  = os.path.join(DATASET_DIR, "labels", "ptbxl_statements.csv")
full_feats = pd.read_csv(feat_path)   # 21799 rows, no header index = ecg_id
statements = pd.read_csv(stmt_path)   # 21799 rows, has ecg_id column + scp_codes
print(f"Full features: {full_feats.shape} | Statements: {statements.shape}")

# Match each test_raw row back to a position in full_feats using first feature col
# (P_Area_I values are unique enough as a fingerprint combined with a second col)
match_col1 = feature_cols[0]   # P_Area_I
match_col2 = feature_cols[1]   # P_PeakTime_I

full_key = full_feats[match_col1].round(6).astype(str) + "_" + full_feats[match_col2].round(6).astype(str)
test_key  = test_raw[match_col1].round(6).astype(str)  + "_" + test_raw[match_col2].round(6).astype(str)

key_to_pos = {k: i for i, k in enumerate(full_key)}
test_positions = [key_to_pos.get(k, -1) for k in test_key]
matched = sum(1 for p in test_positions if p >= 0)
print(f"Matched {matched}/{len(test_positions)} test rows back to full feature set")

# Map position in full_feats → ecg_id from statements (positionally aligned)
ecg_ids = [statements.loc[p, "ecg_id"] if p >= 0 else -1 for p in test_positions]

# ── Load scp_codes from statements ────────────────────────────
stmt_by_ecgid = statements.set_index("ecg_id")

# ── Arrhythmia subtype definitions ────────────────────────────
NORMAL_CODES = {"NORM", "SR", "NSR", "SBRAD", "STACH", "SARRH", "PACE"}
ARRHYTHMIA_MAP = {
    "AFIB":  "Atrial Fibrillation",
    "AFLT":  "Atrial Flutter",
    "SVTAC": "Supraventricular Tachycardia",
    "PSVT":  "Paroxysmal SVT",
    "STACH": "Sinus Tachycardia",
    "SBRAD": "Sinus Bradycardia",
    "SARRH": "Sinus Arrhythmia",
    "1AVB":  "1st-degree AV Block",
    "2AVB":  "2nd-degree AV Block",
    "3AVB":  "3rd-degree AV Block",
    "PACE":  "Pacemaker Rhythm",
    "PVC":   "Premature Ventricular Contractions",
    "BIGU":  "Bigeminy",
    "TRIGU": "Trigeminy",
    "WPW":   "Wolff-Parkinson-White",
    "SVARR": "Supraventricular Arrhythmia",
}

def extract_arrhythmia_subtypes(scp_str):
    """Return list of arrhythmia subtype codes for one record.
    Handles both dict format {'CODE': conf} and list-of-tuples [('CODE', conf), ...]."""
    try:
        parsed = ast.literal_eval(scp_str)
        if isinstance(parsed, dict):
            codes = list(parsed.keys())
        elif isinstance(parsed, list):
            codes = [item[0] if isinstance(item, (list, tuple)) else item for item in parsed]
        else:
            codes = []
        return [code for code in codes if code not in NORMAL_CODES]
    except Exception:
        return []

# ── Build per-row subtype list for all test rows ───────────────
subtype_lookup = {}
for i, ecg_id in enumerate(ecg_ids):
    if ecg_id > 0 and ecg_id in stmt_by_ecgid.index:
        subtype_lookup[i] = extract_arrhythmia_subtypes(
            stmt_by_ecgid.loc[ecg_id, "scp_codes"]
        )
    else:
        subtype_lookup[i] = []

# ── Load models ────────────────────────────────────────────────
model_files = {
    "Logistic Regression": "logistic_regression",
    "Random Forest":       "random_forest",
    "XGBoost":             "xgboost",
}
models = {name: joblib.load(os.path.join(MODELS_DIR, f"{fname}.pkl"))
          for name, fname in model_files.items()}

COLORS = {"Logistic Regression": "#4472C4", "Random Forest": "#ED7D31", "XGBoost": "#70AD47"}

# ── Compute per-subtype FN counts for each model ───────────────
print("\nComputing per-subtype false-negative counts...")

# Identify all subtypes present in the test set arrhythmia cases
all_subtypes = set()
arrhythmia_mask = y_test == 1
for i, is_arr in enumerate(arrhythmia_mask):
    if is_arr:
        all_subtypes.update(subtype_lookup[i])
all_subtypes = sorted(all_subtypes)
print(f"Subtypes found in test set arrhythmia cases: {all_subtypes}")

# Total count of each subtype among all arrhythmia test cases
subtype_totals = defaultdict(int)
for i, is_arr in enumerate(arrhythmia_mask):
    if is_arr:
        for code in subtype_lookup[i]:
            subtype_totals[code] += 1

fn_counts = {}  # model_name -> {subtype: count}
fn_rates  = {}  # model_name -> {subtype: rate}

for model_name, model in models.items():
    y_pred = model.predict(X_test)
    fn_mask = (y_test == 1) & (y_pred == 0)  # true arrhythmia, predicted normal

    counts = defaultdict(int)
    for i, is_fn in enumerate(fn_mask):
        if is_fn:
            for code in subtype_lookup[i]:
                counts[code] += 1

    fn_counts[model_name] = dict(counts)
    fn_rates[model_name]  = {
        code: round(counts[code] / max(subtype_totals[code], 1), 4)
        for code in counts
    }

    total_fn = fn_mask.sum()
    print(f"\n  {model_name}: {total_fn} total FNs")
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for code, cnt in top:
        label = ARRHYTHMIA_MAP.get(code, code)
        rate  = fn_rates[model_name][code]
        print(f"    {code:<8} ({label:<35}) FN={cnt:3d}  FN-rate={rate:.3f}")

# ── Save FN counts CSV ─────────────────────────────────────────
counts_df = pd.DataFrame(fn_counts, index=all_subtypes).fillna(0).astype(int)
counts_df.index.name = "subtype_code"
counts_df.insert(0, "subtype_label",
                 [ARRHYTHMIA_MAP.get(c, c) for c in counts_df.index])
counts_df.insert(1, "total_cases",
                 [subtype_totals.get(c, 0) for c in counts_df.index])
counts_df = counts_df.sort_values("total_cases", ascending=False)
counts_path = os.path.join(RESULTS_DIR, "subtype_false_negative_counts.csv")
counts_df.to_csv(counts_path)
print(f"\nSaved: {counts_path}")

# ── Save FN rates CSV ──────────────────────────────────────────
rates_df = pd.DataFrame(fn_rates, index=all_subtypes).fillna(0).round(4)
rates_df.index.name = "subtype_code"
rates_df.insert(0, "subtype_label",
                [ARRHYTHMIA_MAP.get(c, c) for c in rates_df.index])
rates_df.insert(1, "total_cases",
                [subtype_totals.get(c, 0) for c in rates_df.index])
rates_df = rates_df.sort_values("total_cases", ascending=False)
rates_path = os.path.join(RESULTS_DIR, "subtype_false_negative_rates.csv")
rates_df.to_csv(rates_path)
print(f"Saved: {rates_path}")

# ── Plot: Grouped bar chart — FN counts by subtype ────────────
model_names = list(models.keys())

# Keep top N subtypes by total case count for readability
top_n       = min(12, len(all_subtypes))
top_subtypes = counts_df.head(top_n).index.tolist()
labels      = [ARRHYTHMIA_MAP.get(c, c) for c in top_subtypes]

x      = np.arange(len(top_subtypes))
width  = 0.25
n_bars = len(model_names)
offsets = np.linspace(-(n_bars - 1) * width / 2,
                       (n_bars - 1) * width / 2, n_bars)

fig, ax = plt.subplots(figsize=(16, 7))
for i, model_name in enumerate(model_names):
    vals = [fn_counts[model_name].get(c, 0) for c in top_subtypes]
    ax.bar(x + offsets[i], vals, width, label=model_name,
           color=COLORS[model_name], alpha=0.85, edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
ax.set_xlabel("Arrhythmia Subtype", fontsize=12)
ax.set_ylabel("Number of False Negatives (Missed Cases)", fontsize=12)
ax.set_title(
    f"False Negatives by Arrhythmia Subtype — Top {top_n} Most Prevalent\n"
    "(higher = more missed cases of that subtype)",
    fontsize=13, fontweight="bold"
)
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

fig_path = os.path.join(FIGURES_DIR, "subtype_false_negative_barplot.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {fig_path}")

print("\n[DONE] Step 8 complete.")
