"""
preprocess.py
ECG Trustworthiness Framework — Nazish Atta, George Washington University
Step 2: Imputation, scaling, and SMOTE oversampling (train only).
"""

import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR   = os.path.join(OUTPUT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("STEP 2 — PREPROCESSING")
print("=" * 60)

# ── Load raw splits ───────────────────────────────────────────
train = pd.read_csv(os.path.join(DATA_DIR, "train_raw.csv"), index_col=0)
val   = pd.read_csv(os.path.join(DATA_DIR, "val_raw.csv"),   index_col=0)
test  = pd.read_csv(os.path.join(DATA_DIR, "test_raw.csv"),  index_col=0)

X_train = train.drop("label", axis=1)
y_train = train["label"]
X_val   = val.drop("label", axis=1)
y_val   = val["label"]
X_test  = test.drop("label", axis=1)
y_test  = test["label"]

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# ── Imputation (median, fit on train only) ────────────────────
missing = X_train.isnull().sum().sum()
print(f"\nMissing values in train: {missing}")

imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train),
                        columns=X_train.columns, index=X_train.index)
X_val   = pd.DataFrame(imputer.transform(X_val),
                        columns=X_val.columns, index=X_val.index)
X_test  = pd.DataFrame(imputer.transform(X_test),
                        columns=X_test.columns, index=X_test.index)
print("✅ Median imputation applied")

# ── Scaling (StandardScaler, fit on train only) ───────────────
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train),
                        columns=X_train.columns, index=X_train.index)
X_val   = pd.DataFrame(scaler.transform(X_val),
                        columns=X_val.columns, index=X_val.index)
X_test  = pd.DataFrame(scaler.transform(X_test),
                        columns=X_test.columns, index=X_test.index)
print("✅ StandardScaler applied")

# ── SMOTE (train only) ────────────────────────────────────────
print(f"\nBefore SMOTE: {dict(y_train.value_counts().sort_index())}")
smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE:  {dict(pd.Series(y_train_sm).value_counts().sort_index())}")
print(f"Train size after SMOTE: {len(X_train_sm)}")

# ── Save processed data ───────────────────────────────────────
train_proc = pd.DataFrame(X_train_sm, columns=X_train.columns)
train_proc["label"] = y_train_sm
train_proc.to_csv(os.path.join(DATA_DIR, "train_processed.csv"), index=False)

val_proc = X_val.copy()
val_proc["label"] = y_val.values
val_proc.to_csv(os.path.join(DATA_DIR, "val_processed.csv"), index=False)

test_proc = X_test.copy()
test_proc["label"] = y_test.values
test_proc.to_csv(os.path.join(DATA_DIR, "test_processed.csv"), index=False)

# Save fitted transformers
import joblib
models_dir = os.path.join(OUTPUT_DIR, "models")
os.makedirs(models_dir, exist_ok=True)
joblib.dump(imputer, os.path.join(models_dir, "imputer.pkl"))
joblib.dump(scaler,  os.path.join(models_dir, "scaler.pkl"))
print("\n✅ Transformers saved to outputs/models/")
print("✅ Processed data saved to outputs/data/")
print("\n✅ Step 2 complete.")
