# test_fertilizer.py

import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

BASE_DIR  = os.path.dirname(__file__)
# cleaned fertilizer data (make sure you ran your clean_prices.py to produce this)
DATA_CSV  = os.path.join(BASE_DIR, '..', 'data', 'processed', 'fertilizer_clean.csv')
# your trained sklearn model
MODEL_PKL = os.path.join(BASE_DIR, '..', 'src', 'app', 'models', 'fert_rec.pkl')

# ─── Load Data & Model ───────────────────────────────────────────────────────
df    = pd.read_csv(DATA_CSV)
model = joblib.load(MODEL_PKL)

# ─── Inspect columns so you can confirm naming ───────────────────────────────
print("Columns in fertilizer_clean.csv:", df.columns.tolist())

# ─── Feature + target columns ────────────────────────────────────────────────
# Update these to exactly match your cleaned CSV’s column names:
feature_cols = ['N','P','K','temperature','humidity','rainfall']
target_col   = 'label'  # e.g. the column you named for fertilizer type

# ─── Prepare X and y ─────────────────────────────────────────────────────────
X_true = df[feature_cols].astype(float).values
y_true = df[target_col].values

# ─── Predict & Score ─────────────────────────────────────────────────────────
y_pred = model.predict(X_true)
acc    = accuracy_score(y_true, y_pred)

print(f"Fertilizer Rec Model Accuracy: {acc:.4f} ({acc*100:.1f}%)")
