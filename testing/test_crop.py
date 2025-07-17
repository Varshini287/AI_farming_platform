import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(__file__)
DATA_CSV  = os.path.join(BASE_DIR, '..', 'data', 'processed', 'crop_recommendation_clean.csv')
MODEL_PKL = os.path.join(BASE_DIR, '..', 'src', 'app', 'models', 'crop_rec.pkl')

# ─── Load ─────────────────────────────────────────────────────────────────────
df    = pd.read_csv(DATA_CSV)
model = joblib.load(MODEL_PKL)

# ─── Prepare X, y ─────────────────────────────────────────────────────────────
feature_cols = ['N','P','K','temperature','humidity','ph','rainfall']
X_true       = df[feature_cols].values
y_true       = df['label'].values       # adjust if your label column is named differently

# ─── Predict & Score ──────────────────────────────────────────────────────────
y_pred = model.predict(X_true)
acc    = accuracy_score(y_true, y_pred)

print(f"Crop Rec Model Accuracy: {acc:.4f} ({acc*100:.1f}%)")
