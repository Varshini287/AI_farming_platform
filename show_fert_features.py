# show_fert_features.py
import joblib

# 1. Load your trained fertilizer model
model = joblib.load('models/fert_rec.pkl')

# 2. Print how many and what features it expects
print(f"Model expects {model.n_features_in_} features:")
print(list(model.feature_names_in_))
