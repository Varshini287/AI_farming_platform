# train_fertilizer_6feat.py
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_6feat_model(csv_path, model_out):
    df = pd.read_csv(csv_path)
    # 1) rename columns to match our API
    df = df.rename(columns={
        'Temparature': 'temperature',
        'Humidity':    'humidity',
        'Moisture':    'rainfall',
        'Nitrogen':    'N',
        'Potassium':   'K',
        'Phosphorous': 'P'
    })
    # ensure all needed columns exist
    needed = ['temperature','humidity','rainfall','N','K','P','Fertilizer Name']
    missing = set(needed) - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # 2) features & target
    X = df[['N','P','K','temperature','humidity','rainfall']].astype(float)
    y = df['Fertilizer Name']

    # 3) train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) train
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 5) evaluate
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))

    # 6) save
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(clf, model_out)
    print("Saved model to", model_out)

if __name__ == "__main__":
    BASE = os.path.dirname(__file__)
    CSV  = os.path.join(BASE, "data/processed/fertilizer_clean.csv")
    OUT  = os.path.join(BASE, "models/fert_rec.pkl")
    train_6feat_model(CSV, OUT)
