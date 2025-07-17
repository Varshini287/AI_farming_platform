# src/training/train_crop.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_crop_model(
    input_csv: str,
    model_out: str,
    n_estimators: int = 100,
    random_state: int = 42
):
    # 1. Load cleaned data
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    # 2. Split features and target
    X = df.drop('label', axis=1)    # replace 'label' with your target column name
    y = df['label']

    # 3. Train‑test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # 4. Initialize & fit model
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    print("Training RandomForestClassifier...")
    clf.fit(X_train, y_train)

    # 5. Evaluate on test set
    preds = clf.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, preds))

    # 6. Cross‑validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"5‑fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 7. Save the model
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(clf, model_out)
    print(f"Model saved to {model_out}")

if __name__ == "__main__":
    # Update these paths if needed
    cleaned_csv = os.path.join(
        os.path.dirname(__file__),
        '../../data/processed/crop_recommendation_clean.csv'
    )
    model_file  = os.path.join(
        os.path.dirname(__file__),
        '../../models/crop_rec.pkl'
    )

    train_crop_model(
        input_csv=cleaned_csv,
        model_out=model_file,
        n_estimators=150
    )
