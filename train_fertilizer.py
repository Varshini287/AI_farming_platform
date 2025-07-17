# src/training/train_fertilizer.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_fertilizer_model(
    input_csv: str,
    model_out: str,
    n_estimators: int = 100,
    random_state: int = 42
):
    # 1. Load cleaned data
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")
    print("Columns:", df.columns.tolist())

    # 2. Separate features & target
    target_col = 'Fertilizer Name'
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # 3. One‑hot encode categorical features
    X = pd.get_dummies(
        X,
        columns=['Soil Type', 'Crop Type'],
        drop_first=True
    )
    print("After encoding, feature shape:", X.shape)

    # 4. Train‑test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )

    # 5. Initialize & fit RandomForest
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    print("Training RandomForestClassifier for fertilizer recommendation...")
    clf.fit(X_train, y_train)

    # 6. Evaluate on test set
    preds = clf.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, preds))

    # 7. Cross‑validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"5‑fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 8. Save the model
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(clf, model_out)
    print(f"Model saved to {model_out}")

if __name__ == "__main__":
    base = os.path.dirname(__file__) + '/../../'
    cleaned_csv = os.path.join(base, 'data/processed/fertilizer_clean.csv')
    model_file  = os.path.join(base, 'models/fert_rec.pkl')

    train_fertilizer_model(
        input_csv=cleaned_csv,
        model_out=model_file,
        n_estimators=150
    )
