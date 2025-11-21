import pandas as pd
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def train(test_size=0.2, random_state=42, calibrate=True):
    print("1. Loading Data...")
    try:
        # Load data
        df = pd.read_csv('training_data.csv')

        # Drop empty "garbage" columns caused by trailing commas
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Fill any missing values with 0 just in case
        df = df.fillna(0)

        # 2. Separate Features (X) and Target (y)
        X = df.drop('prognosis', axis=1)
        y = df['prognosis']

        # 3. Train/Test split for evaluation
        print("2. Splitting data (train/test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        # 4. Train Model with class balancing
        print("3. Training Model (RandomForest, class_weight='balanced')...")
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
        model.fit(X_train, y_train)

        # Save feature importances before calibration (helpful for active questioning)
        try:
            feature_importances = model.feature_importances_.tolist()
        except Exception:
            feature_importances = None

        trained_model = model
        # 5. Optional calibration of predicted probabilities
        if calibrate:
            print("4. Calibrating probabilities (isotonic)...")
            try:
                calib = CalibratedClassifierCV(base_estimator=model, cv=3, method='isotonic')
                calib.fit(X_train, y_train)
                trained_model = calib
            except Exception as e:
                print(f"⚠️ Calibration failed: {e}. Continuing with uncalibrated model.")

        # 6. Evaluate
        print("5. Evaluating on hold-out set...")
        preds = trained_model.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)
        print(classification_report(y_test, preds))
        cm = confusion_matrix(y_test, preds)

        # 7. Save Model & Metadata
        print("6. Saving model and metadata...")
        with open("model.pkl", "wb") as f:
            pickle.dump(trained_model, f)
        with open("columns.pkl", "wb") as f:
            pickle.dump(X.columns.tolist(), f)

        meta = {
            'columns': X.columns.tolist(),
            'classes': trained_model.classes_.tolist() if hasattr(trained_model, 'classes_') else None,
            'trained_at': datetime.utcnow().isoformat() + 'Z',
            'metrics': report,
            'confusion_matrix_shape': cm.shape,
            'feature_importances': feature_importances,
        }
        with open('meta.pkl', 'wb') as f:
            pickle.dump(meta, f)

        print("✅ Success! Model trained and saved to 'model.pkl' (metadata in 'meta.pkl').")
        print("You can now run 'python app.py'.")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure 'training_data.csv' is in this folder and is correctly formatted.")


if __name__ == "__main__":
    train()