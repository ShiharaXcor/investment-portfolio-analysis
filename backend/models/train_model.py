
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime
import json 
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]  
print(BASE_DIR)


PROCESSED_DIR = BASE_DIR / "processed"
FEATURE_FILE = os.path.join(PROCESSED_DIR, "features.csv")
MODEL_FILE = os.path.join(PROCESSED_DIR, "model.joblib")
LAST_TRAIN_FILE = os.path.join(PROCESSED_DIR, "last_train_date.txt") 
FEATURES_JSON = os.path.join(PROCESSED_DIR, "feature_cols.json")  

def save_last_train_date():
    with open(LAST_TRAIN_FILE, "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d"))

def train():
    print("Loading features...")
    df = pd.read_csv(FEATURE_FILE)
    

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    

    if "pred_return_5d" not in numeric_cols:
        raise ValueError("Target column 'pred_return_5d' not found in numeric columns.")
    
    feature_cols = [c for c in numeric_cols if c != "pred_return_5d"]
    

    with open(FEATURES_JSON, "w") as f:
        json.dump(feature_cols, f)
    print(f"Feature columns saved to {FEATURES_JSON}")

    X = df[feature_cols]
    y = df["pred_return_5d"]

    print(f"Training with features: {feature_cols}")


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training RandomForest...")
    
    # RandomizedSearch parameters
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    }
    
    rf = RandomForestRegressor(random_state=42)
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=20,
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42,
        scoring='r2',
        error_score='raise'  
    )
    
    search.fit(X_train, y_train)

    print("Best parameters:", search.best_params_)

    # Evaluate
    y_pred = search.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse}, R2: {r2}")

    joblib.dump(search.best_estimator_, MODEL_FILE)
    print("Model saved to", MODEL_FILE)

   
    save_last_train_date()
    print("Last training date saved to", LAST_TRAIN_FILE)


if __name__ == "__main__":
    train()