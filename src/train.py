import json
import os
import sys
import joblib 
import hashlib
from pathlib import Path

import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main() -> None:
    config_path = Path("configs/pipeline_config.json")
    if not config_path.exists():
        print("ERROR: Missing configs/pipeline_config.json")
        sys.exit(1)

    with config_path.open("r") as f:
        cfg = json.load(f)

    data_path = Path(cfg["training_data_path"])
    out_dir = Path(cfg["output_dir"])

    if not data_path.exists():
        print(f"ERROR: Training data not found at {data_path}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    # Expected columns for our synthetic dataset 
    # (we’ll generate it next step if missing)
    # Features: age, prior_admissions, meds_count
    # Label: readmitted (0/1)
    required = {"age", "prior_admissions", "meds_count", "readmitted"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Dataset missing columns: {sorted(missing)}")
        sys.exit(1)

    X = df[["age", "prior_admissions", "meds_count"]]
    y = df["readmitted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
        
    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)
    print(f'Saved trained model to {model_path}')

    # Week 3 day 2 : Model integrity hash
    hasher = hashlib.sha256()
    with model_path.open("rb") as mf:
        hasher.update(mf.read())
    model_hash = hasher.hexdigest()
    
    hash_path = out_dir / "model.sha256"
    with hash_path.open("w") as hf:
        hf.write(model_hash)
        
    print(f"Saved model integrity hash to {hash_path}")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Week 3: Accuracy thereshhold enforcement
    threshold = float(cfg.get("min_accuracy", 0.85)) #default if not set
    passed = acc >= threshold

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(
            {
                "accuracy": acc,
                "min_accuracy": threshold,
                "passed_threshold": passed,
            },
            f,
            indent=2
        )

    print(f"Training complete. Accuracy={acc:.4f}")
    print(f"Minimum required accuracy={threshold:.2f}")
    print(f"Saved metrics to {metrics_path}")

    if not passed:
        print(
            f"ERROR: Accuracy {acc:.4f} is below required threshold {threshold:.2f}"
            
        )
        sys.exit(1)

if __name__ == "__main__":
    main()