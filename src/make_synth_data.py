import numpy as np
import pandas as pd
from pathlib import Path


def main() -> None:
    out_path = Path("data/train_readmission.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 1000
    rng = np.random.default_rng(42)

    age = rng.integers(18, 90, size=n)
    prior_admissions = rng.integers(0, 10, size=n)
    meds_count = rng.integers(0, 25, size=n)

    # Simple synthetic relationship for readmission probability
    risk = (
        0.03 * (age - 50)
        + 0.35 * prior_admissions
        + 0.06 * meds_count
        + rng.normal(0, 1.0, size=n)
    )

    # Convert risk score into a probability and then into 0/1 label
    prob = 1 / (1 + np.exp(-risk / 3.0))
    readmitted = (prob > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "prior_admissions": prior_admissions,
            "meds_count": meds_count,
            "readmitted": readmitted,
        }
    )

    df.to_csv(out_path, index=False)
    print(f"Wrote synthetic dataset to {out_path} with {len(df)} rows")


if __name__ == "__main__":
    main()