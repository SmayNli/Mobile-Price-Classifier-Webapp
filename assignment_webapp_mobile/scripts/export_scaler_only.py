"""Notebook ile aynı train_test_split üzerinde scaler'ı kaydeder (model eğitmez)."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_CSV = BASE_DIR / "train.csv"
MODELS_DIR = BASE_DIR / "models"
SCALER_PATH = MODELS_DIR / "scaler.joblib"


def main() -> None:
    df = pd.read_csv(TRAIN_CSV)
    X = df.drop("price_range", axis=1).values

    X_train, _ = train_test_split(X, test_size=0.20, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler kaydedildi: {SCALER_PATH}")


if __name__ == "__main__":
    main()
