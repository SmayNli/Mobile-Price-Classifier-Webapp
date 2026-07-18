"""Notebook ile aynı pipeline'ı kullanarak model ve scaler dosyalarını üretir."""

from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model import NonlinearModel

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_CSV = BASE_DIR / "train.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "mobile_classifier.pth"
SCALER_PATH = MODELS_DIR / "scaler.joblib"


def main() -> None:
    df = pd.read_csv(TRAIN_CSV)
    X = df.drop("price_range", axis=1).values
    y = df["price_range"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    model = NonlinearModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    accuracy = MulticlassAccuracy(4)

    torch.manual_seed(42)
    epochs = 200

    for epoch in range(epochs):
        model.train()
        logits = model(X_train_tensor)
        loss = loss_fn(logits, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            y_pred = torch.softmax(logits, dim=1).argmax(dim=1)
            acc = accuracy(y_pred, y_train_tensor)

            test_logits = model(X_test_tensor)
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
            test_acc = accuracy(test_pred, y_test_tensor)
            test_loss = loss_fn(test_logits, y_test_tensor)

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}, "
                f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}"
            )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"Model kaydedildi: {MODEL_PATH}")
    print(f"Scaler kaydedildi: {SCALER_PATH}")


if __name__ == "__main__":
    main()
