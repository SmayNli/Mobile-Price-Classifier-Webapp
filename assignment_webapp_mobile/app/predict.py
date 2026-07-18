from functools import lru_cache

import joblib
import numpy as np
import torch

from app.config import FEATURE_COLUMNS, MODEL_PATH, PRICE_COLORS, PRICE_LABELS, SCALER_PATH
from app.model import NonlinearModel


@lru_cache(maxsize=1)
def load_model() -> NonlinearModel:
    model = NonlinearModel()
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@lru_cache(maxsize=1)
def load_scaler():
    return joblib.load(SCALER_PATH)


def predict_price(features: dict) -> dict:
    model = load_model()
    scaler = load_scaler()

    values = [float(features[name]) for name in FEATURE_COLUMNS]
    scaled = scaler.transform(np.array([values], dtype=np.float32))

    with torch.inference_mode():
        logits = model(torch.tensor(scaled, dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).squeeze(0).tolist()
        predicted_class = int(torch.argmax(logits, dim=1).item())

    return {
        "predicted_class": predicted_class,
        "predicted_label": PRICE_LABELS[predicted_class],
        "color": PRICE_COLORS[predicted_class],
        "probabilities": [
            {
                "class": idx,
                "label": PRICE_LABELS[idx],
                "probability": round(prob, 4),
                "color": PRICE_COLORS[idx],
            }
            for idx, prob in enumerate(probabilities)
        ],
    }
