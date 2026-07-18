from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "mobile_classifier.pth"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

FEATURE_COLUMNS = [
    "battery_power",
    "blue",
    "clock_speed",
    "dual_sim",
    "fc",
    "four_g",
    "int_memory",
    "m_dep",
    "mobile_wt",
    "n_cores",
    "pc",
    "px_height",
    "px_width",
    "ram",
    "sc_h",
    "sc_w",
    "talk_time",
    "three_g",
    "touch_screen",
    "wifi",
]

BOOLEAN_FEATURES = {
    "blue",
    "dual_sim",
    "four_g",
    "three_g",
    "touch_screen",
    "wifi",
}

CATEGORIES = {
    "hardware": {"label": "Donanım & Performans", "icon": "🧠"},
    "screen_camera": {"label": "Ekran & Kamera", "icon": "📸"},
    "connectivity": {"label": "Bağlantı & Ağ", "icon": "📶"},
    "physical": {"label": "Tasarım & Fiziksel", "icon": "📐"},
}

FEATURE_META = {
    "battery_power": {"label": "Batarya Gücü (mAh)", "min": 400, "max": 2100, "step": 1, "category": "hardware", "icon": "🔋"},
    "blue": {"label": "Bluetooth", "type": "checkbox", "category": "connectivity", "icon": "📶"},
    "clock_speed": {"label": "İşlemci Hızı (GHz)", "min": 0.3, "max": 3.5, "step": 0.1, "category": "hardware", "icon": "⚡"},
    "dual_sim": {"label": "Çift SIM", "type": "checkbox", "category": "connectivity", "icon": "🎴"},
    "fc": {"label": "Ön Kamera Çözünürlüğü (MP)", "min": 0, "max": 22, "step": 1, "category": "screen_camera", "icon": "🤳"},
    "four_g": {"label": "4G Desteği", "type": "checkbox", "category": "connectivity", "icon": "📶"},
    "int_memory": {"label": "Dahili Hafıza (GB)", "min": 1, "max": 70, "step": 1, "category": "hardware", "icon": "💾"},
    "m_dep": {"label": "Telefon Kalınlığı (cm)", "min": 0.05, "max": 1.1, "step": 0.05, "category": "physical", "icon": "📏"},
    "mobile_wt": {"label": "Ağırlık (g)", "min": 70, "max": 220, "step": 1, "category": "physical", "icon": "⚖️"},
    "n_cores": {"label": "Çekirdek Sayısı", "min": 1, "max": 10, "step": 1, "category": "hardware", "icon": "🧠"},
    "pc": {"label": "Arka Kamera Çözünürlüğü (MP)", "min": 0, "max": 24, "step": 1, "category": "screen_camera", "icon": "📸"},
    "px_height": {"label": "Piksel Yüksekliği (px)", "min": 0, "max": 2100, "step": 1, "category": "screen_camera", "icon": "↕️"},
    "px_width": {"label": "Piksel Genişliği (px)", "min": 400, "max": 2100, "step": 1, "category": "screen_camera", "icon": "↔️"},
    "ram": {"label": "RAM Kapasitesi (MB)", "min": 200, "max": 4200, "step": 1, "category": "hardware", "icon": "⚡"},
    "sc_h": {"label": "Ekran Yüksekliği (cm)", "min": 4, "max": 21, "step": 0.5, "category": "screen_camera", "icon": "📐"},
    "sc_w": {"label": "Ekran Genişliği (cm)", "min": 0, "max": 20, "step": 0.5, "category": "screen_camera", "icon": "📐"},
    "talk_time": {"label": "Konuşma Süresi (saat)", "min": 1, "max": 24, "step": 1, "category": "physical", "icon": "📞"},
    "three_g": {"label": "3G Desteği", "type": "checkbox", "category": "connectivity", "icon": "📶"},
    "touch_screen": {"label": "Dokunmatik Ekran", "type": "checkbox", "category": "screen_camera", "icon": "👉"},
    "wifi": {"label": "Wi-Fi Desteği", "type": "checkbox", "category": "connectivity", "icon": "📶"},
}

PRESETS = {
    "entry": {
        "label": "Giriş Seviyesi (Ucuz)",
        "icon": "🪙",
        "values": {
            "battery_power": 600,
            "blue": 0,
            "clock_speed": 0.6,
            "dual_sim": 0,
            "fc": 2,
            "four_g": 0,
            "int_memory": 8,
            "m_dep": 0.9,
            "mobile_wt": 190,
            "n_cores": 2,
            "pc": 5,
            "px_height": 200,
            "px_width": 600,
            "ram": 512,
            "sc_h": 8,
            "sc_w": 3,
            "talk_time": 4,
            "three_g": 0,
            "touch_screen": 0,
            "wifi": 0,
        }
    },
    "midrange": {
        "label": "Fiyat/Performans (Orta)",
        "icon": "⚖️",
        "values": {
            "battery_power": 1200,
            "blue": 1,
            "clock_speed": 1.6,
            "dual_sim": 1,
            "fc": 6,
            "four_g": 1,
            "int_memory": 32,
            "m_dep": 0.5,
            "mobile_wt": 150,
            "n_cores": 4,
            "pc": 12,
            "px_height": 700,
            "px_width": 1200,
            "ram": 2048,
            "sc_h": 12,
            "sc_w": 6,
            "talk_time": 12,
            "three_g": 1,
            "touch_screen": 1,
            "wifi": 1,
        }
    },
    "flagship": {
        "label": "Amiral Gemisi (Premium)",
        "icon": "👑",
        "values": {
            "battery_power": 1950,
            "blue": 1,
            "clock_speed": 3.0,
            "dual_sim": 1,
            "fc": 16,
            "four_g": 1,
            "int_memory": 64,
            "m_dep": 0.2,
            "mobile_wt": 110,
            "n_cores": 8,
            "pc": 20,
            "px_height": 1500,
            "px_width": 1900,
            "ram": 3950,
            "sc_h": 19,
            "sc_w": 15,
            "talk_time": 20,
            "three_g": 1,
            "touch_screen": 1,
            "wifi": 1,
        }
    }
}

PRICE_LABELS = {
    0: "Düşük Maliyet",
    1: "Orta Maliyet",
    2: "Yüksek Maliyet",
    3: "Çok Yüksek Maliyet",
}

PRICE_COLORS = {
    0: "#10b981", # Emerald
    1: "#f59e0b", # Amber
    2: "#f97316", # Orange
    3: "#ef4444", # Red
}

