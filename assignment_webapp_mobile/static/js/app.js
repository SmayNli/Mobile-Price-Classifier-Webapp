const form = document.getElementById("predict-form");
const resultCard = document.getElementById("result-card");
const resultBadge = document.getElementById("result-badge");
const resultLabel = document.getElementById("result-label");
const probabilityList = document.getElementById("probability-list");
const submitBtn = document.getElementById("submit-btn");
const resetBtn = document.getElementById("reset-btn");

let featureMeta = {};
let featureColumns = [];
let presets = {};

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  const textEl = submitBtn.querySelector(".btn-text");
  const loaderEl = submitBtn.querySelector(".btn-loader");
  if (isLoading) {
    textEl.style.visibility = "hidden";
    loaderEl.hidden = false;
  } else {
    textEl.style.visibility = "visible";
    loaderEl.hidden = true;
  }
}

function collectFormData() {
  const payload = {};
  featureColumns.forEach(name => {
    const input = document.getElementById(name);
    if (input) {
      if (input.type === "checkbox") {
        payload[name] = input.checked ? 1 : 0;
      } else {
        payload[name] = parseFloat(input.value);
      }
    }
  });
  return payload;
}

function renderProbabilities(probabilities) {
  probabilityList.innerHTML = probabilities
    .map(
      (item) => `
        <div class="probability-item">
          <div class="probability-top">
            <span class="probability-name" style="color: ${item.color}">${item.label}</span>
            <span class="probability-value">${(item.probability * 100).toFixed(1)}%</span>
          </div>
          <div class="progress-track">
            <div
              class="progress-fill"
              style="width: ${item.probability * 100}%; background: linear-gradient(90deg, ${item.color}cc, ${item.color});"
            ></div>
          </div>
        </div>
      `
    )
    .join("");
}

function showError(message) {
  let errorBox = form.querySelector(".error-message");
  if (!errorBox) {
    errorBox = document.createElement("div");
    errorBox.className = "error-message";
    form.appendChild(errorBox);
  }
  errorBox.innerHTML = `⚠️ <strong>Hata:</strong> ${message}`;
}

function clearError() {
  const errorBox = form.querySelector(".error-message");
  if (errorBox) {
    errorBox.remove();
  }
}

function initSynchronizedInputs() {
  featureColumns.forEach(name => {
    const meta = featureMeta[name];
    if (meta.type === "checkbox") return;

    const slider = document.getElementById(`slider-${name}`);
    const number = document.getElementById(name);
    const display = document.getElementById(`val-${name}`);

    if (slider && number) {
      // Sync slider -> number and display
      slider.addEventListener("input", (e) => {
        number.value = e.target.value;
        display.textContent = e.target.value;
        clearPresetActives();
      });

      // Sync number -> slider and display
      number.addEventListener("input", (e) => {
        let val = parseFloat(e.target.value);
        if (isNaN(val)) {
          display.textContent = "—";
          return;
        }
        
        // Clamp value inside range bounds for safety
        if (val < meta.min) val = meta.min;
        if (val > meta.max) val = meta.max;
        
        slider.value = val;
        display.textContent = val;
        clearPresetActives();
      });
    }
  });
}

function initTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const targetTab = tab.dataset.tab;
      document.querySelectorAll(".tab-panel").forEach(panel => {
        if (panel.id === `panel-${targetTab}`) {
          panel.classList.add("active");
        } else {
          panel.classList.remove("active");
        }
      });
    });
  });
}

function applyPreset(presetKey) {
  const preset = presets[presetKey];
  if (!preset) return;

  document.querySelectorAll(".btn-preset").forEach(btn => {
    if (btn.dataset.preset === presetKey) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  const values = preset.values;
  Object.entries(values).forEach(([name, val]) => {
    const input = document.getElementById(name);
    const slider = document.getElementById(`slider-${name}`);
    const display = document.getElementById(`val-${name}`);

    if (input) {
      if (input.type === "checkbox") {
        input.checked = val === 1;
      } else {
        input.value = val;
        if (slider) slider.value = val;
        if (display) display.textContent = val;
      }
    }
  });
}

function clearPresetActives() {
  document.querySelectorAll(".btn-preset").forEach(btn => {
    btn.classList.remove("active");
  });
}

function initPresets() {
  document.querySelectorAll(".btn-preset").forEach(btn => {
    btn.addEventListener("click", () => {
      applyPreset(btn.dataset.preset);
    });
  });
}

// App Initialization
document.addEventListener("DOMContentLoaded", async () => {
  try {
    // Load configurations from FastAPI
    const response = await fetch("/api/features");
    const data = await response.json();
    
    featureMeta = data.features;
    featureColumns = data.columns;
    presets = data.presets;

    // Bind UI Actions
    initSynchronizedInputs();
    initTabs();
    initPresets();

    // Load midrange preset as default startup specs
    applyPreset("midrange");
  } catch (error) {
    showError("Konfigürasyon yüklenirken hata oluştu: " + error.message);
  }
});

// Submit Form
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  
  // HTML5 Range Validation Check
  let isValid = true;
  featureColumns.forEach(name => {
    const input = document.getElementById(name);
    if (input && !input.checkValidity()) {
      isValid = false;
      input.reportValidity();
    }
  });
  
  if (!isValid) return;
  
  setLoading(true);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectFormData()),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Tahmin sırasında bir hata oluştu.");
    }

    // Display prediction result with glowing effect
    resultCard.hidden = false;
    resultBadge.textContent = data.predicted_class;
    
    // Apply dynamic color mapping for result header and background glow
    resultBadge.style.borderColor = data.color;
    resultBadge.style.color = data.color;
    resultBadge.style.boxShadow = `0 0 25px ${data.color}40, inset 0 2px 10px rgba(0,0,0,0.4)`;
    
    const glowEl = document.getElementById("result-badge-glow");
    if (glowEl) {
      glowEl.style.background = data.color;
      glowEl.style.opacity = "0.2";
    }

    resultLabel.textContent = data.predicted_label;
    resultLabel.style.color = data.color;

    // Custom desc depending on class
    const descMap = {
      0: "Maliyet performansı odaklı giriş seviyesi bir telefon. Temel işlemler için ideal.",
      1: "Günlük kullanım ihtiyaçlarını karşılayabilecek standart bir orta sınıf telefon.",
      2: "Gelişmiş donanım özelliklerine sahip yüksek performanslı üst sınıf telefon.",
      3: "En son teknoloji donanım bileşenlerini barındıran premium amiral gemisi telefon."
    };
    document.getElementById("result-desc").textContent = descMap[data.predicted_class] || "";

    renderProbabilities(data.probabilities);
    
    // Scroll result card into view on mobile devices
    if (window.innerWidth <= 960) {
      resultCard.scrollIntoView({ behavior: "smooth" });
    }
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(false);
  }
});

// Reset Form
resetBtn.addEventListener("click", () => {
  form.reset();
  applyPreset("midrange");
  clearError();
  resultCard.hidden = true;
});
