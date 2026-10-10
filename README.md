# ⚡ QueueFlux: Frontend & Backend Control Tower

> **Branch: `frontend-backend`**  
> Technical Ownership: **Frontend & Web Backend Engineering**  
> This branch isolates the responsive Canteen Digital Twin Web Interface, the FastAPI REST Telemetry Server, and the Streamlit Operations Center.

---

## 🏛️ Module Overview

```
frontend-backend/
├── frontend/
│   └── index.html           # Modern glassmorphic Canteen Digital Twin UI (Tailwind CSS, Chart.js, Lucide)
├── api.py                   # FastAPI REST backend serving simulation steps, forecasts & static files
├── app.py                   # Full Streamlit Operations Center dashboard with multi-tab scientific views
├── requirements.txt         # Web stack dependencies (fastapi, uvicorn, streamlit, altair)
├── README.md                # Dedicated Frontend & Backend documentation & API specification
├── .vscode/                 # Launch profiles & tasks to run api.py and app.py
└── .gitignore               # Standard ignore patterns
```

---

## 🖥️ UI Implementations & Architectures

### 1. Modern Canteen Digital Twin Web UI (`frontend/index.html`)
A high-production dark-mode glassmorphic single-page web application featuring:
* **Physical Two-Stage Digital Twin**:
  * **Stage A (Billing Counters)**: Visual customer queue line, active counter badges, cashier processing status.
  * **Conveyor Transfer Channel**: Animated pulse showing orders transferring downstream upon payment.
  * **Stage B (Kitchen Prep Stations)**: Order backlog tiles categorized by item type (Beverage, Snack, Meal, Combo) and active chef stations.
* **Live Telemetry HUD**: Real-time cards displaying Actual Ground Truth vs. Bayesian Particle Filter expected queue lengths, 90% credible intervals, pending workload in minutes, and predicted meal waiting times.
* **Live Streaming Chart.js Visuals**: Dual charts tracking historical queue backlogs and 5-minute forward probabilistic forecasts.
* **Disruption Scenario Injectors**: One-click operational buttons:
  * 🟢 **Normal Flow**: Baseline equilibrium operations.
  * ⚡ **Rush Hour Surge**: $2.2\times$ Poisson arrival multiplier.
  * 🛑 **POS Slowdown**: Service duration elongation at Stage A.
  * ⚠️ **Counter Breakdown**: Reductions in active cashier resources.
  * 🍳 **Kitchen Slowdown**: Downstream chef bottlenecks.
* **Uncertainty-Aware Decision Card**: Recommends optimal resource interventions with Bayesian particle stability percentage ($R(a^*)$) and projected waiting time reduction.
* **Modal Benchmark Runner**: Executes quantitative benchmarks comparing QueueFlux against Baselines A, B, and C.
* **Zero-Setup Hybrid Engine**:
  * When `api.py` is running, automatically connects to `http://localhost:8000` via REST API.
  * When opened directly offline (`file:///...`), automatically switches to the built-in JavaScript client-side simulation engine matching the exact mathematical formulas.

---

### 2. FastAPI REST Telemetry Server (`api.py`)
Exposes the simulation state, stepping controls, disruption injection, and benchmarking endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the `frontend/index.html` static web interface |
| `GET` | `/api/state` | Returns current ground truth and particle filter estimated queue telemetry |
| `POST` | `/api/step` | Advances the simulation by 1 minute, runs state filtering and 5-min forecasting |
| `POST` | `/api/disruption` | Injects an operational scenario (`NORMAL`, `DEMAND_SURGE`, `BILLING_SLOWDOWN`, etc.) |
| `POST` | `/api/config` | Updates active billing counters and kitchen preparation resources |
| `POST` | `/api/reset` | Resets the simulation clock and queue states to minute 0 |
| `GET` | `/api/history` | Retrieves the complete chronological telemetry history trace |
| `GET` | `/api/benchmark` | Executes quantitative 30-minute benchmark comparing QueueFlux to Baselines |
| `GET` | `/api/propagation` | Runs matched paired counterfactual disruption impact analysis |

---

### 3. Streamlit Operations Center (`app.py`)
A comprehensive scientific dashboard designed for in-depth algorithmic reviews:
* **Tab 1: 📡 Live Operations & Pipeline**: Multi-series Altair plots of Stage A and Stage B queue dynamics.
* **Tab 2: 🔮 5-Minute Probabilistic Forecast**: Projected backlogs and latency breakdown by food category.
* **Tab 3: 🌊 Cross-Stage Disruption Propagation**: Interactive paired counterfactual analysis.
* **Tab 4: 🎯 Uncertainty-Aware Decision Optimizer**: Mathematical evaluation of operational actions.
* **Tab 5: 🏆 Quantitative Benchmark**: Real-time evaluation against Baselines A, B, and C.

---

## 🚀 How to Launch

### Option 1: Launch FastAPI Server & Web UI
```bash
python api.py
```
Open your browser at **`http://localhost:8000`**.  
*(Or double-click `frontend/index.html` to run in standalone browser mode).*

### Option 2: Launch Streamlit Operations Center
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 🧪 Testing the API

```bash
# Verify FastAPI can initialize and serve endpoints
python -c "from api import app; from fastapi.testclient import TestClient; client = TestClient(app); print('API Status:', client.get('/api/state').status_code)"
```
