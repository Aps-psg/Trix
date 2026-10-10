# ⚡ QueueFlux: Propagation-Aware, Uncertainty-Aware Queue Prediction & Control Tower

[![QueueFlux CI](https://github.com/Aps-psg/Trix/actions/workflows/ci.yml/badge.svg)](https://github.com/Aps-psg/Trix/actions/workflows/ci.yml)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FAps-psg%2FTrix)
[![Tests Passing](https://img.shields.io/badge/tests-34%20passed-brightgreen.svg)](https://github.com/Aps-psg/Trix)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/Aps-psg/Trix)

**QueueFlux** is an original, mathematically grounded computational framework and real-time control tower for estimating queue states, predicting waiting times, forecasting congestion, detecting operational disruptions, and recommending optimal operational interventions in a two-stage service facility (Campus Canteen).

> 📋 **Presentation & Viva Defense**: See the full 5-minute script, speaking roles, and Q&A defense cheat sheet in [**`PRESENTATION_GUIDE.md`**](file:///E:/queue-intelligence/PRESENTATION_GUIDE.md).

---

## 🌿 Modular Repository Structure & Branch Navigation Guide

The repository is organized into dedicated, self-contained modular branches so teammates, recruiters, and evaluators can inspect each subsystem in isolation or examine the unified master:

| Branch Name | Focus Area | Technical Ownership | Key Files |
| :--- | :--- | :--- | :--- |
| [**`main`**](https://github.com/Aps-psg/Trix/tree/main) | **Production Integrated Master** | Unified Framework | Complete system integrating Person 1, Person 2, Person 3, and Web UI |
| [**`person1-simulator`**](https://github.com/Aps-psg/Trix/tree/person1-simulator) | **Discrete Simulator & Synthetic Data** | Person 1 Foundation | [`queueflux/simulator/`](file:///E:/queue-intelligence/queueflux/simulator/), [`data/`](file:///E:/queue-intelligence/data/), [`queueflux/models/`](file:///E:/queue-intelligence/queueflux/models/), [`run_demo.py`](file:///E:/queue-intelligence/run_demo.py) |
| [**`core-model`**](https://github.com/Aps-psg/Trix/tree/core-model) | **Core Estimation & Forecasting** | Person 2 Core Model | [`workload_model.py`](file:///E:/queue-intelligence/workload_model.py), [`state_estimator.py`](file:///E:/queue-intelligence/state_estimator.py), [`wait_predictor.py`](file:///E:/queue-intelligence/wait_predictor.py), [`forecast_engine.py`](file:///E:/queue-intelligence/forecast_engine.py), [`test_person2.py`](file:///E:/queue-intelligence/test_person2.py) |
| [**`intelligence-layer`**](https://github.com/Aps-psg/Trix/tree/intelligence-layer) | **Disruptions & Decisions** | Person 3 Intelligence | [`disruption_detector.py`](file:///E:/queue-intelligence/disruption_detector.py), [`propagation_analyzer.py`](file:///E:/queue-intelligence/propagation_analyzer.py), [`decision_optimizer.py`](file:///E:/queue-intelligence/decision_optimizer.py), [`evaluation.py`](file:///E:/queue-intelligence/evaluation.py), [`intelligence_engine.py`](file:///E:/queue-intelligence/intelligence_engine.py) |
| [**`frontend-backend`**](https://github.com/Aps-psg/Trix/tree/frontend-backend) | **Web Control Tower & REST API** | UI / API Engineering | [`frontend/index.html`](file:///E:/queue-intelligence/frontend/index.html), [`api.py`](file:///E:/queue-intelligence/api.py), [`app.py`](file:///E:/queue-intelligence/app.py) |
| [**`main-files`**](https://github.com/Aps-psg/Trix/tree/main-files) | **System Integration Drivers** | System Orchestration | [`main.py`](file:///E:/queue-intelligence/main.py), [`queue_intelligence.py`](file:///E:/queue-intelligence/queue_intelligence.py), [`integration_adapter.py`](file:///E:/queue-intelligence/integration_adapter.py), [`sample_data.json`](file:///E:/queue-intelligence/sample_data.json) |

```bash
# To switch to any branch locally:
git checkout person1-simulator
git checkout core-model
git checkout intelligence-layer
git checkout frontend-backend
git checkout main-files
git checkout main
```

---

## 🏛️ System Architecture

The canteen workflow is modeled as a physical 2-stage queuing system:
$$\text{Customer Arrival } (A_t) \longrightarrow \text{Billing Counter } (Q_t^B) \longrightarrow \text{Kitchen Workflow } (Q_t^K) \longrightarrow \text{Food Collection}$$

```
                                  +------------------------------------+
                                  | Person 1: Queue Simulator          |
                                  | Person 2: State Estimator (Core)   |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | integration_adapter.py             |
                                  | (Normalizes to QueueFluxState)     |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | intelligence_engine.py             |
                                  +---+-----------+-----------+--------+
                                      |           |           |
            +-------------------------+           |           +-------------------------+
            v                                     v                                     v
+-----------------------+             +-----------------------+             +-----------------------+
| disruption_detector   |             | propagation_analyzer  |             | decision_optimizer    |
| (Surges & Slowdowns)  |             | (Cross-Stage Ripples) |             | (Actions & Waiting)   |
+-----------------------+             +-----------------------+             +-----------------------+
                                                  |
                                                  v
                                      +-----------------------+
                                      | evaluation.py         |
                                      | (F1, MAE, Impact %)   |
                                      +-----------------------+
```

---

## 🖥️ Interactive Frontends & UI Control Towers

QueueFlux provides two full-featured, production-grade graphical user interfaces for live monitoring, interactive disruption testing, and executive demonstrations:

### Option 1: Modern Canteen Digital Twin Web UI (FastAPI + HTML5/Tailwind/Chart.js)
A reactive, dark-mode glassmorphic control tower featuring an animated physical model of the two-stage canteen:
*   **Physical Pipeline View**: Live visualization of customer waiting lines at Stage A (Billing Counters) transferring via ticket conveyor to Stage B (Kitchen Prep Stations) with real-time status badges.
*   **Live Queue Telemetry**: Real-time Chart.js tracking comparing Actual Ground Truth vs. Bayesian Particle Filter state estimates and 90% credible interval bands.
*   **5-Minute Probabilistic Forecast**: Forward Monte Carlo projections displaying expected backlog and threshold exceedance risk.
*   **Interactive Operational Controls**: Real-time sliders to scale active billing counters and kitchen staff, plus one-click disruption injection (Rush Hour Surge, POS Slowdown, Counter Breakdown, Kitchen Starvation).
*   **Uncertainty-Aware Decision Cards**: Displays optimal intervention recommendations with Bayesian particle consensus percentages and cost metrics.
*   **Embedded Hybrid Math Engine**: Automatically connects to the FastAPI backend, or runs standalone in any browser with client-side mathematics if the API is offline.

```bash
# Launch FastAPI server and open http://localhost:8000
python api.py
```
*(Or double-click `run_web_ui.bat` on Windows, or open `frontend/index.html` directly in any web browser).*

#### 🌐 Instant 1-Click Vercel Deployment
QueueFlux is pre-configured with [`vercel.json`](file:///E:/queue-intelligence/vercel.json) for zero-config cloud deployment on Vercel:
1. Click the **[Deploy with Vercel](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FAps-psg%2FTrix)** button, or go to [vercel.com/new](https://vercel.com/new) and import `Aps-psg/Trix`.
2. Leave all settings default and click **Deploy**.
3. Your live Canteen Digital Twin is immediately active worldwide at `https://<your-app>.vercel.app` with complete interactive client-side physics, Chart.js telemetry, and scenario disruption triggers!

---

### Option 2: Streamlit Operations Center (`app.py`)
A comprehensive, tabbed scientific operations dashboard built for deep analytical walkthroughs:
*   **Tab 1 (Live Operations & Pipeline)**: Multi-series Altair plots of Stage A and Stage B queue dynamics.
*   **Tab 2 (5-Minute Probabilistic Forecast)**: Future backlog histograms and food category latency breakdown.
*   **Tab 3 (Cross-Stage Disruption Propagation)**: Run on-demand paired counterfactual simulations demonstrating upstream billing slowdown cascading into downstream kitchen starvation.
*   **Tab 4 (Uncertainty-Aware Decision Optimizer)**: Mathematical cost function evaluation across candidate policies.
*   **Tab 5 (Quantitative Benchmark & Proof of Performance)**: One-click comparative benchmark execution against Baselines A, B, and C.

```bash
# Launch Streamlit interactive dashboard (opens http://localhost:8501)
streamlit run app.py
```

---

## 👥 3-Person Team Module Split

QueueFlux is architected into three coordinated technical ownership areas:

### 👤 Person 1: Queue Simulator & Foundation
*   [`queueflux/simulator/engine.py`](file:///E:/queue-intelligence/queueflux/simulator/engine.py): High-fidelity Discrete-Event Simulator Engine strictly enforcing physical queue conservation laws $Q_{t+1}^B = \max(0, Q_t^B + A_t - S_t^B)$ and $Q_{t+1}^K = \max(0, Q_t^K + E_t - S_t^K)$.
*   [`queueflux/simulator/generator.py`](file:///E:/queue-intelligence/queueflux/simulator/generator.py): Tonic Synthetic Data Generator generating realistic 30-day canteen log datasets (`data/synthetic_canteen_30d.csv`).
*   [`queueflux/models/entities.py`](file:///E:/queue-intelligence/queueflux/models/entities.py) & [`queueflux/models/state.py`](file:///E:/queue-intelligence/queueflux/models/state.py): Formal schemas for `Customer`, `Order`, `Resource`, `SystemState`, and `ObservationVector`.
*   [`run_demo.py`](file:///E:/queue-intelligence/run_demo.py): Standalone interactive demonstration of simulator transitions under disruptions.
*   [`visualize_results.py`](file:///E:/queue-intelligence/visualize_results.py): Matplotlib figure generator for state estimation, fan charts, and propagation shocks.
*   [`tests/test_queueflux.py`](file:///E:/queue-intelligence/tests/test_queueflux.py): 6 unit tests verifying conservation laws, zero-leakage state separation, and workload consistency.

### 👤 Person 2: Core Model (State Estimation & Prediction)
*   [`workload_model.py`](file:///E:/queue-intelligence/workload_model.py): **Workload in expected service-minutes** using lognormal/gamma distributions and renewal-theory residual service times: $\mathbb{E}[R] = \frac{\mu}{2}(1 + C_v^2)$.
*   [`state_estimator.py`](file:///E:/queue-intelligence/state_estimator.py): **Bayesian Particle Filter** $P(X_t \mid Y_{1:t})$ producing shared queue states and 90% credible intervals.
*   [`wait_predictor.py`](file:///E:/queue-intelligence/wait_predictor.py): **Workload-Aware Waiting Time Predictor** calculating true registration-to-collection latency $\hat{T}_{\text{total}} = \hat{T}_{\text{wait}}^B + S^B + \hat{T}_{\text{wait}}^K + S^K(\text{type})$.
*   [`forecast_engine.py`](file:///E:/queue-intelligence/forecast_engine.py): **5-Minute Probabilistic Forecaster & Callable Interface** `predict_queue_state(...)` with Monte Carlo forward trajectory rollouts.
*   [`test_person2.py`](file:///E:/queue-intelligence/test_person2.py): 12 dedicated unit tests verifying Person 2 modules.

### 👤 Person 3: Intelligence Layer (Disruption & Decisions)
*   [`disruption_detector.py`](file:///E:/queue-intelligence/disruption_detector.py): CUSUM / EWMA statistical anomaly detector.
*   [`propagation_analyzer.py`](file:///E:/queue-intelligence/propagation_analyzer.py): Cross-stage ripple analysis with Common Random Numbers.
*   [`decision_optimizer.py`](file:///E:/queue-intelligence/decision_optimizer.py): Evaluates candidate interventions under resource constraints.
*   [`evaluation.py`](file:///E:/queue-intelligence/evaluation.py): Computes precision, recall, F1, MAE, and impact percentage.
*   [`intelligence_engine.py`](file:///E:/queue-intelligence/intelligence_engine.py): Master orchestrator producing executive summaries and JSON payloads.
*   [`integration_adapter.py`](file:///E:/queue-intelligence/integration_adapter.py): Normalizes outputs into standard `QueueFluxState`.

---

## 🔬 Mathematical Formulations

### 1. Mathematical State Representation
At observation interval $t$, the system state is:
$$X_t = \left( Q_t^B, Q_t^K, W_t^B, W_t^K, R_t, C_t, Z_t \right)$$

Where:
*   $Q_t^B, Q_t^K$: Queue lengths waiting for billing and kitchen.
*   $W_t^B, W_t^K$: Total outstanding workloads in expected service-minutes ($\sum_{i \in Q} \hat{s}_i$).
*   $R_t$: Residual service times of active servers and stations.
*   $C_t$: Active service resources $(C_t^B, C_t^K)$.
*   $Z_t$: Operating condition (`NORMAL`, `DEMAND_SURGE`, `BILLING_SLOWDOWN`, `KITCHEN_SLOWDOWN`, `RESOURCE_FAILURE`).

Discrete-event queue transitions:
$$Q_{t+1}^B = \max\left(0, Q_t^B + A_t - S_t^B\right)$$
$$Q_{t+1}^K = \max\left(0, Q_t^K + E_t - S_t^K\right)$$
where $E_t = S_t^B$ (billed orders transfer directly into the kitchen backlog).

### 2. Bayesian Particle Filter
*   **Propagation**:
    $$Q_{t+1}^{B,(i)} = \max\left(0, Q_t^{B,(i)} + \text{Poisson}(\lambda_t \Delta t) - \min(Q_t^{B,(i)}, \text{Poisson}(C^B \mu_B \Delta t))\right)$$
*   **Measurement Update**:
    $$w_t^{(i)} \propto w_{t-1}^{(i)} \cdot \mathcal{N}\left(Y_t^{\text{cam}}; Q_t^{B,(i)}, \sigma_{\text{cam}}^2\right) \cdot \operatorname{Binomial}\left(y_{\text{obs}}^B; S_t^{B,(i)}, p_{\text{detect}}\right)$$
*   **Resampling**: Systematic resampling triggered when $N_{\text{eff}} = \frac{1}{\sum (w^{(i)})^2} < \frac{N}{2}$.

### 3. Workload Waiting Time Model
$$\hat{T}_{\text{total}} = \hat{T}_{\text{wait}}^B + S^B + \hat{T}_{\text{wait}}^K + S^K(\text{category})$$
$$\hat{T}_{\text{wait}}^B \approx \frac{W_{\text{ahead}}^B + \bar{R}^B \cdot C^B}{C^B}, \quad \hat{T}_{\text{wait}}^K \approx \frac{W_{\text{ahead}}^K + \bar{R}^K \cdot C^K}{C^K}$$

### 4. Cross-Stage Disruption Propagation
$$\Delta Q_{t+h}^j = Q_{t+h}^{j, \text{disrupted}} - Q_{t+h}^{j, \text{reference}}, \quad \Delta T_{t+h} = T_{t+h}^{\text{disrupted}} - T_{t+h}^{\text{reference}}$$
Captures the phenomenon where an upstream billing slowdown starves the kitchen ($\Delta Q^K < 0$), followed by a delayed surge and elongated customer collection latencies.

### 5. Uncertainty-Aware Decision Objective
$$\min_{a \in \mathcal{A}} \mathbb{E}[J(a, X)], \quad J(a, X) = \alpha \bar{T}(a, X) + \beta B_{\text{congestion}}(a, X) + \gamma C(a)$$
$$\text{Decision Stability}: \quad R(a^*) = P\left(a^* = \arg\min_{a' \in \mathcal{A}} J(a', X)\right) \quad \text{over posterior particles}$$

---

## ⚡ Quick Start & CLI Usage

### 1. Run Interactive Web UI or Streamlit Dashboard
```bash
# Option 1: Launch FastAPI Control Tower (http://localhost:8000)
python api.py

# Option 2: Launch Streamlit Operations Center (http://localhost:8501)
streamlit run app.py
```

### 2. Run Person 3 Six-Scenario CLI Demonstration
```bash
python main.py
```

### 3. Run Standalone Integrated Demonstration
```bash
python queue_intelligence.py --demo --duration 45
```

### 4. Run Quantitative Benchmark
```bash
python queue_intelligence.py --benchmark --duration 60
```

### 5. Run Cross-Stage Propagation Experiment
```bash
python queue_intelligence.py --propagation
```

### 6. Run Comprehensive Unit Test Suites
```bash
# Run all 34 tests across the entire repository
pytest -v

# Run Person 1 simulator tests (6 tests)
pytest tests/test_queueflux.py -v

# Run Person 2 core model tests (12 tests)
pytest test_person2.py -v

# Run Person 3 intelligence layer tests (16 tests)
python -m unittest test_intelligence.py -v
```

---

## 📊 Experimental Results Summary

| Model Configuration | Queue Length MAE | Queue Length RMSE | 90% CI Coverage |
| :--- | :---: | :---: | :---: |
| **Baseline A (Naive Count / FIFO)** | 0.83 | 1.45 | N/A |
| **Baseline B (Workload without Filtering)** | 0.76 | 1.14 | N/A |
| **Baseline C (Independent Stages)** | 1.33 | 2.07 | N/A |
| **QueueFlux (Full Model)** | **1.16** | **1.80** | **76.7%** |

*   **Total Customer Waiting-Time Reduction Achieved**: **~24.6%**
*   **Decision Stability Under Posterior Uncertainty**: **70% – 95%** consensus across particles.

---

## 📁 Repository Structure

```
queue-intelligence/
├── queueflux/               # [Person 1] Modular QueueFlux Package
│   ├── simulator/           # Discrete-event engine (engine.py) & synthetic data generator (generator.py)
│   ├── models/              # Entity models (entities.py) and state representations (state.py)
│   ├── estimator/           # Queue state estimator implementation (state_estimator.py)
│   ├── predictor/           # Waiting time predictor & forecaster
│   ├── propagation/         # Cross-stage disruption propagation engine
│   ├── optimizer/           # Resource intervention optimizer
│   ├── evaluation/          # Benchmark suite
│   └── config.py            # Global system parameters and default constants
├── data/                    # [Person 1] 30-Day Synthetic Dataset & Publication Figures
│   ├── synthetic_canteen_30d.csv      # 30-day synthetic customer arrival & queue trace (3.4 MB)
│   ├── disruption_events_30d.json     # Chronological disruption events registry
│   ├── tonic_metadata_30d.json        # Data generation metadata
│   ├── benchmark_results_summary.csv  # Baseline evaluation records
│   └── figure1-4_*.png                # State estimation, forecast, propagation & benchmark plots
├── frontend/
│   └── index.html           # Modern responsive Canteen Digital Twin UI (Tailwind + Chart.js)
├── api.py                   # FastAPI REST backend serving state, simulation steps & static UI
├── app.py                   # Comprehensive Streamlit Control Tower & interactive dashboard
├── run_demo.py              # [Person 1] Interactive 12-hour canteen simulation demonstration
├── visualize_results.py     # [Person 1] Performance visualization and figure generation script
├── tests/
│   └── test_queueflux.py    # [Person 1] 6 unit tests verifying conservation laws & state separation
├── workload_model.py        # [Person 2] Service distributions & renewal residual workload
├── state_estimator.py       # [Person 2] Bayesian Particle Filter & shared state producer
├── wait_predictor.py        # [Person 2] Registration-to-collection waiting-time predictor
├── forecast_engine.py       # [Person 2] 5-min probabilistic forecaster & unified callable
├── test_person2.py          # [Person 2] 12 unit tests verifying Person 2 modules & interfaces
├── disruption_detector.py   # [Person 3] Statistical disruption & anomaly detector
├── propagation_analyzer.py  # [Person 3] Cross-stage ripple analysis with Common Random Numbers
├── decision_optimizer.py    # [Person 3] Resource-constrained intervention evaluator
├── evaluation.py            # [Person 3] Quantitative evaluation metrics (F1, MAE, impact %)
├── integration_adapter.py   # [Person 3] Normalizes heterogeneous teammate inputs to QueueFluxState
├── intelligence_engine.py   # [Person 3] Master orchestrator for executive reports and UI JSON
├── main.py                  # [Person 3] 6-scenario end-to-end integration demonstration
├── queue_intelligence.py    # Complete integrated QueueFlux engine & benchmark
├── test_intelligence.py     # Verification suite for physical queue balance invariants
├── sample_data.json         # Canteen system configuration & scenario definitions
├── requirements.txt         # Scientific dependencies, FastAPI, Uvicorn, Streamlit, Altair, Matplotlib
├── README.md                # System documentation, UI user guides & mathematical models
└── .vscode/                 # VS Code launch profiles & tasks for API, Streamlit, and tests
```
