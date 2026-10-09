# ⚡ QueueFlux: Propagation-Aware, Uncertainty-Aware Queue Prediction and Optimization Model

QueueFlux is an original, mathematically grounded computational framework for estimating queue states, predicting waiting times, forecasting congestion, detecting operational disruptions, and recommending optimal operational interventions in a two-stage service system (College Canteen).

---

## 🎯 Central Objectives & Capabilities

1. **Queue-State Estimation Under Incomplete Observations**: Hidden state filtering using a Bayesian Particle Filter $P(X_t \mid Y_{1:t})$ handling noisy camera feeds and missed POS event transmissions.
2. **Workload-Aware Waiting-Time Prediction**: Calculation based on true pending service workloads in expected service-minutes ($\sum \hat{s}_i$) and server residual times, predicting exact registration-to-collection duration ($t_i^{\text{collection}} - t_i^{\text{registration}}$).
3. **Five-Minute Probabilistic Forecasting**: Forward trajectory rollouts projecting median queue backlogs, 80%/90% prediction intervals, and future bottleneck probabilities ($P(Q_{t+5} > \text{threshold})$).
4. **Cross-Stage Disruption Propagation**: Matched paired counterfactual rollouts using Common Random Numbers (CRN) isolating downstream kitchen starvation ($\Delta Q^K < 0$) and subsequent backlog waves caused by upstream billing slowdowns.
5. **Statistical Disruption Detection**: Sequential CUSUM and EWMA change-point detection identifying arrival surges, service slowdowns, and counter failures.
6. **Uncertainty-Aware Intervention Optimization**: Evaluates candidate actions over posterior particles with robust decision stability statistics $R(a^*) = P(a^* = \arg\min J(a, X))$.
7. **Quantitative Evaluation Against Baselines**: Rigorous experimental comparison against Baseline A (Naive FIFO), Baseline B (Workload without filtering), and Baseline C (Independent stages).

---

## 🏛️ System Scope & Mathematical State Representation

### Two Connected Stages
*   **Stage A — Billing**: Multiple billing counters ($C_t^B$), variable lognormal service duration, counter breakdown support.
*   **Stage B — Food Preparation (Kitchen)**: Multiple preparation stations ($C_t^K$), categorized workloads (Beverage: 1.2 min, Snack: 2.5 min, Meal: 5.0 min, Combo: 7.5 min).

### Complete Order Lifecycle
$$\text{ARRIVED} \longrightarrow \text{WAITING\_BILLING} \longrightarrow \text{BILLING} \longrightarrow \text{WAITING\_KITCHEN} \longrightarrow \text{PREPARING} \longrightarrow \text{READY} \longrightarrow \text{COLLECTED}$$

### Mathematical State
At time $t$, the system state is:
$$X_t = \left(Q_t^B, Q_t^K, W_t^B, W_t^K, R_t, C_t, Z_t\right)$$

Where:
*   $Q_t^B, Q_t^K$: Queue lengths waiting for billing and kitchen.
*   $W_t^B, W_t^K$: Total outstanding workloads in expected service-minutes.
*   $R_t$: Residual service times of currently active servers and preparation stations.
*   $C_t$: Active service resources $(C_t^B, C_t^K)$.
*   $Z_t$: Operating condition (`NORMAL`, `DEMAND_SURGE`, `BILLING_SLOWDOWN`, `KITCHEN_SLOWDOWN`, `RESOURCE_FAILURE`).

### Discrete-Event Queue Transitions
$$Q_{t+1}^B = \max\left(0, Q_t^B + A_t - S_t^B\right)$$
$$Q_{t+1}^K = \max\left(0, Q_t^K + E_t - S_t^K\right)$$
where $E_t = S_t^B$ (billed orders transfer directly into the kitchen backlog).

---

## 🔬 Mathematical Formulations

### 1. Bayesian Particle Filter
*   **Propagation**:
    $$Q_{t+1}^{B,(i)} = \max\left(0, Q_t^{B,(i)} + \text{Poisson}(\lambda_t \Delta t) - \min(Q_t^{B,(i)}, \text{Poisson}(C^B \mu_B \Delta t))\right)$$
*   **Measurement Update**:
    $$w_t^{(i)} \propto w_{t-1}^{(i)} \cdot \mathcal{N}\left(Y_t^{\text{cam}}; Q_t^{B,(i)}, \sigma_{\text{cam}}^2\right) \cdot \operatorname{Binomial}\left(y_{\text{obs}}^B; S_t^{B,(i)}, p_{\text{detect}}\right)$$
*   **Resampling**: Systematic resampling triggered when $N_{\text{eff}} = \frac{1}{\sum (w^{(i)})^2} < \frac{N}{2}$.

### 2. Workload Waiting Time Model
$$\hat{T}_{\text{total}} = \hat{T}_{\text{wait}}^B + S^B + \hat{T}_{\text{wait}}^K + S^K(\text{category})$$
$$\hat{T}_{\text{wait}}^B \approx \frac{W_{\text{ahead}}^B + \bar{R}^B \cdot C^B}{C^B}, \quad \hat{T}_{\text{wait}}^K \approx \frac{W_{\text{ahead}}^K + \bar{R}^K \cdot C^K}{C^K}$$

### 3. Cross-Stage Propagation
$$\Delta Q_{t+h}^j = Q_{t+h}^{j, \text{disrupted}} - Q_{t+h}^{j, \text{reference}}$$
$$\Delta T_{t+h} = T_{t+h}^{\text{disrupted}} - T_{t+h}^{\text{reference}}$$
Captures the phenomenon where an upstream billing slowdown starves the kitchen ($\Delta Q^K < 0$), followed by a delayed surge and elongated customer collection latencies.

### 4. Uncertainty-Aware Decision Objective
$$\min_{a \in \mathcal{A}} \mathbb{E}[J(a, X)], \quad J(a, X) = \alpha \bar{T}(a, X) + \beta B_{\text{congestion}}(a, X) + \gamma C(a)$$
$$\text{Decision Stability}: \quad R(a^*) = P\left(a^* = \arg\min_{a' \in \mathcal{A}} J(a', X)\right) \quad \text{over posterior particles}$$

---

## ⚡ Quick Start & CLI Usage

### 1. Run Interactive Live Demonstration
Runs a simulated canteen lunch rush with mid-session surge, step-by-step state estimation, wait-time prediction, bottleneck alerts, and counterfactual propagation analysis:
```bash
python queue_intelligence.py --demo --duration 45
```

### 2. Run Quantitative Benchmark
Executes paired experimental evaluation of QueueFlux vs Baselines A, B, and C:
```bash
python queue_intelligence.py --benchmark --duration 60
```

### 3. Run Cross-Stage Propagation Experiment
Analyzes paired counterfactual trajectories isolating causal disruption effects:
```bash
python queue_intelligence.py --propagation
```

### 4. Run Comprehensive Unit Test Suite
Executes all 16 mandatory verification tests (invariants, non-negativity, capacity, hand-calculable trace, etc.):
```bash
pytest test_intelligence.py -v
```
or with Python standard library:
```bash
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

---

## 👥 3-Person Team Module Split & Person 2 Core Model

QueueFlux is architected into three independent technical ownership modules:
*   **Person 1 (Foundation)**: Queue Simulator + Data Generation (`data_generator.py`, `event_simulator.py`).
*   **Person 2 (Core Model — Current Focus)**: State Estimation + Prediction (`workload_model.py`, `state_estimator.py`, `wait_predictor.py`, `forecast_engine.py`).
*   **Person 3 (Intelligence Layer)**: Disruption Analysis + Decision Optimization (`disruption_detector.py`, `propagation_analyzer.py`, `decision_optimizer.py`, `evaluation.py`).

### 📦 Person 2 Core Model Breakdown

| Module | Core Responsibility & Formulas |
| :--- | :--- |
| [`workload_model.py`](file:///E:/queue-intelligence/workload_model.py) | **Workload in expected service-minutes**.<br>• Order distributions (Lognormal/Gamma).<br>• Renewal-theory residual service times: $\mathbb{E}[R] = \frac{\mu}{2}(1 + C_v^2)$.<br>• Queue-to-workload translation $\sum_{i \in Q} \hat{s}_i$. |
| [`state_estimator.py`](file:///E:/queue-intelligence/state_estimator.py) | **Bayesian Particle Filter** $P(X_t \mid Y_{1:t})$.<br>• Ingests shared event format and noisy camera counts.<br>• Outputs agreed **shared state format** with 90% credible intervals.<br>• Baseline counter comparison & mock event generator. |
| [`wait_predictor.py`](file:///E:/queue-intelligence/wait_predictor.py) | **Workload-Aware Waiting Time Predictor**.<br>• Registration-to-collection latency: $\hat{T}_{\text{total}} = \hat{T}_{\text{wait}}^B + S^B + \hat{T}_{\text{wait}}^K + S^K(\text{type})$.<br>• 10th-90th prediction intervals and exceedance probability $P(T > 15\text{m})$. |
| [`forecast_engine.py`](file:///E:/queue-intelligence/forecast_engine.py) | **5-Minute Probabilistic Forecaster & Callable Interface**.<br>• Monte Carlo trajectory rollouts over $t \in [0, 5]$ min.<br>• Median backlogs, 80%/90% intervals, bottleneck probability.<br>• Unified callable: `predict_queue_state(observed_events, ...)`. |

### 🚀 Running Person 2 Standalone

Run mock evaluation directly:
```bash
python forecast_engine.py --mock --steps 5
```

Run Person 2 dedicated test suite (12 unit tests):
```bash
pytest test_person2.py -v
```

---

## 📁 Repository Structure

```
queue-intelligence/
├── workload_model.py        # [Person 2] Service distributions & renewal residual workload
├── state_estimator.py       # [Person 2] Bayesian Particle Filter & shared state producer
├── wait_predictor.py        # [Person 2] Registration-to-collection waiting-time predictor
├── forecast_engine.py       # [Person 2] 5-min probabilistic forecaster & unified callable
├── test_person2.py          # [Person 2] 12 unit tests verifying Person 2 modules & interfaces
├── queue_intelligence.py    # Complete integrated QueueFlux engine & benchmark
├── test_intelligence.py     # 16 unit tests verifying full end-to-end Section 14 requirements
├── sample_data.json         # Canteen system configuration & scenario definitions
├── requirements.txt         # Core dependencies (numpy, scipy, pandas, pytest, rich)
├── README.md                # System documentation & mathematical formulations
└── .vscode/                 # VS Code launch & test discovery configurations
```
