# ⚡ QueueFlux: Intelligence Layer (Disruptions, Propagation & Decisions)

> **Branch: `intelligence-layer`**  
> Technical Ownership: **Person 3 (Intelligence Layer)**  
> This branch isolates statistical disruption detection, cross-stage ripple propagation analysis, resource-constrained decision optimization, performance evaluation metrics, and the master intelligence orchestration engine.

---

## 🏛️ Module Overview

```
intelligence-layer/
├── disruption_detector.py   # Anomaly detection (CUSUM, EWMA, thresholding) with hysteresis
├── propagation_analyzer.py  # Cross-stage ripple analysis (billing slowdown -> kitchen starvation)
├── decision_optimizer.py    # Resource-constrained operational intervention recommendations
├── evaluation.py            # Quantitative metrics (F1, Precision, Recall, MAE, RMSE, Impact %)
├── intelligence_engine.py   # Master orchestrator producing executive reports and UI JSON
├── integration_adapter.py   # Adapts heterogeneous teammate data into standard QueueFluxState
├── test_intelligence.py     # 16 comprehensive unit & integration tests
├── extension.js             # Visual panel integration script
├── package.json             # NPM package specification
├── requirements.txt         # Dependencies (numpy, scipy, pandas, pytest)
├── README.md                # Dedicated Intelligence Layer documentation
└── .gitignore               # Standard ignore patterns
```

---

## 🔬 Mathematical Formulations & Capabilities

### 1. Statistical Disruption Detection (`disruption_detector.py`)
Monitors real-time operational streams to detect anomalies with false-positive suppression:
* **Arrival Surges**: Detects sudden load bursts exceeding threshold $\lambda_t > \kappa_{\text{surge}} \cdot \lambda_0$.
* **Billing Counter Slowdowns**: Identifies POS terminal delays or cashier bottlenecking ($\tau^B > 1.4 \times \tau_0^B$).
* **Kitchen Preparation Delays**: Tracks chef backlog and station preparation delays ($\tau^K > 1.3 \times \tau_0^K$).
* **Capacity Deficits**: Detects when demand outstrips total physical service capacity: $A_t > S_t^B$ or $S_t^B > S_t^K$.
* **Hysteresis & Smoothing**: Requires persistent anomaly signals across multiple intervals before raising alert severity to prevent false alarm thrashing.

---

### 2. Cross-Stage Disruption Propagation (`propagation_analyzer.py`)
Evaluates how upstream disruptions propagate downstream through the physical 2-stage canteen pipeline:
$$\Delta Q_{t+h}^j = Q_{t+h}^{j, \text{disrupted}} - Q_{t+h}^{j, \text{reference}}$$

* **Kitchen Starvation Phenomenon**: An upstream billing slowdown prevents completed tickets from reaching the kitchen, causing an immediate dip in kitchen utilization ($\Delta Q^K < 0$).
* **Delayed Recovery Wave**: When billing recovers, the accumulated backlog discharges as an overwhelming wave into the kitchen, creating prolonged customer latency spikes.

---

### 3. Resource-Constrained Decision Optimizer (`decision_optimizer.py`)
Evaluates candidate operational interventions subject to strict real-world physical constraints:
* Candidate Actions:
  * `OPEN_EXTRA_BILLING_COUNTER`
  * `ALLOCATE_WORKER_TO_KITCHEN`
  * `REALLOCATE_COUNTER_TO_KITCHEN`
  * `PREPARE_RUSH_STAGING`
  * `NO_ACTION`
* Constraint Verification: Respects `available_extra_counters` and `available_extra_workers`.
* Optimization Objective:
  $$\min_{a \in \mathcal{A}_{\text{feasible}}} J(a, X) = \alpha \bar{T}(a, X) + \beta B_{\text{congestion}}(a, X) + \gamma C_{\text{staff}}(a)$$

---

### 4. Quantitative Evaluation Metrics (`evaluation.py`)
Computes empirical performance metrics across validation traces:
* **Detection Accuracy**: Precision, Recall, F1-Score, Detection Delay (latency in minutes from disruption onset to alert).
* **Estimation Error**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE).
* **Intervention Efficacy**: Percentage reduction in expected customer waiting times:
  $$\text{Impact} = \frac{\bar{T}_{\text{baseline}} - \bar{T}_{\text{intervention}}}{\bar{T}_{\text{baseline}}} \times 100\%$$

---

## 🚀 Usage & Execution

### Run Comprehensive Intelligence Test Suite (16 Tests)
Verifies input adaptation, disruption detection, propagation analysis, constraint enforcement, and evaluation metrics:
```bash
python -m unittest test_intelligence.py -v
```
or with pytest:
```bash
pytest test_intelligence.py -v
```

---

## 🔗 Integration Interface (`QueueFluxState`)

The Intelligence Layer accepts data from Person 1 (Simulator) and Person 2 (Estimator) via `integration_adapter.py`:

```python
from intelligence_engine import IntelligenceEngine

engine = IntelligenceEngine()

# Ingest data from teammates
report = engine.process_raw_input(
    person1_data={"billing_queue": 14, "kitchen_queue": 6, "arrivals_last_min": 15.0},
    person2_data={"active_billing_counters": 2, "active_kitchen_servers": 3, "forecast_wait_minutes": 12.5}
)

# 1. Output Human-Readable Summary
print(report.to_human_summary())

# 2. Output Machine-Readable JSON for Web Frontends
payload = report.to_dict()
```
