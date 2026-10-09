# QueueFlux — Intelligence Layer & Module Integration

**QueueFlux** is an intelligent canteen queue management and decision-support system built for high-density campus dining facilities. It predicts queue conditions, detects operational disruptions, analyzes cross-stage disruption propagation, recommends optimal resource-constrained interventions, and evaluates operational metrics against baselines.

---

## 1. System Architecture

The canteen workflow is modeled as a 2-stage queuing system:
$$\text{Customer Arrival } (A_t) \longrightarrow \text{Billing Counter } (Q_t^B) \longrightarrow \text{Kitchen Workflow } (Q_t^K) \longrightarrow \text{Food Collection}$$

```
                                  +------------------------------------+
                                  | Person 1: Queue Simulator          |
                                  | Person 2: State Estimator          |
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

## 2. Directory Structure & File Roles

- `integration_adapter.py`: Defines standard `QueueFluxState` dataclass and `IntegrationAdapter` to parse heterogeneous output formats from teammates.
- `disruption_detector.py`: Configurable anomaly detection engine for arrival surges, billing slowdowns, kitchen slowdowns, and capacity deficits with false-positive hysteresis.
- `propagation_analyzer.py`: Evaluates cross-stage ripple effects (e.g. billing slowdowns causing downstream order starvation followed by a delayed kitchen surge).
- `decision_optimizer.py`: Evaluates candidate interventions (`OPEN_EXTRA_BILLING_COUNTER`, `ALLOCATE_WORKER_TO_KITCHEN`, `REALLOCATE_COUNTER_TO_KITCHEN`, `PREPARE_RUSH_STAGING`, `NO_ACTION`) under strict resource constraints (`available_extra_workers`, `available_extra_counters`).
- `evaluation.py`: Computes Precision, Recall, F1-Score, Detection Delay, MAE, RMSE, MAPE, and baseline intervention impact %.
- `intelligence_engine.py`: Master orchestrator producing dual outputs: machine-readable JSON dictionary payloads for UI dashboards and human-readable executive summaries for canteen management.
- `main.py`: Complete CLI demonstration running 6 real-world canteen operational scenarios.
- `test_intelligence.py`: Comprehensive `unittest` test suite covering all modules.

---

## 3. Mathematical State Formulation

At observation interval $t$, the system state is represented by:
$$X_t = \left( Q_t^B, Q_t^K, W_t^B, W_t^K, R_t, C_t, Z_t \right)$$

- $Q_t^B$: Number of customers waiting in the billing queue.
- $Q_t^K$: Number of billed orders waiting in the kitchen queue.
- $A_t$: Customer arrival rate (customers/minute).
- $S_t^B$: Billing service capacity ($C^B \times \frac{60}{\tau^B}$).
- $S_t^K$: Kitchen preparation capacity ($C^K \times \frac{60}{\tau^K}$).

Queue balance transition equations:
$$Q_{t+1}^B = \max\left(0, Q_t^B + A_t - S_t^B\right)$$
$$Q_{t+1}^K = \max\left(0, Q_t^K + E_t - S_t^K\right)$$
*(where $E_t$ is the rate of billed orders entering the kitchen).*

---

## 4. Standard Input Interface (`QueueFluxState`)

| Field Name | Type | Unit | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | `str` | ISO-8601 | Now | Observation timestamp |
| `billing_queue` ($Q_t^B$) | `int` | Customers | `0` | Customers waiting for billing |
| `kitchen_queue` ($Q_t^K$) | `int` | Orders | `0` | Billed orders waiting in kitchen queue |
| `arrivals_last_min` ($A_t$) | `float` | Cust/min | `5.0` | Customer arrival rate in past 1 minute |
| `normal_arrivals_per_min` ($\lambda_0$) | `float` | Cust/min | `5.0` | Expected baseline arrival rate |
| `billing_service_time_sec` ($\tau^B$) | `float` | Seconds | `30.0` | Observed average billing duration |
| `normal_billing_service_time_sec` ($\tau_0^B$)| `float` | Seconds | `30.0` | Normal baseline billing duration |
| `kitchen_service_time_sec` ($\tau^K$) | `float` | Seconds | `120.0` | Observed average kitchen prep duration |
| `normal_kitchen_service_time_sec` ($\tau_0^K$)| `float` | Seconds | `120.0` | Normal baseline kitchen prep duration |
| `active_billing_counters` ($C^B$) | `int` | Counters | `2` | Open billing counters |
| `active_kitchen_servers` ($C^K$) | `int` | Servers | `3` | Active kitchen prep stations/servers |
| `available_extra_counters` | `int` | Counters | `1` | Spare billing counters available |
| `available_extra_workers` | `int` | Staff | `2` | Spare floater workers available |
| `forecast_billing_queue` | `List[float]` | Customers | `[0]*5` | 5-minute forecast of billing queue |
| `forecast_kitchen_queue` | `List[float]` | Orders | `[0]*5` | 5-minute forecast of kitchen queue |
| `forecast_wait_minutes` | `float` | Minutes | Auto | Projected wait time for new arrival |
| `uncertainty` | `float` | `0.0 - 1.0`| `0.0` | Data uncertainty / noise penalty |

---

## 5. Teammate Integration Guide (Person 1 & Person 2)

### How Person 1 (Queue Simulator) connects:
Person 1 generates simulated event outputs. Simply return a dictionary with any of the following key names:
```python
simulator_output = {
    "q_billing": 12,                  # or "billing_queue"
    "q_kitchen": 5,                   # or "kitchen_queue"
    "arrival_rate": 14.0,             # or "arrivals_last_min"
    "billing_service_time": 30.0,     # in seconds
    "kitchen_service_time": 120.0,    # in seconds
}
```

### How Person 2 (State Estimator & Predictor) connects:
Person 2 provides forecast vectors and estimated resource state:
```python
predictor_output = {
    "active_billing_counters": 2,
    "active_kitchen_servers": 3,
    "pred_billing_q": [14.0, 16.0, 18.0, 15.0, 12.0],
    "pred_kitchen_q": [5.0, 8.0, 12.0, 14.0, 10.0],
    "uncertainty": 0.10
}
```

### Running the pipeline with teammate outputs:
```python
from intelligence_engine import IntelligenceEngine

# Instantiate Intelligence Engine
engine = IntelligenceEngine()

# Pass both dictionary outputs directly into process_raw_input
report = engine.process_raw_input(
    person1_data=simulator_output,
    person2_data=predictor_output
)

# 1. Print Human Summary for Management:
print(report.to_human_summary())

# 2. Get JSON Payload for Dashboard / Teammate Frontend:
dashboard_json = report.to_dict()
```

---

## 6. How to Run the System

### Run End-to-End Demonstration (6 Scenarios):
```powershell
python main.py
```

### Run Full Unit & Integration Test Suite:
```powershell
python -m unittest test_intelligence.py -v
```
