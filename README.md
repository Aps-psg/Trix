# ⚡ QueueFlux: Core Model (State Estimation & Probabilistic Forecasting)

> **Branch: `core-model`**  
> Technical Ownership: **Person 2 (Core Mathematical Model)**  
> This branch isolates the core queue state estimation, renewal-theory residual workload calculations, registration-to-collection latency predictor, and 5-minute probabilistic forecasting engine.

---

## 🏛️ Module Overview

```
core-model/
├── workload_model.py        # Workload calculation, lognormal service models & renewal residual times
├── state_estimator.py       # Bayesian Particle Filter P(X_t | Y_{1:t}) producing shared queue states
├── wait_predictor.py        # Registration-to-collection latency predictor with prediction intervals
├── forecast_engine.py       # 5-minute probabilistic forecaster & unified callable interface
├── test_person2.py          # 12 unit tests verifying Person 2 modules & shared schema compliance
├── requirements.txt         # Core scientific dependencies (numpy, scipy, pandas, pytest)
├── README.md                # Dedicated mathematical documentation & usage guide
└── .vscode/                 # Debug launch profiles & tasks for Person 2
```

---

## 🔬 Mathematical Formulations

### 1. Workload in Expected Service-Minutes (`workload_model.py`)
Rather than relying on naive customer headcounts, QueueFlux measures queue burden in expected service minutes:
$$W_t = \sum_{i \in Q_t} \hat{s}_i$$

* **Order Service Distributions**: Lognormal and Gamma distributions parameterized for 4 order categories:
  * **Beverage**: $\mu = 1.2$ min, $\sigma = 0.35$ min
  * **Snack**: $\mu = 2.5$ min, $\sigma = 0.50$ min
  * **Meal**: $\mu = 5.0$ min, $\sigma = 1.10$ min
  * **Combo**: $\mu = 7.5$ min, $\sigma = 1.60$ min
* **Renewal Theory Residual Service Time**:
  When observing servers currently busy with an ongoing order, the expected remaining service duration is modeled via renewal theory:
  $$\mathbb{E}[R] = \frac{\mu}{2}\left(1 + C_v^2\right) = \frac{\mu}{2}\left(1 + \frac{\sigma^2}{\mu^2}\right)$$

---

### 2. Bayesian Particle Filter (`state_estimator.py`)
Estimates hidden true queue states $X_t = (Q_t^B, Q_t^K, W_t^B, W_t^K)$ under incomplete observations (noisy overhead camera counts and dropped POS packet transmissions):

* **Prior State Propagation**:
  $$Q_{t+1}^{B,(i)} = \max\left(0, Q_t^{B,(i)} + \text{Poisson}(\lambda_t \Delta t) - \min(Q_t^{B,(i)}, \text{Poisson}(C^B \mu_B \Delta t))\right)$$
* **Observation Likelihood Weighting**:
  $$w_t^{(i)} \propto w_{t-1}^{(i)} \cdot \mathcal{N}\left(Y_t^{\text{cam}}; Q_t^{B,(i)}, \sigma_{\text{cam}}^2\right) \cdot \operatorname{Binomial}\left(y_{\text{obs}}^B; S_t^{B,(i)}, p_{\text{detect}}\right)$$
* **Adaptive Resampling**:
  Systematic resampling when effective sample size drops below threshold:
  $$N_{\text{eff}} = \frac{1}{\sum_{i=1}^N (w^{(i)})^2} < \frac{N}{2}$$
* **Outputs**: Expected queue counts, 90% credible intervals $[\hat{q}_{0.05}, \hat{q}_{0.95}]$, uncertainty score, and arrival rate estimate in standard `SharedQueueState` format.

---

### 3. Workload-Aware Waiting Time Predictor (`wait_predictor.py`)
Predicts the full registration-to-collection duration ($t_i^{\text{collection}} - t_i^{\text{registration}}$):

$$\hat{T}_{\text{total}} = \hat{T}_{\text{wait}}^B + S^B + \hat{T}_{\text{wait}}^K + S^K(\text{order\_category})$$
$$\hat{T}_{\text{wait}}^B \approx \frac{W_{\text{ahead}}^B + \bar{R}^B \cdot C^B}{C^B}, \quad \hat{T}_{\text{wait}}^K \approx \frac{W_{\text{ahead}}^K + \bar{R}^K \cdot C^K}{C^K}$$

* Computes 10th-90th prediction intervals.
* Evaluates threshold exceedance risk: $P(T > 15\text{ min})$.

---

### 4. 5-Minute Probabilistic Forecaster (`forecast_engine.py`)
Projects future queue evolution over a 5-minute horizon via Monte Carlo forward trajectory rollouts:
* Simulates future stochastic arrivals and service transitions for $t \in [0, 5]$ minutes.
* Produces:
  * Median projected backlog: $Q_{t+5}^B, Q_{t+5}^K$
  * 80% and 90% prediction interval bands
  * Bottleneck exceedance probability: $P(Q_{t+5}^B + Q_{t+5}^K > \text{threshold})$

---

## 🚀 Usage & Execution

### Run Standalone Mock Forecast Engine
Runs 5 simulation steps using synthetic mock event streams:
```bash
python forecast_engine.py --mock --steps 5
```

### Run Dedicated Person 2 Test Suite (12 Tests)
Verifies mathematical bounds, credible interval coverage, monotonicity, and schema compliance:
```bash
pytest test_person2.py -v
```

---

## 🔗 Integration Interface

Other modules (Person 1 simulator or Person 3 decision optimizer) call Person 2 via the unified interface:

```python
from forecast_engine import predict_queue_state

result = predict_queue_state(
    observed_events=event_list,
    noisy_camera_count=12,
    active_billing_counters=2,
    active_kitchen_servers=3,
    target_category="MEAL",
    bottleneck_threshold=15
)

print(result["state_estimate"]["expected_q_billing"])
print(result["wait_prediction"]["expected_total_wait_minutes"])
print(result["forecast_5min"]["bottleneck_probability"])
```
