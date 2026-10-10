# ⚡ QueueFlux: Discrete-Event Queue Simulator & Synthetic Data Generator

> **Branch: `person1-simulator`**  
> Technical Ownership: **Person 1 (Foundation & Simulation)**  
> This branch isolates the high-fidelity discrete-event canteen simulation engine, 30-day multi-regime synthetic dataset generation, entity models, and publication figure visualizers.

---

## 🏛️ Module Overview

```
person1-simulator/
├── queueflux/               # Core Simulator & Model Architecture
│   ├── simulator/
│   │   ├── engine.py        # Discrete-event engine enforcing queue balance transitions
│   │   └── generator.py     # Tonic Synthetic Data Generator (Poisson arrivals, lognormal services)
│   ├── models/
│   │   ├── entities.py      # Customer, Order, Resource, OrderType, and RegimeType dataclasses
│   │   └── state.py         # SystemState, ObservationVector, and ServiceResidual representations
│   └── config.py            # Global system parameters, service rates, and default capacities
├── data/                    # 30-Day Benchmark Dataset & Figures
│   ├── synthetic_canteen_30d.csv      # Complete 30-day synthetic customer log (3.4 MB)
│   ├── disruption_events_30d.json     # Ground-truth disruption timestamps and regimes
│   ├── tonic_metadata_30d.json        # Data generation metadata
│   ├── benchmark_results_summary.csv  # Baseline evaluation records
│   └── figure1-4_*.png                # State estimation, forecast, propagation & benchmark plots
├── run_demo.py              # Interactive 12-hour simulation demonstration under disruptions
├── visualize_results.py     # Figure generator for state estimation, fan charts, and propagation
├── tests/
│   └── test_queueflux.py    # 6 unit tests verifying conservation laws & state separation
├── requirements.txt         # Core dependencies (numpy, scipy, pandas, matplotlib, pytest)
└── README.md                # Dedicated Person 1 documentation
```

---

## 🔬 Mathematical Formulations & Capabilities

### 1. Discrete-Event Queue Conservation Laws (`queueflux/simulator/engine.py`)
Enforces exact mass balance at every minute step $t$:
$$Q_{t+1}^B = \max\left(0, Q_t^B + A_t - S_t^B\right)$$
$$Q_{t+1}^K = \max\left(0, Q_t^K + E_t - S_t^K\right)$$
where $E_t = S_t^B$ (billed orders immediately enter the kitchen preparation queue).

* **Complete Customer Lifecycle**:
  $$\text{ARRIVED} \longrightarrow \text{WAITING\_BILLING} \longrightarrow \text{BILLING} \longrightarrow \text{WAITING\_KITCHEN} \longrightarrow \text{PREPARING} \longrightarrow \text{READY} \longrightarrow \text{COLLECTED}$$
* **Resource Contention**:
  * Tracks individual billing counters ($C^B$) and kitchen stations ($C^K$).
  * Explicitly calculates residual service times for active servers: $\mathbb{E}[R] = \frac{\mu}{2}(1 + C_v^2)$.

---

### 2. Multi-Regime Synthetic Data Generation (`queueflux/simulator/generator.py`)
Generates realistic canteen operational traces spanning 30 consecutive operational days:
* **Time-Varying Arrival Rates**: Sinusoidal lunch rush peaks modeling campus lunch rushes ($\lambda_{\text{peak}} \approx 18$ cust/min).
* **Multi-Category Order Distributions**:
  * **Beverage**: $\mu = 1.2$ min, $\sigma = 0.35$ min
  * **Snack**: $\mu = 2.5$ min, $\sigma = 0.50$ min
  * **Meal**: $\mu = 5.0$ min, $\sigma = 1.10$ min
  * **Combo**: $\mu = 7.5$ min, $\sigma = 1.60$ min
* **Disruption Events**: Injects realistic anomalies:
  * Sudden arrival surges (flash crowds)
  * Billing counter slowdowns (POS terminal delays)
  * Kitchen preparation bottlenecks
  * Cashier station failures

---

## 🚀 Usage & Execution

### Run Interactive Rush-Hour Demonstration
Runs a full 12-hour (720 minute) canteen simulation day with an injected operational disruption:
```bash
python run_demo.py
```

### Generate Publication-Quality Visualizations
Produces the 4 figures in `data/`:
```bash
python visualize_results.py
```

### Run Dedicated Person 1 Test Suite (6 Tests)
Verifies conservation laws, zero-leakage state separation, and workload consistency:
```bash
pytest tests/test_queueflux.py -v
```
