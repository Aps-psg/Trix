# ⚡ QueueFlux: Main System & Integration Pipeline

> **Branch: `main-files`**  
> Technical Ownership: **System Orchestration & Integration**  
> This branch isolates the primary execution drivers: the 6-scenario end-to-end integration demonstrator (`main.py`), the discrete-event canteen simulation and benchmark engine (`queue_intelligence.py`), the universal input adapter (`integration_adapter.py`), and system scenario definitions (`sample_data.json`).

---

## 🏛️ Module Overview

```
main-files/
├── main.py                  # End-to-end demonstration running 6 real-world canteen operational scenarios
├── queue_intelligence.py    # Complete discrete-event simulation, particle filter & benchmark engine
├── integration_adapter.py   # Normalizes heterogeneous teammate inputs to standard QueueFluxState
├── sample_data.json         # Canteen system configuration & operational scenario definitions
├── push_to_github.py        # Automated GitHub repository synchronization utility
├── test_intelligence.py     # 16 unit tests verifying adapter, disruptions, and decisions
├── intelligence_engine.py   # Master engine orchestrating executive reports and telemetry
├── disruption_detector.py   # Statistical anomaly detection
├── propagation_analyzer.py  # Cross-stage ripple analysis
├── decision_optimizer.py    # Resource-constrained operational interventions
├── evaluation.py            # Quantitative evaluation metrics
├── requirements.txt         # Core dependencies
├── README.md                # Dedicated Main System documentation
└── .gitignore               # Standard ignore patterns
```

---

## 🚀 Execution & Demonstrations

### 1. Run 6-Scenario Operational Demonstration (`main.py`)
Executes an end-to-end walkthrough covering:
1. **Normal Canteen Operation**: Baseline flow under steady customer arrivals.
2. **Sudden Arrival Surge**: Lunch-hour demand spikes testing queue resilience.
3. **Billing Counter Slowdown**: Upstream POS terminal delays.
4. **Kitchen Preparation Slowdown**: Downstream chef bottlenecks.
5. **Resource-Constrained Surge**: High load under severe counter/staff scarcity.
6. **High Data Uncertainty**: Operation under noisy and incomplete telemetry.

```bash
python main.py
```

---

### 2. Run Integrated Discrete-Event Simulation & Benchmarks (`queue_intelligence.py`)

#### Interactive Live Rush-Hour Simulation:
```bash
python queue_intelligence.py --demo --duration 45
```

#### Quantitative Benchmark Against Baselines A, B, and C:
```bash
python queue_intelligence.py --benchmark --duration 60
```

#### Cross-Stage Disruption Propagation Experiment:
```bash
python queue_intelligence.py --propagation
```

---

## 🧪 Test Verification

Run the complete test suite:
```bash
python -m unittest test_intelligence.py -v
```
