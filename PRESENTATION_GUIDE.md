# 🏆 QueueFlux: Project Presentation & Viva Defense Guide

> **Project Title**: QueueFlux — Propagation-Aware, Uncertainty-Aware Queue State Estimation, Prediction, and Optimization Model for High-Density Campus Dining Facilities  
> **Repository**: [https://github.com/Aps-psg/Trix](https://github.com/Aps-psg/Trix)  
> **Target Audience**: Evaluation Committee, Hackathon Judging Panel, Technical Reviewers

---

## ⚡ 1. The 30-Second Elevator Pitch

> *"Most queue management systems rely on naive customer headcounts and FIFO assumptions, which fail in multi-stage dining facilities where orders have wildly variable preparation times. **QueueFlux** is an end-to-end computational digital twin and decision control tower. Using **renewal-theory residual workloads**, a **Bayesian Particle Filter** under noisy camera observations, and **matched paired counterfactuals**, QueueFlux predicts exact registration-to-collection latencies, isolates cross-stage disruptions like downstream kitchen starvation, and optimizes staffing interventions—achieving a **24.6% reduction in customer waiting times**."*

---

## 🎤 2. 5-Minute Team Presentation Script

```
[0:00 - 0:45] INTRODUCTION & THE PROBLEM STATEMENT (Person 1)
-------------------------------------------------------------------------
"Good morning, judges. In campus dining halls, customer lines are frustrating and unpredictable.
Traditional systems measure queue length simply by counting heads. But in reality, a line of 5 people
ordering coffees clears in 2 minutes, while a line of 5 people ordering multi-item combos takes 20 minutes!
Furthermore, dining facilities operate in two tightly coupled physical stages:
Stage A (Billing Counters) and Stage B (Kitchen Preparation).
If Stage A slows down, Stage B starves of orders, creating an illusion of low kitchen load, followed
by an overwhelming surge when billing recovers.
To solve this, we built QueueFlux."

[0:45 - 2:00] PERSON 1: SIMULATION FOUNDATION & SYNTHETIC DATA
-------------------------------------------------------------------------
"I developed the Foundation and Discrete-Event Simulation Engine:
1. Enforces strict physical queue balance equations at every minute:
   Q_{t+1}^B = max(0, Q_t^B + A_t - S_t^B)
   Q_{t+1}^K = max(0, Q_t^K + E_t - S_t^K)
2. Developed the Synthetic Data Generator modeling 30 consecutive operational days, featuring time-varying
   Poisson arrivals, lognormal and gamma service distributions across 4 order categories, and injected
   disruptions like billing slowdowns, kitchen bottlenecks, and counter failures.
3. Modeled real-world sensor imperfections: overhead camera counting noise and intermittent POS packet dropouts."

[2:00 - 3:15] PERSON 2: CORE MATHEMATICAL MODEL & ESTIMATION
-------------------------------------------------------------------------
"I developed the Core Mathematical Engine: State Estimation and Latency Forecasting:
1. Workload Model: Instead of counting customers, we compute total pending workload in expected service-minutes.
   To account for orders currently being prepared, we apply Renewal Theory:
   E[R] = (mu / 2) * (1 + C_v^2), capturing service variance.
2. Bayesian Particle Filter: Ingests noisy camera feeds and missing POS events to track hidden queue states
   with 90% Credible Intervals.
3. Registration-to-Collection Predictor: Projects exact customer wait times across 4 food categories.
4. 5-Minute Probabilistic Forecaster: Uses Monte Carlo forward rollouts to calculate future bottleneck
   exceedance risk: P(Q_{t+5} > Threshold)."

[3:15 - 4:15] PERSON 3: INTELLIGENCE LAYER & DECISION OPTIMIZATION
-------------------------------------------------------------------------
"I developed the Intelligence and Decision Optimization Layer:
1. Disruption Detection: CUSUM and EWMA sequential change-point algorithms with hysteresis to eliminate false alarms.
2. Cross-Stage Disruption Propagation: Using Common Random Numbers (CRN), we run paired counterfactual simulations
   that prove an upstream billing slowdown causes an immediate downstream kitchen starvation (Delta Q^K < 0),
   followed by a delayed surge.
3. Uncertainty-Aware Decision Optimizer: Evaluates candidate actions like opening counters or shifting floaters
   under strict resource constraints, achieving 70% to 95% Bayesian posterior consensus stability.
4. Quantitative Proof: Compared against Baselines A, B, and C, QueueFlux cuts total customer waiting times by 24.6%."

[4:15 - 5:00] LIVE DEMO & CONCLUSION (Shared)
-------------------------------------------------------------------------
"We brought all models together into two live interfaces:
A modern glassmorphic Web Control Twin (FastAPI + Tailwind + Chart.js) deployed on Vercel,
and a 5-tab Streamlit scientific operations center.
All 34 unit tests pass, and the repository is organized into clean modular branches.
Thank you, and we welcome your questions!"
```

---

## 📊 3. Key Quantitative Results

| Performance Metric | QueueFlux Achievement | Baseline Comparison |
| :--- | :---: | :--- |
| **Total Waiting-Time Reduction** | **24.6%** | vs. Baseline A (Naive FIFO) |
| **Queue Length Estimation MAE** | **1.16** | vs. 1.33 for Baseline C (Independent stages) |
| **90% Credible Interval Coverage**| **76.7%** | Under heavy camera noise & POS dropouts |
| **Decision Stability Under Uncertainty** | **70% – 95%** | Consistent action across posterior particles |
| **Automated Test Coverage** | **34 / 34 Tests** | 100% passing across all 3 modules |

---

## 🛡️ 4. Q&A Defense Cheat Sheet (Answers to Tough Viva Questions)

### Q1: Why use Renewal Theory for residual service time $\mathbb{E}[R]$ instead of memoryless exponential distributions?
> **Answer**:  
> *"Food preparation and cashier billing times are strictly non-exponential; human service tasks exhibit a minimum preparation threshold and lognormal variance. Under non-exponential service distributions, memorylessness does not hold. By renewal theory, the forward recurrence time for an ongoing renewal process has expected value $\mathbb{E}[R] = \frac{\mu}{2}(1 + C_v^2)$, where $C_v = \sigma / \mu$ is the coefficient of variation. If service were exponential ($C_v = 1$), $\mathbb{E}[R] = \mu$. But for predictable or variable tasks, $C_v \neq 1$, making renewal residual calculation mathematically necessary for unbiased workload estimation."*

### Q2: How does the Particle Filter prevent sample impoverishment (particle degeneracy)?
> **Answer**:  
> *"We monitor the Effective Sample Size $N_{\text{eff}} = \frac{1}{\sum_{i=1}^N (w^{(i)})^2}$. When $N_{\text{eff}}$ drops below $N/2$, systematic resampling is triggered. Furthermore, during state propagation, we inject stochastic process diffusion variance ($\sigma_{\text{diff}}$) reflecting arrival and service rate uncertainty, ensuring particle diversity is maintained without collapsing onto a single state."*

### Q3: What is the purpose of Common Random Numbers (CRN) in disruption propagation?
> **Answer**:  
> *"To prove causality when comparing a normal operational trajectory against a disrupted trajectory, we must eliminate variance caused by differing random arrival sequences. Common Random Numbers synchronize identical pseudo-random seeds between the reference and counterfactual rollouts. This isolates the exact delta $\Delta Q_{t+h} = Q_{t+h}^{\text{disrupted}} - Q_{t+h}^{\text{reference}}$, proving that kitchen starvation was directly caused by the upstream billing slowdown rather than random arrival fluctuations."*

### Q4: How does the decision optimizer balance conflicting objectives?
> **Answer**:  
> *"We evaluate a scalarized multi-attribute objective:  
> $$J(a, X) = \alpha \bar{T}(a, X) + \beta B_{\text{congestion}}(a, X) + \gamma C_{\text{staff}}(a)$$  
> $\alpha$ weights student waiting time, $\beta$ penalizes server queue saturation above safety limits, and $\gamma$ accounts for labor costs associated with deploying spare counters or calling floater staff. Interventions are evaluated across all posterior particles to measure Bayesian stability $R(a^*)$ before recommending action to management."*

---

## 🖥️ 5. Live Demonstration Commands

| Interface | Launch Command | Target URL |
| :--- | :--- | :--- |
| **Modern Web Digital Twin** | Double-click `run_web_ui.bat` or `python api.py` | `http://localhost:8000` |
| **Streamlit Operations Center** | Double-click `run_streamlit.bat` or `streamlit run app.py` | `http://localhost:8501` |
| **12-Hour Simulation Demo** | Double-click `run_demo.bat` or `python run_demo.py` | Terminal Output |
| **Run All 34 Tests** | Double-click `run_tests.bat` or `pytest -v` | All 34 Tests PASS |
