"""QueueFlux — FastAPI Backend & Web Engine
==========================================
Exposes the mathematical models, discrete-event simulation, particle filter,
wait-time predictor, 5-minute forecaster, and decision optimizer via REST API.
Serves the modern web frontend at http://localhost:8000.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from forecast_engine import FiveMinuteProbabilisticForecaster
from queue_intelligence import (
    CanteenEventSimulator,
    CrossStagePropagationAnalyzer,
    GroundTruthState,
    OperatingCondition,
    OrderCategory,
    ParticleFilterEstimator,
    ProbabilisticForecaster,
    QueueFluxBenchmark,
    QueueFluxDecisionOptimizer,
    StatisticalDisruptionDetector,
    WorkloadWaitPredictor,
)
from workload_model import OrderType, WorkloadModel

app = FastAPI(
    title="QueueFlux Control Tower API",
    description="Mathematical Queue Intelligence API for State Estimation, Forecasting, and Optimization",
    version="2.0.0",
)

# Enable CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# RUNTIME SIMULATION STATE
# ============================================================================
class SimulationEngineState:
    def __init__(self) -> None:
        self.reset()

    def reset(self, num_billing: int = 2, num_kitchen: int = 3) -> None:
        self.current_minute = 0
        self.simulator = CanteenEventSimulator(
            num_billing_counters=num_billing,
            num_kitchen_resources=num_kitchen,
            seed=int(np.random.default_rng().integers(1, 10000)),
        )
        self.pf_estimator = ParticleFilterEstimator(num_particles=120, seed=42)
        self.predictor = WorkloadWaitPredictor()
        self.forecaster = ProbabilisticForecaster(num_rollouts=60, seed=42)
        self.detector = StatisticalDisruptionDetector()
        self.optimizer = QueueFluxDecisionOptimizer()
        self.workload_model = WorkloadModel()
        self.history: List[Dict[str, Any]] = []
        self.condition = OperatingCondition.NORMAL
        self.missing_noise = True


state = SimulationEngineState()


# ============================================================================
# API MODELS
# ============================================================================
class StepRequest(BaseModel):
    step_minutes: float = 1.0
    bottleneck_threshold: int = 15


class ConfigRequest(BaseModel):
    num_billing_counters: int
    num_kitchen_resources: int
    condition: Optional[str] = "NORMAL"
    missing_noise: Optional[bool] = True


class DisruptionRequest(BaseModel):
    condition: str  # NORMAL, DEMAND_SURGE, BILLING_SLOWDOWN, KITCHEN_SLOWDOWN, RESOURCE_FAILURE


# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/api/state")
def get_current_state() -> Dict[str, Any]:
    """Retrieve the current ground truth and estimated queue telemetry."""
    gt = state.simulator.get_ground_truth_state()
    latest_hist = state.history[-1] if state.history else None
    return {
        "minute": state.current_minute,
        "ground_truth": {
            "q_billing": gt.q_billing,
            "q_kitchen": gt.q_kitchen,
            "w_billing": gt.w_billing,
            "w_kitchen": gt.w_kitchen,
            "active_billing_counters": gt.active_billing_counters,
            "active_kitchen_resources": gt.active_kitchen_resources,
            "condition": gt.condition.value,
        },
        "latest_telemetry": latest_hist,
        "history_length": len(state.history),
    }


@app.post("/api/step")
def step_simulation(req: StepRequest) -> Dict[str, Any]:
    """Advance simulation by 1 minute, execute Particle Filter update,

    generate 5-min forecast, and optimize decisions.
    """
    state.current_minute += 1
    m = state.current_minute
    sim = state.simulator
    pf = state.pf_estimator

    sim.condition = state.condition

    # Time-dependent arrival rate
    base_lam = 1.0 + 2.5 * np.exp(-((m - 45.0) ** 2) / (2 * (15.0 ** 2)))
    surge_mult = 2.2 if sim.condition == OperatingCondition.DEMAND_SURGE else 1.0
    lam = base_lam * surge_mult
    arrivals = int(np.random.default_rng().poisson(lam))

    cats = [OrderCategory.BEVERAGE, OrderCategory.SNACK, OrderCategory.MEAL, OrderCategory.COMBO]
    probs = [0.35, 0.30, 0.25, 0.10]
    new_orders = []
    for _ in range(arrivals):
        cat_idx = int(np.random.choice(len(cats), p=probs))
        cat = cats[cat_idx]
        arr_t = (m - 1) + float(np.random.uniform(0.0, 0.99))
        o = sim.register_arrival(f"ORD-{m:02d}-{_+1}", cat, arr_t)
        new_orders.append({"id": o.id, "category": cat.value})

    # Step simulator
    b_done, k_done = sim.step_to(float(m))
    gt = sim.get_ground_truth_state()

    # Generate noisy observation
    obs = sim.generate_noisy_observation(gt, b_done, k_done)
    if not state.missing_noise:
        obs.noisy_camera_billing_count = gt.q_billing

    # Particle Filter update
    pf.predict(delta_t=1.0)
    est = pf.update(obs)

    # Predictions
    pred_bev = state.predictor.predict(OrderCategory.BEVERAGE, est, gt.active_billing_counters, gt.active_kitchen_resources)
    pred_snk = state.predictor.predict(OrderCategory.SNACK, est, gt.active_billing_counters, gt.active_kitchen_resources)
    pred_meal = state.predictor.predict(OrderCategory.MEAL, est, gt.active_billing_counters, gt.active_kitchen_resources)
    pred_combo = state.predictor.predict(OrderCategory.COMBO, est, gt.active_billing_counters, gt.active_kitchen_resources)

    # 5-Minute Forecast
    fc = state.forecaster.forecast_5min(
        est,
        gt.active_billing_counters,
        gt.active_kitchen_resources,
        bottleneck_threshold=req.bottleneck_threshold,
    )

    # Disruption Detector & Optimizer
    cond_det, conf = state.detector.inspect(obs, est)
    decision = state.optimizer.evaluate_interventions(est, gt.active_billing_counters, gt.active_kitchen_resources)

    step_data = {
        "minute": m,
        "arrivals_count": arrivals,
        "billing_completions": b_done,
        "kitchen_completions": k_done,
        "ground_truth": {
            "q_billing": gt.q_billing,
            "q_kitchen": gt.q_kitchen,
            "w_billing": gt.w_billing,
            "w_kitchen": gt.w_kitchen,
            "active_billing": gt.active_billing_counters,
            "active_kitchen": gt.active_kitchen_resources,
            "condition": gt.condition.value,
        },
        "estimated": {
            "q_billing": est.expected_q_billing,
            "q_kitchen": est.expected_q_kitchen,
            "q_billing_ci": est.q_billing_ci,
            "q_kitchen_ci": est.q_kitchen_ci,
            "w_billing": est.expected_w_billing,
            "w_kitchen": est.expected_w_kitchen,
            "uncertainty_score": est.uncertainty_score,
            "arrival_rate": est.estimated_arrival_rate,
        },
        "waiting_times": {
            "BEVERAGE": pred_bev["expected_total_wait_minutes"],
            "SNACK": pred_snk["expected_total_wait_minutes"],
            "MEAL": pred_meal["expected_total_wait_minutes"],
            "COMBO": pred_combo["expected_total_wait_minutes"],
            "meal_range": [pred_meal["lower_bound_minutes"], pred_meal["upper_bound_minutes"]],
        },
        "forecast_5min": fc,
        "detection": {
            "condition": cond_det.value,
            "confidence": conf,
        },
        "recommendation": decision,
    }

    state.history.append(step_data)
    return step_data


@app.post("/api/disruption")
def set_disruption(req: DisruptionRequest) -> Dict[str, Any]:
    """Inject a live operational disruption scenario."""
    try:
        cond = OperatingCondition(req.condition.upper())
        state.condition = cond
        state.simulator.condition = cond
        return {"status": "success", "active_condition": cond.value}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid condition: {req.condition}")


@app.post("/api/config")
def update_configuration(req: ConfigRequest) -> Dict[str, Any]:
    """Update active server resources and operational flags."""
    state.simulator.num_billing_counters = max(1, req.num_billing_counters)
    state.simulator.num_kitchen_resources = max(1, req.num_kitchen_resources)
    if req.condition:
        state.condition = OperatingCondition(req.condition.upper())
        state.simulator.condition = state.condition
    if req.missing_noise is not None:
        state.missing_noise = req.missing_noise
    return {
        "status": "success",
        "billing_counters": state.simulator.num_billing_counters,
        "kitchen_resources": state.simulator.num_kitchen_resources,
        "condition": state.condition.value,
        "missing_noise": state.missing_noise,
    }


@app.post("/api/reset")
def reset_simulation() -> Dict[str, Any]:
    """Reset simulation engine state to minute 0."""
    state.reset()
    return {"status": "success", "message": "Simulation reset to t=0"}


@app.get("/api/history")
def get_history() -> List[Dict[str, Any]]:
    """Retrieve full history trace for graphing."""
    return state.history


@app.get("/api/benchmark")
def run_benchmark(duration: int = 30) -> Dict[str, Any]:
    """Execute quantitative benchmark comparing QueueFlux against Baselines A, B, and C."""
    bm = QueueFluxBenchmark()
    return bm.run_full_benchmark(duration_minutes=duration)


@app.get("/api/propagation")
def run_propagation(disruption: str = "BILLING_SLOWDOWN") -> Dict[str, Any]:
    """Run matched paired counterfactual propagation analysis."""
    cond = OperatingCondition(disruption.upper())
    analyzer = CrossStagePropagationAnalyzer()
    gt = state.simulator.get_ground_truth_state()
    return analyzer.analyze_disruption_impact(base_state=gt, disruption=cond, horizon_minutes=15, replications=20)


# ============================================================================
# STATIC FRONTEND SERVING
# ============================================================================
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    def serve_frontend():
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse("<h1>QueueFlux API Running</h1><p>Frontend index.html not found.</p>")


if __name__ == "__main__":
    import uvicorn

    print("Starting QueueFlux API Server on http://localhost:8000 ...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
