"""
server.py - QueueFlux Production FastAPI Backend Server

Provides RESTful API endpoints and WebSockets for real-time canteen queue monitoring,
state estimation, cross-stage disruption propagation analysis, decision optimization,
and live simulation control.
Authored by: sbsureka (https://github.com/sbsureka)
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import asyncio
import json
import datetime

# Import QueueFlux Modules from Aps-psg/Trix codebase
from integration_adapter import IntegrationAdapter, QueueFluxState
from intelligence_engine import IntelligenceEngine, DecisionReport


# ==============================================================================
# FASTAPI APP INITIALIZATION & CORS SETUP
# ==============================================================================
app = FastAPI(
    title="QueueFlux Intelligence Backend API",
    description="Backend API for QueueFlux - Propagation-Aware, Uncertainty-Aware Canteen Queue Prediction & Optimization",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# SCENARIO CONFIGURATIONS & SIMULATION MANAGER
# ==============================================================================
SCENARIO_CONFIGS = {
    "normal": {
        "arrival_rate_per_min": 4.0,
        "billing_service_sec": 28.0,
        "kitchen_service_sec": 110.0,
        "active_billing_counters": 2,
        "active_kitchen_servers": 8,
        "extra_counters": 1,
        "extra_workers": 2,
        "uncertainty": 0.05
    },
    "arrival_surge": {
        "arrival_rate_per_min": 14.0,
        "billing_service_sec": 30.0,
        "kitchen_service_sec": 120.0,
        "active_billing_counters": 2,
        "active_kitchen_servers": 8,
        "extra_counters": 1,
        "extra_workers": 2,
        "uncertainty": 0.10
    },
    "billing_slowdown": {
        "arrival_rate_per_min": 5.0,
        "billing_service_sec": 75.0,
        "kitchen_service_sec": 120.0,
        "active_billing_counters": 2,
        "active_kitchen_servers": 8,
        "extra_counters": 1,
        "extra_workers": 2,
        "uncertainty": 0.10
    },
    "kitchen_slowdown": {
        "arrival_rate_per_min": 5.0,
        "billing_service_sec": 30.0,
        "kitchen_service_sec": 240.0,
        "active_billing_counters": 2,
        "active_kitchen_servers": 8,
        "extra_counters": 1,
        "extra_workers": 2,
        "uncertainty": 0.15
    },
    "constrained_surge": {
        "arrival_rate_per_min": 12.0,
        "billing_service_sec": 30.0,
        "kitchen_service_sec": 120.0,
        "active_billing_counters": 2,
        "active_kitchen_servers": 8,
        "extra_counters": 0,
        "extra_workers": 0,
        "uncertainty": 0.10
    },
    "high_uncertainty": {
        "arrival_rate_per_min": 9.0,
        "billing_service_sec": 45.0,
        "kitchen_service_sec": 150.0,
        "active_billing_counters": 2,
        "active_kitchen_servers": 6,
        "extra_counters": 1,
        "extra_workers": 1,
        "uncertainty": 0.65
    }
}


class SimulationManager:
    """Manages global live simulation state and connected WebSocket clients."""

    def __init__(self):
        self.scenario_name = "normal"
        self.config = SCENARIO_CONFIGS["normal"].copy()
        self.sim_time_minutes = 0
        self.billing_queue = 2
        self.kitchen_queue = 3
        self.intelligence_engine = IntelligenceEngine()
        self.history: List[Dict[str, Any]] = []
        self.active_websockets: List[WebSocket] = []
        self.step_simulation()

    def calculate_5min_forecast(self, b_q: float, k_q: float, arr_rate: float, b_sec: float, k_sec: float, c_b: int, c_k: int):
        """Calculates 5-minute queue trajectory forecast list."""
        b_cap = (60.0 / max(1.0, b_sec)) * max(1, c_b)
        k_cap = (60.0 / max(1.0, k_sec)) * max(1, c_k)

        cur_b, cur_k = float(b_q), float(k_q)
        fc_b, fc_k = [], []

        for _ in range(5):
            b_arr = arr_rate
            b_done = min(cur_b + b_arr, b_cap)
            cur_b = max(0.0, cur_b + b_arr - b_done)

            k_arr = b_done
            k_done = min(cur_k + k_arr, k_cap)
            cur_k = max(0.0, cur_k + k_arr - k_done)

            fc_b.append(round(cur_b, 1))
            fc_k.append(round(cur_k, 1))

        return fc_b, fc_k

    def step_simulation(self, duration_min: int = 1) -> Dict[str, Any]:
        """Steps simulator, estimates state, runs 5min forecast & intelligence engine."""
        self.sim_time_minutes += duration_min
        arr_rate = self.config["arrival_rate_per_min"]
        b_sec = self.config["billing_service_sec"]
        k_sec = self.config["kitchen_service_sec"]
        active_c = self.config["active_billing_counters"]
        active_k = self.config["active_kitchen_servers"]

        b_cap = (60.0 / b_sec) * active_c if b_sec > 0 else 0
        k_cap = (60.0 / k_sec) * active_k if k_sec > 0 else 0

        b_served = min(self.billing_queue + arr_rate, b_cap)
        self.billing_queue = max(0, int(self.billing_queue + arr_rate - b_served))

        k_served = min(self.kitchen_queue + b_served, k_cap)
        self.kitchen_queue = max(0, int(self.kitchen_queue + b_served - k_served))

        b_fc, k_fc = self.calculate_5min_forecast(
            self.billing_queue, self.kitchen_queue, arr_rate, b_sec, k_sec, active_c, active_k
        )

        state_dict = {
            "timestamp": datetime.datetime.now().isoformat(),
            "billing_queue": self.billing_queue,
            "kitchen_queue": self.kitchen_queue,
            "arrivals_last_min": arr_rate,
            "normal_arrivals_per_min": 5.0,
            "billing_service_time_sec": b_sec,
            "normal_billing_service_time_sec": 30.0,
            "kitchen_service_time_sec": k_sec,
            "normal_kitchen_service_time_sec": 120.0,
            "active_billing_counters": active_c,
            "active_kitchen_servers": active_k,
            "available_extra_counters": self.config["extra_counters"],
            "available_extra_workers": self.config["extra_workers"],
            "forecast_billing_queue": b_fc,
            "forecast_kitchen_queue": k_fc,
            "uncertainty": self.config["uncertainty"]
        }

        report = self.intelligence_engine.process_raw_input(custom_dict=state_dict)

        payload = {
            "step": self.sim_time_minutes,
            "timestamp": state_dict["timestamp"],
            "scenario": self.scenario_name,
            "estimated_data": state_dict,
            "forecast_billing_queue": b_fc,
            "forecast_kitchen_queue": k_fc,
            "report": report.to_dict()
        }

        self.history.append(payload)
        if len(self.history) > 100:
            self.history.pop(0)

        return payload

    def reset_simulation(self, scenario_name: str = "normal") -> Dict[str, Any]:
        self.scenario_name = scenario_name.lower()
        self.config = SCENARIO_CONFIGS.get(self.scenario_name, SCENARIO_CONFIGS["normal"]).copy()
        self.sim_time_minutes = 0
        self.billing_queue = 2
        self.kitchen_queue = 3
        self.history.clear()
        return self.step_simulation()

    async def broadcast_ws(self, message: Dict[str, Any]):
        disconnected = []
        for ws in self.active_websockets:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            if ws in self.active_websockets:
                self.active_websockets.remove(ws)


sim_manager = SimulationManager()


# ==============================================================================
# PYDANTIC INPUT REQUEST SCHEMAS
# ==============================================================================
class ScenarioRunRequest(BaseModel):
    scenario: str = Field(default="normal")

class SimulationStepRequest(BaseModel):
    duration_minutes: int = Field(default=1, ge=1, le=60)

class SimulationResetRequest(BaseModel):
    scenario: str = Field(default="normal")

class CustomStateInput(BaseModel):
    person1_data: Optional[Dict[str, Any]] = None
    person2_data: Optional[Dict[str, Any]] = None
    custom_dict: Optional[Dict[str, Any]] = None

class InterventionApplyRequest(BaseModel):
    action_name: str = Field(...)
    target_stage: Optional[str] = "Billing"


# ==============================================================================
# API ENDPOINTS
# ==============================================================================
@app.get("/")
def root():
    return {
        "app": "QueueFlux Backend Engine API",
        "author": "sbsureka (https://github.com/sbsureka)",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs",
        "scenarios": "/api/scenarios",
        "websocket": "/ws/live"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "active_ws_connections": len(sim_manager.active_websockets),
        "current_scenario": sim_manager.scenario_name,
        "sim_time_minutes": sim_manager.sim_time_minutes
    }


@app.get("/api/scenarios")
def get_available_scenarios():
    return {"scenarios": SCENARIO_CONFIGS}


@app.post("/api/scenarios/run")
def run_preset_scenario(req: ScenarioRunRequest):
    scenario_name = req.scenario.lower()
    if scenario_name not in SCENARIO_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown scenario '{req.scenario}'")

    cfg = SCENARIO_CONFIGS[scenario_name]
    custom_dict = {
        "timestamp": datetime.datetime.now().isoformat(),
        "billing_queue": 14 if "surge" in scenario_name or "slowdown" in scenario_name else 2,
        "kitchen_queue": 16 if scenario_name == "kitchen_slowdown" else 4,
        "arrivals_last_min": cfg["arrival_rate_per_min"],
        "normal_arrivals_per_min": 5.0,
        "billing_service_time_sec": cfg["billing_service_sec"],
        "normal_billing_service_time_sec": 30.0,
        "kitchen_service_time_sec": cfg["kitchen_service_sec"],
        "normal_kitchen_service_time_sec": 120.0,
        "active_billing_counters": cfg["active_billing_counters"],
        "active_kitchen_servers": cfg["active_kitchen_servers"],
        "available_extra_counters": cfg["extra_counters"],
        "available_extra_workers": cfg["extra_workers"],
        "uncertainty": cfg["uncertainty"]
    }

    report = sim_manager.intelligence_engine.process_raw_input(custom_dict=custom_dict)
    return {
        "scenario": scenario_name,
        "report": report.to_dict(),
        "executive_summary_text": report.to_human_summary()
    }


@app.get("/api/simulation/state")
def get_current_simulation_state():
    if not sim_manager.history:
        sim_manager.step_simulation()
    return sim_manager.history[-1]


@app.post("/api/simulation/step")
async def step_simulation_clock(req: SimulationStepRequest, background_tasks: BackgroundTasks):
    payload = sim_manager.step_simulation(duration_min=req.duration_minutes)
    background_tasks.add_task(sim_manager.broadcast_ws, payload)
    return payload


@app.post("/api/simulation/reset")
async def reset_simulation_state(req: SimulationResetRequest, background_tasks: BackgroundTasks):
    scenario_name = req.scenario.lower()
    if scenario_name not in SCENARIO_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown scenario '{req.scenario}'")

    payload = sim_manager.reset_simulation(scenario_name=scenario_name)
    background_tasks.add_task(sim_manager.broadcast_ws, payload)
    return payload


@app.post("/api/simulation/apply-action")
async def apply_manager_action(req: InterventionApplyRequest, background_tasks: BackgroundTasks):
    cfg = sim_manager.config
    action = req.action_name.upper()

    notes = ""
    if action == "OPEN_EXTRA_BILLING_COUNTER":
        if cfg["extra_counters"] > 0:
            cfg["active_billing_counters"] += 1
            cfg["extra_counters"] -= 1
            notes = "Opened 1 extra billing counter."
        else:
            raise HTTPException(status_code=400, detail="No extra counters available to open.")

    elif action in ["REALLOCATE_KITCHEN_HELP", "PREPARE_PREP_RUSH"]:
        if cfg["extra_workers"] > 0:
            cfg["active_kitchen_servers"] += 1
            cfg["extra_workers"] -= 1
            notes = f"Allocated 1 extra worker to {req.target_stage}."
        else:
            raise HTTPException(status_code=400, detail="No extra workers available to allocate.")

    elif action == "NO_ACTION":
        notes = "No action applied."
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported action '{req.action_name}'")

    payload = sim_manager.step_simulation(duration_min=1)
    payload["action_applied_note"] = notes
    background_tasks.add_task(sim_manager.broadcast_ws, payload)
    return payload


@app.post("/api/adapt")
def adapt_raw_teammate_input(input_data: CustomStateInput):
    state, warnings = IntegrationAdapter.adapt_raw_input(
        person1_data=input_data.person1_data,
        person2_data=input_data.person2_data,
        custom_dict=input_data.custom_dict
    )
    return {
        "normalized_state": state.to_dict(),
        "warnings": warnings,
        "calculated_wait_time_min": state.calculate_estimated_wait_time_min(),
        "billing_capacity_per_min": state.get_billing_capacity_per_min(),
        "kitchen_capacity_per_min": state.get_kitchen_capacity_per_min()
    }


@app.post("/api/evaluate")
def evaluate_custom_state(input_data: CustomStateInput):
    report = sim_manager.intelligence_engine.process_raw_input(
        person1_data=input_data.person1_data,
        person2_data=input_data.person2_data,
        custom_dict=input_data.custom_dict
    )
    return {
        "report": report.to_dict(),
        "executive_summary_text": report.to_human_summary()
    }


@app.get("/api/history")
def get_simulation_history():
    return {"history": sim_manager.history, "count": len(sim_manager.history)}


# ==============================================================================
# WEBSOCKET REAL-TIME STREAMING ENDPOINT
# ==============================================================================
@app.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    await websocket.accept()
    sim_manager.active_websockets.append(websocket)
    try:
        if sim_manager.history:
            await websocket.send_json(sim_manager.history[-1])

        while True:
            data_text = await websocket.receive_text()
            try:
                msg = json.loads(data_text)
                cmd = msg.get("command", "").lower()

                if cmd == "step":
                    dur = int(msg.get("duration", 1))
                    payload = sim_manager.step_simulation(duration_min=dur)
                    await sim_manager.broadcast_ws(payload)

                elif cmd == "reset":
                    scen = msg.get("scenario", "normal")
                    payload = sim_manager.reset_simulation(scenario_name=scen)
                    await sim_manager.broadcast_ws(payload)

                else:
                    await websocket.send_json({"error": f"Unknown WS command '{cmd}'"})

            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})

    except WebSocketDisconnect:
        if websocket in sim_manager.active_websockets:
            sim_manager.active_websockets.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    print("Starting QueueFlux Backend FastAPI Engine on Port 8000...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
