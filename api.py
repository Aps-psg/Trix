"""QueueFlux — Dynamic Consumer API & Queue Intelligence Backend
==============================================================
FastAPI Backend linking:
- Person 1 ML Model (Queue Prediction & 5-Min Forecast)
- Person 2 Intelligence (Surge Detection & Root-Cause Optimization)
- Database Layer (TimescaleDB / SQLite Time-Series & Order Tracking)
- Real Consumer App Flow: Login -> Ordering -> Real-Time Estimation -> Live Buzzer Tracker
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import database as db
from person1_ml_model import person1_predictor
from person2_intelligence import person2_detector

app = FastAPI(
    title="QueueFlux — Campus Dining Queue Intelligence System",
    description="Full-Stack Real-Time Queue Prediction, Detection & Consumer Order Tracking",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# LIVE SYSTEM STATE ENGINE
# ============================================================================
class LiveCanteenState:
    def __init__(self) -> None:
        self.queue_length = 3
        self.arrival_rate = 3.5
        self.service_rate = 4.0
        self.active_counters = 2
        self.avg_service_time = 1.2
        self.disruption_mode = "NORMAL"

    def step_minute(self) -> None:
        """Simulates live customer arrivals and counter departures for 1 minute."""
        # Adjust rates based on active disruption
        arr = self.arrival_rate
        ast = self.avg_service_time

        if self.disruption_mode == "HIGH_DEMAND":
            arr = 7.2
        elif self.disruption_mode == "SLOW_SERVICE":
            ast = 2.4
        elif self.disruption_mode == "COUNTER_IMBALANCE":
            self.active_counters = 1

        self.service_rate = self.active_counters / max(0.1, ast)
        net = int(arr - self.service_rate)
        self.queue_length = max(0, self.queue_length + net)

        # Record to TimescaleDB / SQLite
        db.record_telemetry(
            arrival_rate=round(arr, 2),
            service_rate=round(self.service_rate, 2),
            queue_length=self.queue_length,
            active_counters=self.active_counters,
            avg_service_time=round(ast, 2),
            congestion_level="CRITICAL" if self.queue_length > 10 else ("WARNING" if self.queue_length > 5 else "NORMAL")
        )


live_canteen = LiveCanteenState()


# ============================================================================
# DATA SCHEMAS
# ============================================================================
class LoginRequest(BaseModel):
    roll_or_email: str
    role: Optional[str] = "STUDENT"


class StudentRegisterRequest(BaseModel):
    roll_no: str
    name: str
    department: Optional[str] = "Computer Science"


class BillingTelemetryUpdate(BaseModel):
    queue_length: Optional[int] = None
    active_counters: Optional[int] = None
    avg_service_time: Optional[float] = None


class CartItem(BaseModel):
    item_id: str
    name: str
    category: str  # BEVERAGE, SNACK, MEAL, COMBO
    quantity: int
    base_prep_time_min: float
    price: float


class EstimateRequest(BaseModel):
    items: List[CartItem]


class PlaceOrderRequest(BaseModel):
    student_id: str
    student_name: str
    items: List[CartItem]


class CounterUpdateRequest(BaseModel):
    active_counters: int


class DisruptionTriggerRequest(BaseModel):
    disruption_type: str  # NORMAL, HIGH_DEMAND, SLOW_SERVICE, COUNTER_IMBALANCE


# ============================================================================
# MENU CATALOG
# ============================================================================
CANTEEN_MENU = [
    {
        "id": "item_bev_1",
        "name": "☕ South Indian Filter Coffee",
        "category": "BEVERAGE",
        "price": 25.0,
        "base_prep_time_min": 1.2,
        "description": "Freshly brewed chicory coffee. Quick 1-minute counter pickup.",
    },
    {
        "id": "item_snk_1",
        "name": "🥪 Grilled Veg Cheese Sandwich",
        "category": "SNACK",
        "price": 60.0,
        "base_prep_time_min": 2.5,
        "description": "Toasted multigrain bread with fresh vegetables and cheese.",
    },
    {
        "id": "item_meal_1",
        "name": "🍛 Executive Lunch Thali",
        "category": "MEAL",
        "price": 95.0,
        "base_prep_time_min": 5.0,
        "description": "Steamed basmati rice, paneer butter masala, dal fry & 2 rotis.",
    },
    {
        "id": "item_combo_1",
        "name": "🍱 Campus Rush Hour Combo Box",
        "category": "COMBO",
        "price": 120.0,
        "base_prep_time_min": 7.5,
        "description": "Full mini-meal with samosa snack and cold beverage included.",
    },
]


# ============================================================================
# 1. AUTHENTICATION & PROFILE APIS
# ============================================================================
@app.post("/api/auth/login")
def login(req: LoginRequest) -> Dict[str, Any]:
    """Authenticates student, cashier, or manager session."""
    conn = db.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE roll_or_email = ?", (req.roll_or_email.strip(),))
    user = cursor.fetchone()
    conn.close()

    if not user:
        # Create student account on the fly for demo convenience
        user_id = f"usr_{int(datetime.datetime.now().timestamp() % 10000)}"
        name = "Student " + req.roll_or_email.strip()
        conn = db.get_db_connection()
        conn.execute("INSERT INTO users (id, name, roll_or_email, role, avatar) VALUES (?, ?, ?, ?, ?)",
                     (user_id, name, req.roll_or_email.strip(), req.role or "STUDENT", "🎓"))
        conn.commit()
        conn.close()
        return {"user_id": user_id, "name": name, "roll_or_email": req.roll_or_email, "role": req.role or "STUDENT", "avatar": "🎓"}

    return dict(user)


@app.post("/api/auth/register-student")
def register_student(req: StudentRegisterRequest) -> Dict[str, Any]:
    """Registers or logs in a student using their College Roll Number and generates a unique UID."""
    clean_roll = req.roll_no.strip().upper()
    clean_name = req.name.strip()
    if not clean_roll or not clean_name:
        raise HTTPException(status_code=400, detail="Name and Roll Number are required")

    conn = db.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE roll_or_email = ?", (clean_roll,))
    user = cursor.fetchone()

    if not user:
        # Generate persistent student UID
        import random
        uid_num = random.randint(1000, 9999)
        user_id = f"STU-{clean_roll}-{uid_num}"
        cursor.execute(
            "INSERT INTO users (id, name, roll_or_email, role, avatar) VALUES (?, ?, ?, ?, ?)",
            (user_id, clean_name, clean_roll, "STUDENT", "🎓")
        )
        conn.commit()
        user_record = {
            "id": user_id,
            "user_id": user_id,
            "name": clean_name,
            "roll_no": clean_roll,
            "department": req.department,
            "role": "STUDENT",
            "avatar": "🎓"
        }
    else:
        user_record = dict(user)
        if not user_record["id"].startswith(f"STU-{clean_roll}"):
            import random
            uid_num = random.randint(1000, 9999)
            new_id = f"STU-{clean_roll}-{uid_num}"
            cursor.execute(
                "UPDATE users SET id = ?, name = ? WHERE roll_or_email = ?",
                (new_id, clean_name, clean_roll)
            )
            conn.commit()
            user_record["id"] = new_id
        user_record["user_id"] = user_record["id"]
        user_record["roll_no"] = clean_roll
        user_record["department"] = req.department

    conn.close()
    return user_record


@app.get("/api/auth/demo-users")
def get_demo_users() -> List[Dict[str, Any]]:
    """Lists pre-configured accounts for fast 1-click login."""
    conn = db.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    users = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return users


# ============================================================================
# BILLING COUNTER LIVE TELEMETRY & WAIT CALCULATION APIS
# ============================================================================
@app.get("/api/billing/telemetry")
def get_billing_telemetry() -> Dict[str, Any]:
    """Returns live inputs from billing counter: queue length, active counters,
    throughput rate, and real-time calculated billing waiting time.
    """
    effective_service_rate = live_canteen.active_counters / max(0.1, live_canteen.avg_service_time)
    billing_wait_min = (live_canteen.queue_length + 1) / max(0.1, effective_service_rate)
    congestion_level = "CRITICAL" if live_canteen.queue_length > 10 else ("WARNING" if live_canteen.queue_length > 5 else "NORMAL")

    return {
        "queue_length": live_canteen.queue_length,
        "active_counters": live_canteen.active_counters,
        "avg_service_time": round(live_canteen.avg_service_time, 2),
        "arrival_rate": round(live_canteen.arrival_rate, 2),
        "service_rate": round(effective_service_rate, 2),
        "billing_wait_min": round(billing_wait_min, 1),
        "billing_wait_sec": int(billing_wait_min * 60),
        "congestion_level": congestion_level,
        "formula": f"({live_canteen.queue_length} queue + 1) / ({live_canteen.active_counters} counters * {round(1.0 / max(0.1, live_canteen.avg_service_time), 2)} cust/min)"
    }


@app.post("/api/billing/update-telemetry")
def update_billing_telemetry(req: BillingTelemetryUpdate) -> Dict[str, Any]:
    """Updates live billing counter parameters directly."""
    if req.queue_length is not None:
        live_canteen.queue_length = max(0, req.queue_length)
    if req.active_counters is not None:
        live_canteen.active_counters = max(1, req.active_counters)
    if req.avg_service_time is not None:
        live_canteen.avg_service_time = max(0.2, req.avg_service_time)
    return get_billing_telemetry()


@app.post("/api/billing/serve-next")
def serve_next_billing_customer() -> Dict[str, Any]:
    """Decrements billing queue by 1 when a customer finishes checkout."""
    live_canteen.queue_length = max(0, live_canteen.queue_length - 1)
    return get_billing_telemetry()


@app.post("/api/billing/add-customer")
def add_billing_customer() -> Dict[str, Any]:
    """Increments billing queue by 1 when a customer joins line."""
    live_canteen.queue_length += 1
    return get_billing_telemetry()


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "QueueFlux", "version": "3.0.0"}


# ============================================================================
# 2. CONSUMER ORDERING & ESTIMATION ENGINE
# ============================================================================
@app.get("/api/consumer/menu")
def get_menu() -> List[Dict[str, Any]]:
    """Returns canteen menu with item categories and baseline preparation workloads."""
    return CANTEEN_MENU


@app.post("/api/consumer/estimate")
def compute_live_estimation(req: EstimateRequest) -> Dict[str, Any]:
    """Executes the internal processing engine:

    Combines Person 1's ML queue forecast, Person 2's disruption detections,
    and order item workloads to compute exact Billing Time + Prep Time + Queue Wait.
    """
    if not req.items:
        return {
            "billing_wait_min": 0.0,
            "queue_delay_min": 0.0,
            "kitchen_prep_min": 0.0,
            "total_wait_min": 0.0,
            "total_wait_sec": 0,
            "ready_time_str": datetime.datetime.now().strftime("%I:%M %p"),
        }

    # 1. Run Person 1 ML Prediction
    ml_res = person1_predictor.predict(
        queue_length=live_canteen.queue_length,
        arrival_rate=live_canteen.arrival_rate,
        service_rate=live_canteen.service_rate,
        active_counters=live_canteen.active_counters,
        avg_service_time=live_canteen.avg_service_time,
    )
    predicted_queue = ml_res["predicted_queue_length"]

    # 2. Run Person 2 Diagnostic
    det_res = person2_detector.analyze(
        queue_length=live_canteen.queue_length,
        arrival_rate=live_canteen.arrival_rate,
        service_rate=live_canteen.service_rate,
        active_counters=live_canteen.active_counters,
        avg_service_time=live_canteen.avg_service_time,
        ml_predicted_queue=predicted_queue,
        ml_predicted_wait=ml_res["predicted_wait_time"],
    )

    # 3. Stage A: Billing Time Estimation
    billing_time_min = live_canteen.avg_service_time
    queue_delay_min = (predicted_queue * live_canteen.avg_service_time) / max(1, live_canteen.active_counters)

    # 4. Stage B: Kitchen Preparation Time (Item-Specific Workload)
    max_prep_item = max(item.base_prep_time_min for item in req.items)
    total_items = sum(item.quantity for item in req.items)
    batching_factor = 1.0 + 0.2 * max(0, total_items - 1)
    kitchen_prep_min = max_prep_item * batching_factor

    # Adjust for slowdown disruption if detected by Person 2
    if det_res["slowdown_detected"]:
        kitchen_prep_min *= 1.35

    # 5. Total Expected Registration-to-Collection Time
    total_wait_min = billing_time_min + queue_delay_min + kitchen_prep_min
    total_wait_sec = int(total_wait_min * 60)

    now = datetime.datetime.now()
    ready_time = now + datetime.timedelta(seconds=total_wait_sec)

    return {
        "billing_wait_min": round(billing_time_min, 1),
        "queue_delay_min": round(queue_delay_min, 1),
        "kitchen_prep_min": round(kitchen_prep_min, 1),
        "total_wait_min": round(total_wait_min, 1),
        "total_wait_sec": total_wait_sec,
        "ready_time_str": ready_time.strftime("%I:%M %p"),
        "predicted_queue_ahead": int(round(predicted_queue)),
        "congestion_level": det_res["congestion_level"],
        "congestion_cause": det_res["congestion_cause"],
    }


@app.post("/api/consumer/orders")
def place_order(req: PlaceOrderRequest) -> Dict[str, Any]:
    """Places consumer order, stores in database, and generates active ticket with countdown."""
    if not req.items:
        raise HTTPException(status_code=400, detail="Cannot place empty order")

    # Run estimation
    est_req = EstimateRequest(items=req.items)
    estimation = compute_live_estimation(est_req)

    category = req.items[0].category if req.items else "MEAL"
    assigned_counter = (int(datetime.datetime.now().timestamp()) % live_canteen.active_counters) + 1

    # Record to DB
    order = db.create_order(
        student_id=req.student_id,
        student_name=req.student_name,
        items=[item.model_dump() for item in req.items],
        order_category=category,
        billing_wait_sec=estimation["billing_wait_min"] * 60,
        prep_time_sec=estimation["kitchen_prep_min"] * 60,
        total_wait_sec=estimation["total_wait_sec"],
        counter_assigned=assigned_counter,
    )

    # Increment live queue
    live_canteen.queue_length += 1

    return {
        "status": "success",
        "order": order,
        "estimation": estimation,
    }


@app.get("/api/consumer/my-orders")
def get_my_orders(student_id: str) -> List[Dict[str, Any]]:
    """Retrieves active orders and past tickets for a student."""
    return db.get_student_orders(student_id)


# ============================================================================
# 3. KITCHEN STAFF & CASHIER CONSOLE APIS
# ============================================================================
@app.get("/api/staff/queue")
def get_staff_queue() -> Dict[str, Any]:
    """Retrieves all active orders segmented by Stage A (Billing) and Stage B (Kitchen)."""
    orders = db.get_active_orders()
    billing_orders = [o for o in orders if o["order_status"] == "WAITING_BILLING"]
    kitchen_orders = [o for o in orders if o["order_status"] == "PREPARING"]
    ready_orders = [o for o in orders if o["order_status"] == "READY"]

    return {
        "active_counters": live_canteen.active_counters,
        "total_active": len(orders),
        "billing_stage": billing_orders,
        "kitchen_stage": kitchen_orders,
        "ready_stage": ready_orders,
    }


@app.post("/api/staff/orders/{order_id}/advance")
def advance_order(order_id: str) -> Dict[str, Any]:
    """Advances order status through the canteen lifecycle."""
    res = db.advance_order_status(order_id)
    if res.get("current_status") == "READY":
        live_canteen.queue_length = max(0, live_canteen.queue_length - 1)
    return res


# ============================================================================
# 4. CANTEEN MANAGER CONTROL TOWER & ANALYTICS
# ============================================================================
@app.get("/api/manager/control-tower")
def get_manager_control_tower() -> Dict[str, Any]:
    """Combines Person 1 predictions, Person 2 detections, and TimescaleDB telemetry."""
    # 1. Person 1 Model Forecast
    ml_res = person1_predictor.predict(
        queue_length=live_canteen.queue_length,
        arrival_rate=live_canteen.arrival_rate,
        service_rate=live_canteen.service_rate,
        active_counters=live_canteen.active_counters,
        avg_service_time=live_canteen.avg_service_time,
    )

    # 2. Person 2 Intelligence Diagnostic
    det_res = person2_detector.analyze(
        queue_length=live_canteen.queue_length,
        arrival_rate=live_canteen.arrival_rate,
        service_rate=live_canteen.service_rate,
        active_counters=live_canteen.active_counters,
        avg_service_time=live_canteen.avg_service_time,
        ml_predicted_queue=ml_res["predicted_queue_length"],
        ml_predicted_wait=ml_res["predicted_wait_time"],
    )

    # 3. Recent Time-Series from DB
    history = db.get_recent_telemetry(limit=15)

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "live_metrics": {
            "queue_length": live_canteen.queue_length,
            "arrival_rate": round(live_canteen.arrival_rate, 2),
            "service_rate": round(live_canteen.service_rate, 2),
            "active_counters": live_canteen.active_counters,
            "avg_service_time": round(live_canteen.avg_service_time, 2),
            "disruption_mode": live_canteen.disruption_mode,
        },
        "person1_prediction": ml_res,
        "person2_intelligence": det_res,
        "time_series_history": history,
    }


@app.post("/api/manager/counters")
def update_counters(req: CounterUpdateRequest) -> Dict[str, Any]:
    """Adjusts active billing counters."""
    live_canteen.active_counters = max(1, req.active_counters)
    return {"status": "success", "active_counters": live_canteen.active_counters}


@app.post("/api/manager/disruption")
def set_disruption(req: DisruptionTriggerRequest) -> Dict[str, Any]:
    """Injects operational scenario (HIGH_DEMAND, SLOW_SERVICE, COUNTER_IMBALANCE, NORMAL)."""
    live_canteen.disruption_mode = req.disruption_type.upper()
    return {"status": "success", "active_disruption": live_canteen.disruption_mode}


@app.post("/api/manager/step")
def simulate_step() -> Dict[str, Any]:
    """Advances queue simulation by 1 minute and writes snapshot to TimescaleDB/SQLite."""
    live_canteen.step_minute()
    return get_manager_control_tower()


# ============================================================================
# 5. STATIC FRONTEND SERVING
# student.html = student login + app  (served at / and /student)
# model.html   = AI Control Tower     (served at /model)
# ============================================================================
root_dir = Path(__file__).parent

@app.get("/")
def serve_student_app():
    """Student App — login page first, then live dashboard."""
    f = root_dir / "student.html"
    if f.exists():
        return FileResponse(f)
    return HTMLResponse("<h1>QueueFlux — student.html not found</h1>", status_code=404)

@app.get("/student")
def serve_student_alias():
    return serve_student_app()

@app.get("/model")
def serve_model_dashboard():
    """AI Control Tower — Person 1 ML + Person 2 Intelligence."""
    f = root_dir / "model.html"
    if f.exists():
        return FileResponse(f)
    return HTMLResponse("<h1>QueueFlux — model.html not found</h1>", status_code=404)

@app.get("/dashboard")
def serve_legacy_dashboard():
    """Legacy full-combined dashboard (backward compat)."""
    f = root_dir / "index.html"
    if f.exists():
        return FileResponse(f)
    return serve_student_app()

frontend_dir = root_dir / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


if __name__ == "__main__":
    import uvicorn

    print("Starting QueueFlux Consumer Web & Intelligence Server on http://localhost:8000 ...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
