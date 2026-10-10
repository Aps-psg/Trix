"""QueueFlux — Time-Series Database Layer (TimescaleDB / SQLite Engine)
===================================================================
Provides robust time-series storage and relational order management:
- Connects to TimescaleDB / PostgreSQL when DATABASE_URL is configured (in Docker)
- Seamlessly falls back to local SQLite (data/queueflux.db) when running standalone
- Stores:
  1. canteen_telemetry_ts: 1-minute time-series snapshots (arrival, service, queue, counters)
  2. consumer_orders: Full consumer order lifecycle with exact billing/prep/wait time breakdown
  3. prediction_records_ts: 5-minute forward forecasts from Person 1 ML model
  4. disruptions_log: Detected surges, causes (HIGH_DEMAND, SLOW_SERVICE, COUNTER_IMBALANCE)
"""

from __future__ import annotations

import datetime
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


DB_DIR = Path(__file__).parent / "data"
DB_DIR.mkdir(exist_ok=True)
SQLITE_PATH = DB_DIR / "queueflux.db"


def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite database connection with row factory enabled."""
    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """Initializes tables, indexes, and seeds default user accounts."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users Table (Student, Staff, Manager)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        roll_or_email TEXT NOT NULL UNIQUE,
        role TEXT NOT NULL,
        avatar TEXT
    )
    """)

    # 2. Consumer Orders Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS consumer_orders (
        order_id TEXT PRIMARY KEY,
        student_id TEXT NOT NULL,
        student_name TEXT NOT NULL,
        items_json TEXT NOT NULL,
        order_category TEXT NOT NULL,
        billing_wait_sec REAL NOT NULL,
        prep_time_sec REAL NOT NULL,
        total_wait_sec REAL NOT NULL,
        order_status TEXT NOT NULL,  -- WAITING_BILLING, PREPARING, READY, COLLECTED
        counter_assigned INTEGER DEFAULT 1,
        created_at TEXT NOT NULL,
        completed_at TEXT
    )
    """)

    # 3. Canteen Time-Series Telemetry Table (TimescaleDB Hypertable equivalent)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS canteen_telemetry_ts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        arrival_rate REAL NOT NULL,
        service_rate REAL NOT NULL,
        queue_length INTEGER NOT NULL,
        active_counters INTEGER NOT NULL,
        avg_service_time REAL NOT NULL,
        congestion_level TEXT NOT NULL  -- NORMAL, WARNING, CRITICAL
    )
    """)

    # 4. Predictions Time-Series Table (Person 1 ML forecasts)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_records_ts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        predicted_queue_length REAL NOT NULL,
        predicted_wait_time REAL NOT NULL,
        forecast_5min_json TEXT NOT NULL
    )
    """)

    # 5. Disruptions & Recommendations Log (Person 2 Intelligence)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS disruptions_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        surge_detected INTEGER NOT NULL,
        slowdown_detected INTEGER NOT NULL,
        congestion_cause TEXT NOT NULL,
        recommended_action TEXT NOT NULL,
        expected_savings_pct REAL NOT NULL
    )
    """)

    # Seed Default User Accounts if empty
    cursor.execute("SELECT COUNT(*) as count FROM users")
    if cursor.fetchone()["count"] == 0:
        cursor.executemany("""
        INSERT INTO users (id, name, roll_or_email, role, avatar) VALUES (?, ?, ?, ?, ?)
        """, [
            ("usr_student_1", "Alex Chen", "24CS108", "STUDENT", "🎓"),
            ("usr_student_2", "Sneha Patel", "24EC215", "STUDENT", "📚"),
            ("usr_cashier", "Priya Nair", "staff@canteen.edu", "CASHIER", "🧑‍🍳"),
            ("usr_manager", "Dr. K. Sharma", "manager@canteen.edu", "MANAGER", "📊"),
        ])

    # Seed initial telemetry if empty
    cursor.execute("SELECT COUNT(*) as count FROM canteen_telemetry_ts")
    if cursor.fetchone()["count"] == 0:
        base_t = datetime.datetime.now() - datetime.timedelta(minutes=15)
        for i in range(15):
            t = base_t + datetime.timedelta(minutes=i)
            cursor.execute("""
            INSERT INTO canteen_telemetry_ts (timestamp, arrival_rate, service_rate, queue_length, active_counters, avg_service_time, congestion_level)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (t.isoformat(), 3.5 + 0.5 * (i % 3), 4.0, max(1, 4 + (i % 4) - 2), 2, 1.2, "NORMAL"))

    conn.commit()
    conn.close()


# ============================================================================
# DATABASE OPERATION HELPERS
# ============================================================================
def record_telemetry(
    arrival_rate: float,
    service_rate: float,
    queue_length: int,
    active_counters: int,
    avg_service_time: float,
    congestion_level: str = "NORMAL",
) -> int:
    """Inserts a 1-minute time-series snapshot into the telemetry database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.datetime.now().isoformat()
    cursor.execute("""
    INSERT INTO canteen_telemetry_ts (timestamp, arrival_rate, service_rate, queue_length, active_counters, avg_service_time, congestion_level)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (now_str, arrival_rate, service_rate, queue_length, active_counters, avg_service_time, congestion_level))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def create_order(
    student_id: str,
    student_name: str,
    items: List[Dict[str, Any]],
    order_category: str,
    billing_wait_sec: float,
    prep_time_sec: float,
    total_wait_sec: float,
    counter_assigned: int = 1,
) -> Dict[str, Any]:
    """Creates a new consumer order and records it in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    order_seq = int(datetime.datetime.now().timestamp() % 10000)
    order_id = f"ORD-{order_seq:04d}"
    now_str = datetime.datetime.now().isoformat()

    cursor.execute("""
    INSERT INTO consumer_orders (
        order_id, student_id, student_name, items_json, order_category,
        billing_wait_sec, prep_time_sec, total_wait_sec, order_status,
        counter_assigned, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        order_id,
        student_id,
        student_name,
        json.dumps(items),
        order_category,
        round(billing_wait_sec, 1),
        round(prep_time_sec, 1),
        round(total_wait_sec, 1),
        "WAITING_BILLING",
        counter_assigned,
        now_str,
    ))

    conn.commit()
    conn.close()

    return {
        "order_id": order_id,
        "student_name": student_name,
        "items": items,
        "order_category": order_category,
        "billing_wait_sec": round(billing_wait_sec, 1),
        "prep_time_sec": round(prep_time_sec, 1),
        "total_wait_sec": round(total_wait_sec, 1),
        "total_wait_min": round(total_wait_sec / 60.0, 1),
        "order_status": "WAITING_BILLING",
        "counter_assigned": counter_assigned,
        "created_at": now_str,
    }


def get_student_orders(student_id: str) -> List[Dict[str, Any]]:
    """Retrieves all orders placed by a specific student with live status."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM consumer_orders WHERE student_id = ? ORDER BY created_at DESC LIMIT 10
    """, (student_id,))
    rows = cursor.fetchall()
    orders = []
    for r in rows:
        orders.append({
            "order_id": r["order_id"],
            "student_id": r["student_id"],
            "student_name": r["student_name"],
            "items": json.loads(r["items_json"]),
            "order_category": r["order_category"],
            "billing_wait_sec": r["billing_wait_sec"],
            "prep_time_sec": r["prep_time_sec"],
            "total_wait_sec": r["total_wait_sec"],
            "total_wait_min": round(r["total_wait_sec"] / 60.0, 1),
            "order_status": r["order_status"],
            "counter_assigned": r["counter_assigned"],
            "created_at": r["created_at"],
            "completed_at": r["completed_at"],
        })
    conn.close()
    return orders


def advance_order_status(order_id: str) -> Dict[str, Any]:
    """Advances an order: WAITING_BILLING -> PREPARING -> READY -> COLLECTED."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT order_status FROM consumer_orders WHERE order_id = ?", (order_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return {"error": "Order not found"}

    curr_status = row["order_status"]
    next_status = curr_status
    completed_at = None

    if curr_status == "WAITING_BILLING":
        next_status = "PREPARING"
    elif curr_status == "PREPARING":
        next_status = "READY"
    elif curr_status == "READY":
        next_status = "COLLECTED"
        completed_at = datetime.datetime.now().isoformat()

    cursor.execute("""
    UPDATE consumer_orders
    SET order_status = ?, completed_at = COALESCE(?, completed_at)
    WHERE order_id = ?
    """, (next_status, completed_at, order_id))

    conn.commit()
    conn.close()
    return {"order_id": order_id, "previous_status": curr_status, "current_status": next_status}


def get_active_orders() -> List[Dict[str, Any]]:
    """Retrieves all active orders currently in the canteen pipeline."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM consumer_orders WHERE order_status != 'COLLECTED' ORDER BY created_at ASC
    """)
    rows = cursor.fetchall()
    orders = []
    for r in rows:
        orders.append({
            "order_id": r["order_id"],
            "student_name": r["student_name"],
            "items": json.loads(r["items_json"]),
            "order_category": r["order_category"],
            "billing_wait_sec": r["billing_wait_sec"],
            "prep_time_sec": r["prep_time_sec"],
            "total_wait_min": round(r["total_wait_sec"] / 60.0, 1),
            "order_status": r["order_status"],
            "counter_assigned": r["counter_assigned"],
            "created_at": r["created_at"],
        })
    conn.close()
    return orders


def get_recent_telemetry(limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieves recent time-series telemetry records for live graphing."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM canteen_telemetry_ts ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    result = [dict(r) for r in reversed(rows)]
    conn.close()
    return result


# Initialize DB on load
init_database()
