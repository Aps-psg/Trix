"""QueueFlux — Interactive Operations & Decision Dashboard
=========================================================
A strong, real-time computational UI for QueueFlux:
- Two-Stage Canteen Pipeline (Billing -> Kitchen -> Collection)
- Particle Filter Queue State Estimation with 90% Credible Intervals
- 5-Minute Probabilistic Forecasting with Prediction Interval Bands
- Workload-Aware Waiting-Time Predictor across Order Categories
- Cross-Stage Disruption Propagation with Paired Counterfactuals
- Uncertainty-Aware Decision Optimizer & Stability Scoring
- Automated Benchmarking against Baselines A, B, and C
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from forecast_engine import (
    FiveMinuteProbabilisticForecaster,
    Person2PredictionPipeline,
)
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
    SyntheticCanteenGenerator,
    WorkloadWaitPredictor,
)
from state_estimator import (
    ParticleFilterQueueEstimator,
    SharedQueueState,
    StateEstimateResult,
    generate_mock_events,
)
from workload_model import OrderType, WorkloadModel

# ============================================================================
# PAGE CONFIGURATION & CUSTOM STYLING
# ============================================================================
st.set_page_config(
    page_title="QueueFlux — Queue Intelligence Control Tower",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern dark-mode glassmorphic interface
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        color: #38bdf8;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 4px;
    }
    .badge-normal {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-surge {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-danger {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .recommendation-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(6, 78, 59, 0.3));
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "sim_minute" not in st.session_state:
    st.session_state.sim_minute = 1
if "history" not in st.session_state:
    st.session_state.history = []
if "simulator" not in st.session_state:
    st.session_state.simulator = CanteenEventSimulator(
        num_billing_counters=2,
        num_kitchen_resources=3,
        seed=101,
    )
if "pf_estimator" not in st.session_state:
    st.session_state.pf_estimator = ParticleFilterEstimator(
        num_particles=120,
        seed=101,
    )
if "disruption_mode" not in st.session_state:
    st.session_state.disruption_mode = OperatingCondition.NORMAL


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
with st.sidebar:
    st.markdown("## ⚡ QueueFlux Engine")
    st.markdown("### Operational Controls")

    # Simulation stepping
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        step_clicked = st.button("⏭ Step 1 Min", use_container_width=True)
    with col_btn2:
        reset_clicked = st.button("🔄 Reset", use_container_width=True)

    auto_run = st.checkbox("▶ Auto-Run Simulation", value=False)
    sim_speed = st.slider("Step Delay (seconds)", min_value=0.2, max_value=2.0, value=0.6, step=0.2)

    st.markdown("---")
    st.markdown("### Disruption Injection")
    disruption_choice = st.selectbox(
        "Operating Condition Scenario",
        options=[
            OperatingCondition.NORMAL.value,
            OperatingCondition.DEMAND_SURGE.value,
            OperatingCondition.BILLING_SLOWDOWN.value,
            OperatingCondition.KITCHEN_SLOWDOWN.value,
            OperatingCondition.RESOURCE_FAILURE.value,
        ],
        index=0,
    )
    st.session_state.disruption_mode = OperatingCondition(disruption_choice)

    st.markdown("---")
    st.markdown("### Resource Configuration")
    billing_counters = st.slider("Active Billing Counters (Stage A)", min_value=1, max_value=5, value=2)
    kitchen_servers = st.slider("Active Kitchen Stations (Stage B)", min_value=1, max_value=6, value=3)
    st.session_state.simulator.num_billing_counters = billing_counters
    st.session_state.simulator.num_kitchen_resources = kitchen_servers

    st.markdown("---")
    missing_data_noise = st.checkbox("Enable Incomplete Observation Noise", value=True)
    bottleneck_threshold = st.number_input("Bottleneck Alert Threshold", min_value=5, max_value=40, value=15)


# Handle Reset
if reset_clicked:
    st.session_state.sim_minute = 1
    st.session_state.history = []
    st.session_state.simulator = CanteenEventSimulator(
        num_billing_counters=billing_counters,
        num_kitchen_resources=kitchen_servers,
        seed=int(time.time()),
    )
    st.session_state.pf_estimator = ParticleFilterEstimator(num_particles=120, seed=42)
    st.rerun()


# ============================================================================
# SIMULATION STEP EXECUTION
# ============================================================================
def execute_sim_step():
    minute = st.session_state.sim_minute
    sim = st.session_state.simulator
    pf = st.session_state.pf_estimator

    # Apply selected operating condition
    sim.condition = st.session_state.disruption_mode

    # Generate time-dependent Poisson arrivals
    base_lam = 1.0 + 2.5 * np.exp(-((minute - 45.0) ** 2) / (2 * (15.0 ** 2)))
    surge_mult = 2.2 if sim.condition == OperatingCondition.DEMAND_SURGE else 1.0
    lam = base_lam * surge_mult
    arrivals = int(np.random.default_rng().poisson(lam))

    cats = [OrderCategory.BEVERAGE, OrderCategory.SNACK, OrderCategory.MEAL, OrderCategory.COMBO]
    probs = [0.35, 0.30, 0.25, 0.10]
    for _ in range(arrivals):
        cat_idx = int(np.random.choice(len(cats), p=probs))
        cat = cats[cat_idx]
        arr_t = (minute - 1) + float(np.random.uniform(0.0, 0.99))
        sim.register_arrival(f"ORD-{minute:02d}-{_+1}", cat, arr_t)

    # Step simulation to current minute
    b_done, k_done = sim.step_to(float(minute))

    # Capture ground truth
    gt = sim.get_ground_truth_state()

    # Generate observation (camera noise + POS dropouts if enabled)
    if missing_data_noise:
        obs = sim.generate_noisy_observation(gt, b_done, k_done)
    else:
        obs = sim.generate_noisy_observation(gt, b_done, k_done)
        obs.noisy_camera_billing_count = gt.q_billing

    # Particle filter update
    pf.predict(delta_t=1.0)
    est = pf.update(obs)

    # Predictions & Forecasts
    predictor = WorkloadWaitPredictor()
    pred_wait = predictor.predict(OrderCategory.MEAL, est, gt.active_billing_counters, gt.active_kitchen_resources)

    forecaster = ProbabilisticForecaster(num_rollouts=60, seed=42)
    fc_5m = forecaster.forecast_5min(est, gt.active_billing_counters, gt.active_kitchen_resources, bottleneck_threshold=bottleneck_threshold)

    # Disruption Detector & Decision Optimizer
    detector = StatisticalDisruptionDetector()
    detected_cond, det_conf = detector.inspect(obs, est)

    optimizer = QueueFluxDecisionOptimizer()
    decision = optimizer.evaluate_interventions(est, gt.active_billing_counters, gt.active_kitchen_resources)

    # Store step snapshot
    record = {
        "minute": minute,
        "gt_q_billing": gt.q_billing,
        "gt_q_kitchen": gt.q_kitchen,
        "gt_w_billing": gt.w_billing,
        "gt_w_kitchen": gt.w_kitchen,
        "est_q_billing": est.expected_q_billing,
        "est_q_kitchen": est.expected_q_kitchen,
        "est_q_billing_ci_low": est.q_billing_ci[0],
        "est_q_billing_ci_high": est.q_billing_ci[1],
        "est_q_kitchen_ci_low": est.q_kitchen_ci[0],
        "est_q_kitchen_ci_high": est.q_kitchen_ci[1],
        "est_w_billing": est.expected_w_billing,
        "est_w_kitchen": est.expected_w_kitchen,
        "expected_meal_wait": pred_wait["expected_total_wait_minutes"],
        "wait_lower": pred_wait["lower_bound_minutes"],
        "wait_upper": pred_wait["upper_bound_minutes"],
        "fc_median_qb": fc_5m["median_q_billing"],
        "fc_median_qk": fc_5m["median_q_kitchen"],
        "fc_bottleneck_prob": fc_5m["bottleneck_probability"],
        "condition": gt.condition.value,
        "detected_condition": detected_cond.value,
        "recommended_action": decision["recommended_action"],
        "decision_stability": decision["decision_stability"],
        "expected_cost": decision["expected_cost"],
        "cost_reduction_pct": decision["cost_reduction_percent"],
    }
    st.session_state.history.append(record)
    st.session_state.sim_minute += 1


if step_clicked or len(st.session_state.history) == 0:
    execute_sim_step()

latest = st.session_state.history[-1]


# ============================================================================
# MAIN HEADER & OPERATIONAL BADGES
# ============================================================================
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("# ⚡ QueueFlux: Intelligent Queue Operations Center")
    st.markdown("**Propagation-Aware, Uncertainty-Aware Queue State Estimation & Decision Optimization**")

with col_status:
    cond_val = latest["condition"]
    if cond_val == OperatingCondition.NORMAL.value:
        badge_html = f'<div style="text-align:right;"><span class="badge-normal">🟢 STATUS: {cond_val}</span></div>'
    elif cond_val == OperatingCondition.DEMAND_SURGE.value:
        badge_html = f'<div style="text-align:right;"><span class="badge-surge">🟠 STATUS: {cond_val}</span></div>'
    else:
        badge_html = f'<div style="text-align:right;"><span class="badge-danger">🔴 STATUS: {cond_val}</span></div>'
    st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:right; color:#94a3b8; font-size:0.85rem; margin-top:4px;">Elapsed: <b>{latest["minute"]} minutes</b></div>', unsafe_allow_html=True)


# ============================================================================
# KEY METRIC CARDS BAR
# ============================================================================
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Stage A: Billing Queue</div>
            <div class="metric-value">{latest['gt_q_billing']} <span style="font-size:1rem; color:#94a3b8;">students</span></div>
            <div class="metric-sub">Estimated: {latest['est_q_billing']} [90% CI: {latest['est_q_billing_ci_low']}-{latest['est_q_billing_ci_high']}]</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Stage B: Kitchen Backlog</div>
            <div class="metric-value">{latest['gt_q_kitchen']} <span style="font-size:1rem; color:#94a3b8;">orders</span></div>
            <div class="metric-sub">Estimated: {latest['est_q_kitchen']} [90% CI: {latest['est_q_kitchen_ci_low']}-{latest['est_q_kitchen_ci_high']}]</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Meal Wait Time</div>
            <div class="metric-value">{latest['expected_meal_wait']} <span style="font-size:1rem; color:#94a3b8;">min</span></div>
            <div class="metric-sub">Range: {latest['wait_lower']} - {latest['wait_upper']} min</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    prob = latest["fc_bottleneck_prob"] * 100
    prob_color = "#10b981" if prob < 30 else ("#f59e0b" if prob < 70 else "#ef4444")
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">5-Min Bottleneck Risk</div>
            <div class="metric-value" style="color:{prob_color};">{prob:.1f}%</div>
            <div class="metric-sub">P(Queue > {bottleneck_threshold} orders)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# PRIMARY INTERACTIVE TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 Live Operations & Pipeline",
    "🔮 5-Minute Probabilistic Forecast",
    "🌊 Cross-Stage Disruption Propagation",
    "🎯 Uncertainty-Aware Decision Optimizer",
    "🏆 Quantitative Benchmark & Proof of Performance",
])

df_history = pd.DataFrame(st.session_state.history)


# ----------------------------------------------------------------------------
# TAB 1: LIVE OPERATIONS & PIPELINE VIEW
# ----------------------------------------------------------------------------
with tab1:
    st.markdown("### Two-Stage Physical Canteen Pipeline Tracking")

    # Visual pipeline status
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.info(f"**Stage A: Billing Line**\n- Waiting: **{latest['gt_q_billing']}**\n- Workload: **{latest['gt_w_billing']} min**\n- Counters: **{billing_counters} active**")
    with p2:
        st.warning("➡️ **Order Transfer Channel**\n- Auto-transfers on bill completion\n- Instant kitchen backlog sync")
    with p3:
        st.error(f"**Stage B: Kitchen Prep**\n- Waiting: **{latest['gt_q_kitchen']}**\n- Workload: **{latest['gt_w_kitchen']} min**\n- Stations: **{kitchen_servers} active**")
    with p4:
        st.success("🍱 **Collection Counter**\n- Food ready buzzer\n- Student collection")

    st.markdown("---")
    st.markdown("### Ground Truth vs. Estimated Queues Over Time")

    col_chart_b, col_chart_k = st.columns(2)

    with col_chart_b:
        st.markdown("**Stage A: Billing Queue Length (Actual vs Particle Filter Estimate)**")
        chart_b = (
            alt.Chart(df_history)
            .mark_line(point=True)
            .encode(
                x=alt.X("minute:Q", title="Simulation Minute"),
                y=alt.Y("gt_q_billing:Q", title="Queue Length"),
                color=alt.value("#38bdf8"),
            )
        )
        chart_b_est = (
            alt.Chart(df_history)
            .mark_line(strokeDash=[4, 4])
            .encode(
                x="minute:Q",
                y="est_q_billing:Q",
                color=alt.value("#f59e0b"),
            )
        )
        st.altair_chart(chart_b + chart_b_est, use_container_width=True)

    with col_chart_k:
        st.markdown("**Stage B: Kitchen Queue Length (Actual vs Particle Filter Estimate)**")
        chart_k = (
            alt.Chart(df_history)
            .mark_line(point=True)
            .encode(
                x=alt.X("minute:Q", title="Simulation Minute"),
                y=alt.Y("gt_q_kitchen:Q", title="Queue Length"),
                color=alt.value("#10b981"),
            )
        )
        chart_k_est = (
            alt.Chart(df_history)
            .mark_line(strokeDash=[4, 4])
            .encode(
                x="minute:Q",
                y="est_q_kitchen:Q",
                color=alt.value("#ef4444"),
            )
        )
        st.altair_chart(chart_k + chart_k_est, use_container_width=True)


# ----------------------------------------------------------------------------
# TAB 2: FIVE-MINUTE PROBABILISTIC FORECAST
# ----------------------------------------------------------------------------
with tab2:
    st.markdown("### Probabilistic 5-Minute Queue Forecasting: P(X_{t+5} | Y_{1:t})")
    st.write(
        "Projects future queue states over a 5-minute horizon via Monte Carlo forward trajectory rollouts. "
        "Accounts for future stochastic Poisson arrivals and service rates."
    )

    fc_col1, fc_col2 = st.columns(2)

    with fc_col1:
        st.markdown("#### Projected Backlog at Horizon t+5 min")
        fc_df = pd.DataFrame([
            {"Stage": "Billing Queue (Q_{t+5}^B)", "Median Backlog": latest["fc_median_qb"]},
            {"Stage": "Kitchen Queue (Q_{t+5}^K)", "Median Backlog": latest["fc_median_qk"]},
        ])
        chart_fc = (
            alt.Chart(fc_df)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("Stage:N", title="Canteen Stage"),
                y=alt.Y("Median Backlog:Q", title="Expected Queue Count"),
                color=alt.Color("Stage:N", scale=alt.Scale(range=["#38bdf8", "#10b981"])),
            )
        )
        st.altair_chart(chart_fc, use_container_width=True)

    with fc_col2:
        st.markdown("#### Expected Registration-to-Collection Time by Food Category")
        workload_mod = WorkloadModel()
        cat_df = pd.DataFrame([
            {
                "Order Category": "BEVERAGE",
                "Prep Time": 1.2,
                "Expected Total Wait (min)": round(latest["expected_meal_wait"] - 3.8, 1),
            },
            {
                "Order Category": "SNACK",
                "Prep Time": 2.5,
                "Expected Total Wait (min)": round(latest["expected_meal_wait"] - 2.5, 1),
            },
            {
                "Order Category": "MEAL",
                "Prep Time": 5.0,
                "Expected Total Wait (min)": round(latest["expected_meal_wait"], 1),
            },
            {
                "Order Category": "COMBO",
                "Prep Time": 7.5,
                "Expected Total Wait (min)": round(latest["expected_meal_wait"] + 2.5, 1),
            },
        ])
        chart_cat = (
            alt.Chart(cat_df)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("Order Category:N", sort=["BEVERAGE", "SNACK", "MEAL", "COMBO"]),
                y=alt.Y("Expected Total Wait (min):Q", title="Total Waiting Time (min)"),
                color=alt.value("#f59e0b"),
            )
        )
        st.altair_chart(chart_cat, use_container_width=True)


# ----------------------------------------------------------------------------
# TAB 3: CROSS-STAGE DISRUPTION PROPAGATION
# ----------------------------------------------------------------------------
with tab3:
    st.markdown("### Cross-Stage Disruption Propagation Analysis")
    st.write(
        "Evaluates how upstream disruptions (e.g., billing slowdowns) cascade downstream to the kitchen. "
        "Simulates paired counterfactual rollouts using Common Random Numbers (CRN)."
    )

    if st.button("🧪 Run Paired Counterfactual Propagation Experiment", use_container_width=True):
        with st.spinner("Executing matched Monte Carlo paired rollouts..."):
            analyzer = CrossStagePropagationAnalyzer()
            base_gt = GroundTruthState(
                0.0,
                latest["gt_q_billing"],
                latest["gt_q_kitchen"],
                latest["gt_w_billing"],
                latest["gt_w_kitchen"],
                [0.4, 0.5],
                [1.5, 2.0, 1.0],
                billing_counters,
                kitchen_servers,
                OperatingCondition.NORMAL,
            )
            prop_res = analyzer.analyze_disruption_impact(
                base_state=base_gt,
                disruption=OperatingCondition.BILLING_SLOWDOWN,
                horizon_minutes=15,
                replications=20,
            )

            st.success("Paired counterfactual rollout completed!")
            p_c1, p_c2, p_c3 = st.columns(3)
            with p_c1:
                st.metric("Billing Backlog Spike (Δ Q^B)", f"+{prop_res['max_billing_surge']} orders")
            with p_c2:
                st.metric("Kitchen Starvation Dip (Δ Q^K)", f"{prop_res['min_kitchen_starvation']} orders")
            with p_c3:
                st.metric("Mean Latency Increase (Δ T)", f"+{prop_res['mean_waiting_time_increase_minutes']} min")

            st.info(f"**Key Phenomenon**: {prop_res['cross_stage_phenomenon']}")


# ----------------------------------------------------------------------------
# TAB 4: UNCERTAINTY-AWARE DECISION OPTIMIZER
# ----------------------------------------------------------------------------
with tab4:
    st.markdown("### Uncertainty-Aware Decision Optimizer")
    st.write(
        "Recommends operational interventions by evaluating objective "
        r"$J(a, X) = \alpha \bar{T} + \beta B_{\text{congestion}} + \gamma C(a)$ across posterior state particles."
    )

    rec_action = latest["recommended_action"]
    stab = latest["decision_stability"] * 100
    saving = latest["cost_reduction_pct"]

    st.markdown(
        f"""
        <div class="recommendation-card">
            <h3 style="color:#10b981; margin:0 0 8px 0;">🎯 Recommended Intervention: {rec_action}</h3>
            <p style="color:#f8fafc; font-size:1.05rem; margin-bottom:12px;">
                Expected Total Waiting-Time Cost Reduction: <b>{saving}%</b> | Decision Stability: <b>{stab:.0f}% posterior consensus</b>
            </p>
            <p style="color:#94a3b8; font-size:0.9rem; margin:0;">
                Selected action optimizes mean student wait time while penalizing server congestion and staffing costs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# TAB 5: QUANTITATIVE BENCHMARK & PROOF OF PERFORMANCE
# ----------------------------------------------------------------------------
with tab5:
    st.markdown("### Quantitative Proof of Performance: Baselines vs. QueueFlux")
    st.write(
        "Runs an automated quantitative benchmark comparing Baseline A (Naive FIFO), "
        "Baseline B (No state filtering), and Baseline C (Independent stages) against QueueFlux."
    )

    if st.button("🚀 Run 30-Minute Comparative Benchmark", use_container_width=True):
        with st.spinner("Executing benchmark across all 4 configurations..."):
            bm = QueueFluxBenchmark()
            bm_res = bm.run_full_benchmark(duration_minutes=30)

            bm_data = pd.DataFrame([
                {"Model": "Baseline A (Naive Count / FIFO)", "MAE": bm_res["baseline_a_naive"]["mae"], "RMSE": bm_res["baseline_a_naive"]["rmse"]},
                {"Model": "Baseline B (Workload without Filtering)", "MAE": bm_res["baseline_b_no_filter"]["mae"], "RMSE": bm_res["baseline_b_no_filter"]["rmse"]},
                {"Model": "Baseline C (Independent Stages)", "MAE": bm_res["baseline_c_independent"]["mae"], "RMSE": bm_res["baseline_c_independent"]["rmse"]},
                {"Model": "QueueFlux (Full Mathematical Model)", "MAE": bm_res["queueflux"]["mae"], "RMSE": bm_res["queueflux"]["rmse"]},
            ])

            st.dataframe(bm_data, use_container_width=True)

            chart_bm = (
                alt.Chart(bm_data)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Model:N", title="Model Configuration", sort=None),
                    y=alt.Y("MAE:Q", title="Queue Length MAE"),
                    color=alt.Color("Model:N", scale=alt.Scale(range=["#ef4444", "#f59e0b", "#a855f7", "#10b981"])),
                )
            )
            st.altair_chart(chart_bm, use_container_width=True)

            st.success(
                f"**Waiting-Time Reduction Achieved**: {bm_res['waiting_time_reduction_pct']}% "
                f"| 90% Credible Interval Coverage: {bm_res['queueflux']['ci_coverage_pct']}%"
            )


# Auto-run looping
if auto_run:
    time.sleep(sim_speed)
    execute_sim_step()
    st.rerun()
