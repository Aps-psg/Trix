"""
Publication-Quality Visualization Generator for QueueFlux.
Produces:
1. figure1_state_estimation.png: Ground Truth vs Noisy Sensors vs Estimated State (80% CI).
2. figure2_probabilistic_forecasting.png: 5-Minute Probabilistic Forecast Fan Chart.
3. figure3_cross_stage_propagation.png: Upstream-to-Downstream Shockwave & Lag Dynamics.
4. figure4_benchmark_comparison.png: Multi-Policy Comparative Performance Across Baselines.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reliable server generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from queueflux.config import DEFAULT_CONFIG
from queueflux.models.entities import Customer, OrderType
from queueflux.simulator.generator import TonicSyntheticDataGenerator
from queueflux.simulator.engine import QueueSimulatorEngine
from queueflux.estimator.state_estimator import QueueStateEstimator
from queueflux.predictor.forecaster import FiveMinuteQueueForecaster
from queueflux.evaluation.benchmark import QueueFluxBenchmark


def generate_all_visualizations(output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[QueueFlux Visualizer] Generating performance plots in '{output_dir}'...")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    config = DEFAULT_CONFIG

    # --- Run Single Disruption Day Simulation to Collect High-Resolution Time Series ---
    generator = TonicSyntheticDataGenerator(config=config, seed=42)
    engine = QueueSimulatorEngine(config=config, seed=42)
    estimator = QueueStateEstimator(num_particles=400, config=config, seed=42)
    forecaster = FiveMinuteQueueForecaster(config=config, num_rollout_samples=200, seed=42)

    day_df, disruptions = generator.generate_day(day_index=3)  # Day 4: Billing Slowdown Disruption
    minute_groups = {}
    for _, row in day_df.iterrows():
        m = int(row["arrival_minute"])
        cust = Customer(
            customer_id=row["customer_id"],
            arrival_time=row["arrival_minute"],
            order_type=OrderType(row["order_type"]),
            expected_billing_duration=row["expected_billing_duration"],
            actual_billing_duration=row["sampled_billing_duration"],
            patience_billing=row["patience_billing"],
            patience_kitchen=row["patience_kitchen"]
        )
        minute_groups.setdefault(m, []).append(cust)

    time_pts = []
    gt_qb = []
    gt_qk = []
    gt_wb = []
    gt_wk = []
    obs_qb = []
    est_qb = []
    est_qb_p10 = []
    est_qb_p90 = []

    # Store a specific peak time forecast snapshot
    snapshot_t = 285
    snapshot_forecast = None

    for t in range(config.minutes_per_day):
        arrs = minute_groups.get(t, [])
        state_gt, obs = engine.step_minute(arrs, disruptions)
        est_state = estimator.update(obs)

        time_pts.append(t)
        gt_qb.append(state_gt.Q_B)
        gt_qk.append(state_gt.Q_K)
        gt_wb.append(state_gt.W_B)
        gt_wk.append(state_gt.W_K)
        obs_qb.append(obs.y_Q_B)
        est_qb.append(est_state.Q_B_hat)
        est_qb_p10.append(est_state.Q_B_p10)
        est_qb_p90.append(est_state.Q_B_p90)

        if t == snapshot_t:
            snapshot_forecast = forecaster.forecast_5min(est_state, t)

    time_hrs = np.array(time_pts) / 60.0 + 8.0  # Convert to clock hour (08:00 to 20:00)

    # =========================================================================
    # Figure 1: State Estimation (Ground Truth vs Sensor vs Particle Filter)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    # Focus around lunch rush (11:30 to 14:30 -> hours 11.5 to 14.5)
    mask = (time_hrs >= 11.5) & (time_hrs <= 14.5)
    t_sub = time_hrs[mask]

    ax.plot(t_sub, np.array(gt_qb)[mask], label="Ground Truth $Q_t^B$", color="#1f77b4", linewidth=2.2)
    ax.scatter(t_sub, np.array(obs_qb)[mask], label="Noisy Sensor Obs $y_t^B$", color="#7f7f7f", alpha=0.45, s=18)
    ax.plot(t_sub, np.array(est_qb)[mask], label="QueueFlux Estimate $\\hat{Q}_t^B$", color="#d62728", linestyle="--", linewidth=2.0)
    ax.fill_between(t_sub, np.array(est_qb_p10)[mask], np.array(est_qb_p90)[mask], 
                    color="#d62728", alpha=0.18, label="Credible Interval [P10, P90]")

    # Mark disruption window (270 to 315 min -> 12:30 to 13:15 -> hrs 12.5 to 13.25)
    ax.axvspan(12.5, 13.25, color="orange", alpha=0.15, label="Active Disruption: Billing POS Latency")
    ax.set_title("QueueFlux State Estimation under Incomplete/Noisy Observations (Lunch Rush & Disruption)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time of Day (Hours)", fontsize=11)
    ax.set_ylabel("Billing Queue Length ($Q_t^B$)", fontsize=11)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "figure1_state_estimation.png")
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f"  Saved: {fig1_path}")

    # =========================================================================
    # Figure 2: 5-Minute Probabilistic Forecast Fan Chart
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    if snapshot_forecast:
        horizons = [0] + [fc.horizon_minutes for fc in snapshot_forecast]
        q_med = [est_qb[snapshot_t]] + [fc.Q_B_median for fc in snapshot_forecast]
        q_p10 = [est_qb[snapshot_t]] + [fc.Q_B_p10 for fc in snapshot_forecast]
        q_p90 = [est_qb[snapshot_t]] + [fc.Q_B_p90 for fc in snapshot_forecast]

        # Actual trajectory realized by simulator
        actual_realized = [gt_qb[snapshot_t]] + [gt_qb[snapshot_t + h] for h in [1, 2, 3, 4, 5]]

        ax.plot(horizons, q_med, label="Forecast Median (P50)", color="#2ca02c", marker="o", linewidth=2.2)
        ax.fill_between(horizons, q_p10, q_p90, color="#2ca02c", alpha=0.25, label="Uncertainty Fan [P10, P90]")
        ax.plot(horizons, actual_realized, label="Actual Realized State $Q_{t+k}^B$", color="#000000", marker="x", linestyle=":", linewidth=2.0)

        ax.set_title("QueueFlux 5-Minute Forward Probabilistic Queue Forecast (t = 12:45 PM Peak)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Forecast Horizon $k$ (Minutes Ahead)", fontsize=11)
        ax.set_ylabel("Projected Billing Queue $Q_{t+k}^B$", fontsize=11)
        ax.set_xticks(horizons)
        ax.set_xticklabels([f"t (Current)" if h == 0 else f"t+{h}m" for h in horizons])
        ax.legend(loc="upper left", frameon=True, fontsize=10)
        plt.tight_layout()
        fig2_path = os.path.join(output_dir, "figure2_probabilistic_forecasting.png")
        fig.savefig(fig2_path)
        plt.close(fig)
        print(f"  Saved: {fig2_path}")

    # =========================================================================
    # Figure 3: Cross-Stage Disruption Propagation Dynamics
    # =========================================================================
    fig, ax1 = plt.subplots(figsize=(12, 5), dpi=300)
    ax2 = ax1.twinx()

    ax1.plot(t_sub, np.array(gt_wb)[mask], color="#d62728", linewidth=2.2, label="Stage A: Billing Workload $W_t^B$ (mins)")
    ax2.plot(t_sub, np.array(gt_wk)[mask], color="#1f77b4", linewidth=2.2, linestyle="--", label="Stage B: Kitchen Workload $W_t^K$ (mins)")

    ax1.axvspan(12.5, 13.25, color="orange", alpha=0.15, label="Stage A Disruption")
    ax1.set_title("Cross-Stage Disruption Propagation: Upstream Shockwave to Downstream Delay Surge", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Time of Day (Hours)", fontsize=11)
    ax1.set_ylabel("Billing Backlog $W_t^B$ (Service-Minutes)", color="#d62728", fontsize=11)
    ax2.set_ylabel("Kitchen Backlog $W_t^K$ (Service-Minutes)", color="#1f77b4", fontsize=11)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, fontsize=10)
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "figure3_cross_stage_propagation.png")
    fig.savefig(fig3_path)
    plt.close(fig)
    print(f"  Saved: {fig3_path}")

    # =========================================================================
    # Figure 4: Policy Benchmark Comparison
    # =========================================================================
    print("  Running multi-policy benchmark for Figure 4...")
    benchmark = QueueFluxBenchmark(config=config, seed=42)
    bench_results = benchmark.run_benchmark_on_day(day_df, disruptions)

    policies = ["Baseline_Naive", "Baseline_MMC", "Baseline_Static_Heuristic", "QueueFlux_Full"]
    labels = ["Naive FIFO", "M/M/c Erlang-C", "Static Heuristic", "QueueFlux (Ours)"]
    colors = ["#7f7f7f", "#ff7f0e", "#1f77b4", "#2ca02c"]

    avg_waits = [bench_results[p].avg_customer_wait_minutes for p in policies]
    reneges = [bench_results[p].reneging_rate_pct for p in policies]
    pred_mae = [bench_results[p].waiting_time_mae for p in policies]
    fc_crps = [bench_results[p].forecast_crps for p in policies]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8), dpi=300)

    # Panel A: Average Wait Time
    bars1 = ax1.bar(labels, avg_waits, color=colors, alpha=0.85, edgecolor="black")
    ax1.set_title("Average Customer Wait Time (Minutes)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Minutes")
    for b in bars1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{b.get_height():.1f}m", ha="center", fontsize=9)

    # Panel B: Reneging Rate
    bars2 = ax2.bar(labels, reneges, color=colors, alpha=0.85, edgecolor="black")
    ax2.set_title("Customer Reneging / Cancellation Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Reneging Rate (%)")
    for b in bars2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{b.get_height():.1f}%", ha="center", fontsize=9)

    # Panel C: Waiting Time Prediction MAE
    bars3 = ax3.bar(labels, pred_mae, color=colors, alpha=0.85, edgecolor="black")
    ax3.set_title("Waiting Time Prediction MAE (Minutes)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("MAE (Minutes)")
    for b in bars3:
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{b.get_height():.2f}m", ha="center", fontsize=9)

    # Panel D: 5-Minute Forecast CRPS
    bars4 = ax4.bar(labels, fc_crps, color=colors, alpha=0.85, edgecolor="black")
    ax4.set_title("5-Minute Forecast Error (CRPS - Lower is Better)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("CRPS")
    for b in bars4:
        ax4.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}", ha="center", fontsize=9)

    plt.suptitle("QueueFlux Comprehensive Benchmark vs. Classical Baselines", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "figure4_benchmark_comparison.png")
    fig.savefig(fig4_path)
    plt.close(fig)
    print(f"  Saved: {fig4_path}")
    print("[QueueFlux Visualizer] All 4 publication-quality figures successfully generated.")


if __name__ == "__main__":
    generate_all_visualizations()
