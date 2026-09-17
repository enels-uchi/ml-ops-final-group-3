"""Generate presentation-ready PNGs for the monitoring + data-drift slides (10 & 11).

Outputs (16:9, high-DPI, transparent-friendly white bg) to presentation/:
  * slide10_monitoring_flow.png    - serving + monitoring architecture + baseline-validation result
  * slide11_drift_results.png      - drift simulation results across the 4 scenarios
  * slide12_grafana_prometheus.png - live Prometheus + Grafana stack, dashboard KPIs, alerts

Run: venv/bin/python presentation/make_slide_assets.py   (from the repo root)
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = "presentation"
BLUE, GREEN, RED, PURPLE, GRAY, INK = "#3B7DD8", "#4C9F70", "#D9534F", "#7E63B8", "#8A8A8A", "#222222"
AMBER = "#E0932F"


def _box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.5, edgecolor=color, facecolor=color + "22"))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=INK)
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=9.5, color="#444")


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.6, color=GRAY))


def slide10():
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.75); ax.axis("off")

    ax.text(0.3, 6.35, "Production Monitoring — Serving + Evidently + Prometheus",
            fontsize=17, fontweight="bold", color=INK)

    # top pipeline row
    y, h = 4.3, 1.2
    _box(ax, 0.3, y, 2.5, h, "Clean test data", "13,998 rows (golden set)", GRAY)
    _box(ax, 3.4, y, 2.7, h, "FastAPI service", "/predict  /health  /metrics", BLUE)
    _box(ax, 6.7, y, 2.5, h, "EvidentlyAI", "drift + model-quality report", PURPLE)
    _box(ax, 9.5, y, 2.2, h, "Prometheus", "/metrics live gauges", GREEN)
    _arrow(ax, 2.8, y + h / 2, 3.4, y + h / 2)
    _arrow(ax, 6.1, y + h / 2, 6.7, y + h / 2)
    _arrow(ax, 9.2, y + h / 2, 9.5, y + h / 2)

    # baseline validation result panel
    ax.add_patch(FancyBboxPatch((0.3, 0.5), 11.4, 3.0, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.5, edgecolor=GREEN, facecolor=GREEN + "14"))
    ax.text(0.7, 3.05, "Baseline validation  —  live API reproduces training exactly, zero drift on clean data",
            fontsize=13, fontweight="bold", color=INK)
    metrics = [("F1", "0.436"), ("AUC-ROC", "0.648"), ("Precision", "0.563"), ("Recall", "0.355")]
    for i, (k, v) in enumerate(metrics):
        cx = 1.7 + i * 2.5
        ax.text(cx, 2.25, v, ha="center", fontsize=22, fontweight="bold", color=GREEN)
        ax.text(cx, 1.75, k, ha="center", fontsize=11, color="#444")
        ax.text(cx, 1.35, "live = recorded", ha="center", fontsize=8.5, color=GRAY)
    ax.text(0.7, 0.8, "Data drift vs. training reference:  0 / 18 columns drifted  →  healthy reference established",
            fontsize=11, color=INK)

    fig.tight_layout()
    p = os.path.join(OUT_DIR, "slide10_monitoring_flow.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def slide11():
    scenarios = ["out_of_bounds", "column_swap", "schema_change", "dist_shift"]
    pos = [0.004, 0.255, 0.218, 0.953]
    drifted = ["5 / 18", "4 / 18", "3 / 18", "5 / 18"]
    desc = ["out-of-bounds values", "swapped columns",
            "unseen categories", "shifted distribution"]
    baseline = 0.257

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.75), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle("Data Drift — Simulation & Detection  (all 4 scenarios caught + alerted)",
                 fontsize=16, fontweight="bold", color=INK, x=0.5, y=0.98)

    # left: predicted-positive ratio per scenario vs baseline
    bars = ax1.bar(scenarios, pos, color=RED, width=0.6, zorder=3)
    ax1.axhline(baseline, ls="--", color=GRAY, lw=1.5, zorder=2)
    ax1.text(-0.4, baseline + 0.025, f"clean baseline {baseline:.3f}", ha="left", fontsize=9, color=GRAY)
    for b, v in zip(bars, pos):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Predicted-positive ratio", fontsize=11)
    ax1.set_title("Model output shifts under drift", fontsize=12)
    ax1.tick_params(axis="x", rotation=20, labelsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # right: results table
    ax2.axis("off")
    ax2.set_title("What the monitor caught", fontsize=12, y=0.98)
    rows = [["Scenario", "Injected", "Drifted", "Alert"]]
    for s, d, dr in zip(scenarios, desc, drifted):
        rows.append([s, d.replace("\n", " "), dr, "FIRED"])
    tbl = ax2.table(cellText=rows[1:], colLabels=rows[0], cellLoc="left", loc="center",
                    colWidths=[0.30, 0.37, 0.18, 0.15])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r == 0:
            cell.set_facecolor(PURPLE + "22"); cell.set_text_props(fontweight="bold")
        if c == 3 and r > 0:
            cell.set_text_props(color=RED, fontweight="bold")
    ax2.text(0.0, 0.02,
             "Detection: Wasserstein (numeric) / Jensen-Shannon (categorical) distance per column\n"
             "vs. clean reference, threshold 0.1. Key insight: input drift catches pipeline bugs before\n"
             "accuracy drops — column_swap moved 4 columns while predictions barely changed (0.255 vs 0.257).",
             transform=ax2.transAxes, fontsize=8.6, color="#444", va="bottom")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT_DIR, "slide11_drift_results.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def slide12():
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.75); ax.axis("off")
    ax.text(0.3, 6.4, "Live Monitoring — Prometheus + Grafana", fontsize=17, fontweight="bold", color=INK)
    ax.text(0.3, 6.02, "Orchestrated with Docker Compose:  docker compose up  →  api :8000 · prometheus :9090 · grafana :3000",
            fontsize=10, color="#555")

    # architecture flow
    y, h = 4.55, 1.1
    _box(ax, 0.3, y, 2.9, h, "FastAPI  /metrics", "7 gauges + counters", BLUE)
    _box(ax, 3.7, y, 3.0, h, "Prometheus", "scrape /metrics @ 5s\n+ alert rules", AMBER)
    _box(ax, 7.2, y, 2.6, h, "Grafana", "live dashboard", GREEN)
    _arrow(ax, 3.2, y + h / 2, 3.7, y + h / 2)
    _arrow(ax, 6.7, y + h / 2, 7.2, y + h / 2)
    ax.text(8.5 + 1.4, y + h / 2, "", ha="center")

    # left panel: live dashboard KPIs (dist_shift drift state)
    ax.add_patch(FancyBboxPatch((0.3, 0.5), 6.4, 3.5, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.5, edgecolor=GRAY, facecolor="#F4F4F4"))
    ax.text(0.6, 3.6, "Grafana dashboard  —  live during drift (dist_shift)", fontsize=12, fontweight="bold", color=INK)
    # big status
    ax.add_patch(FancyBboxPatch((0.6, 2.55), 2.7, 0.8, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=0, facecolor=RED))
    ax.text(1.95, 2.95, "DRIFT DETECTED", ha="center", va="center", fontsize=13, fontweight="bold", color="white")
    kpis = [("Predicted-positive", "92.1%", RED), ("vs clean baseline", "24.4%", GRAY),
            ("Drifted columns", "5 / 18", INK), ("Drift share", "27.8%", INK)]
    for i, (k, v, c) in enumerate(kpis):
        cx = 4.0 + (i % 2) * 1.45
        cy = 2.95 - (i // 2) * 0.85
        ax.text(cx, cy, v, ha="center", fontsize=15, fontweight="bold", color=c)
        ax.text(cx, cy - 0.32, k, ha="center", fontsize=8, color="#555")
    ax.text(0.6, 1.55, "Panels: drift status · Evidently verdict · drifted columns · drift share ·\n"
                       "predicted-positive vs baseline · input-drift signals · throughput · p50/p95 latency",
            fontsize=8.6, color="#444", va="top")
    ax.text(0.6, 0.72, "Served model: catboost-champion-1.0.0   ·   140K rows scored", fontsize=8.6, color="#555")

    # right panel: alert rules
    ax.add_patch(FancyBboxPatch((7.0, 0.5), 4.7, 3.5, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.5, edgecolor=AMBER, facecolor="#FBF3E6"))
    ax.text(7.3, 3.6, "Prometheus alert rules", fontsize=12, fontweight="bold", color=INK)
    alerts = [
        ("DriftedColumnsDetected", "≥ 3 cols drift", "warning", True),
        ("PredictionDistributionShift", "pred+ outside [0.15, 0.40]", "critical", True),
        ("DatasetDriftDetected", "≥ 50% cols drift", "critical", False),
        ("InferenceAPIDown", "scrape fails 15s", "critical", False),
        ("HighPredictionLatency", "p95 > 2s", "warning", False),
    ]
    yy = 3.15
    for name, cond, sev, fired in alerts:
        dot = RED if fired else "#BBBBBB"
        ax.text(7.35, yy, "●", fontsize=11, color=dot, va="center")
        ax.text(7.62, yy + 0.06, name, fontsize=9.5, fontweight="bold", color=INK, va="center")
        ax.text(7.62, yy - 0.19, f"{cond}  ·  {sev}", fontsize=8, color="#666", va="center")
        yy -= 0.55
    ax.text(7.3, 0.62, "● fired on dist_shift    ● armed", fontsize=8, color="#666")

    fig.tight_layout()
    p = os.path.join(OUT_DIR, "slide12_grafana_prometheus.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    print("[slides]", slide10())
    print("[slides]", slide11())
    print("[slides]", slide12())


if __name__ == "__main__":
    main()
