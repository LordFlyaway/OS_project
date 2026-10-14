"""Module 5 - Evaluation & Visualisation.

Collects the ScheduleResults, builds a metrics table, and renders:
  * a grouped bar chart comparing waiting / turnaround / response time
  * one Gantt chart per algorithm
All figures are written to ai_task_scheduler/outputs/.
"""

from __future__ import annotations

import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless - just write files
import matplotlib.pyplot as plt  # noqa: E402

from .schedulers import ScheduleResult  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

# Distinct colour per PID so the same process is the same colour in every Gantt.
_PALETTE = plt.get_cmap("tab20").colors


def metrics_table(results: list[ScheduleResult]) -> pd.DataFrame:
    df = pd.DataFrame(r.metrics() for r in results)
    return df.set_index("algorithm")


def print_report(results: list[ScheduleResult]) -> pd.DataFrame:
    df = metrics_table(results)
    print("\n================  SCHEDULING COMPARISON  ================\n")
    print(df.to_string())
    best = df["avg_waiting"].idxmin()
    sjf_wait = df.loc["SJF", "avg_waiting"] if "SJF" in df.index else None
    ai_wait = df.loc["AI Scheduler", "avg_waiting"] if "AI Scheduler" in df.index else None
    print(f"\nLowest average waiting time : {best} ({df.loc[best, 'avg_waiting']} ms)")
    if sjf_wait is not None and ai_wait is not None:
        gap = ai_wait - sjf_wait
        fcfs_wait = df.loc["FCFS", "avg_waiting"] if "FCFS" in df.index else ai_wait
        closed = 100 * (fcfs_wait - ai_wait) / (fcfs_wait - sjf_wait) if fcfs_wait != sjf_wait else 0.0
        print(f"AI vs SJF (oracle) gap     : {gap:+.2f} ms  "
              f"(AI closes {closed:.0f}% of the FCFS->SJF gap)")
    print("\n========================================================\n")
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_metric_bars(results: list[ScheduleResult],
                     path: str | None = None) -> str:
    df = metrics_table(results)
    metrics = ["avg_waiting", "avg_turnaround", "avg_response"]
    labels = ["Avg waiting", "Avg turnaround", "Avg response"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = range(len(df))
    width = 0.26
    for k, (m, lbl) in enumerate(zip(metrics, labels)):
        ax.bar([i + (k - 1) * width for i in x], df[m], width, label=lbl)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df.index, rotation=15, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title("CPU scheduling algorithms - lower is better")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = path or os.path.join(OUTPUT_DIR, "metrics_comparison.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_gantt(results: list[ScheduleResult],
               path: str | None = None) -> str:
    fig, axes = plt.subplots(len(results), 1, figsize=(11, 2.1 * len(results)),
                             squeeze=False)
    for ax, r in zip(axes[:, 0], results):
        for seg in r.gantt:
            colour = _PALETTE[(seg.pid - 1) % len(_PALETTE)]
            ax.barh(0, seg.duration, left=seg.start, height=0.6,
                    color=colour, edgecolor="black", linewidth=0.5)
            if seg.duration >= 2:
                ax.text(seg.start + seg.duration / 2, 0, f"P{seg.pid}",
                        ha="center", va="center", fontsize=8)
        ax.set_yticks([])
        ax.set_xlim(0, max(r.makespan for r in results) + 1)
        ax.set_title(f"{r.algorithm}   "
                     f"(avg wait {r.avg_waiting_time:.1f} ms, "
                     f"{r.context_switches} context switches)",
                     fontsize=10, loc="left")
        ax.grid(axis="x", alpha=0.3)
    axes[-1, 0].set_xlabel("Time (ms)")
    fig.tight_layout()

    path = path or os.path.join(OUTPUT_DIR, "gantt_charts.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_all(results: list[ScheduleResult]) -> list[str]:
    return [plot_metric_bars(results), plot_gantt(results)]
