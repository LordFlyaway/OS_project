"""Extra ablations and naive baselines used in the novelty assessment.

Run from the project root:   python experiments/run_extra_baselines.py
Writes experiments/results/e6_ablation_extra.csv  (200 batches x 20 processes).

Question answered: where does the AI scheduler's gain actually come from?
"""

from __future__ import annotations

import copy
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generation import generate_processes  # noqa: E402
from src.prediction import Predictor  # noqa: E402
from src.schedulers import _run_non_preemptive, ai_scheduler, fcfs, sjf  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e6_ablation_extra.csv")
N_BATCHES, BATCH_SIZE, SEED0 = 200, 20, 1000

def type_means() -> dict:
    """Mean realised burst per process type, measured on 300 separate batches (seeds 50000+)."""
    procs = [p for k in range(300) for p in generate_processes(BATCH_SIZE, seed=50000 + k)]
    return {t: float(np.mean([p.actual_burst_time for p in procs if p.proc_type == t])) for t in ("IO-bound", "CPU-bound")}


def main() -> None:
    predictor = Predictor.load()
    TYPE_MEANS = type_means()
    rows = []
    for k in range(N_BATCHES):
        procs = generate_processes(BATCH_SIZE, seed=SEED0 + k)
        predictor.annotate(procs)

        def run(key):
            return _run_non_preemptive(copy.deepcopy(procs), "x", key=key).avg_waiting_time

        rows.append({
            "FCFS": fcfs(copy.deepcopy(procs)).avg_waiting_time,
            "SJF oracle": sjf(copy.deepcopy(procs)).avg_waiting_time,
            "AI (full: burst+0.5*prio-aging)": ai_scheduler(copy.deepcopy(procs)).avg_waiting_time,
            "AI, no priority term": run(lambda p, t: p.predicted_burst_time - 0.35 * (t - p.arrival_time)),
            "AI, no aging": run(lambda p, t: p.predicted_burst_time + 0.5 * p.predicted_priority),
            "Predicted-burst SJF only": run(lambda p, t: p.predicted_burst_time),
            "Naive SJF: burst = past_cpu_usage": run(lambda p, t: p.past_cpu_usage),
            "Naive SJF: burst = type mean": run(lambda p, t: TYPE_MEANS[p.proc_type]),
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    fcfs_w, sjf_w = df["FCFS"].mean(), df["SJF oracle"].mean()
    summary = pd.DataFrame({
        "mean_avg_wait": df.mean().round(2),
        "std": df.std().round(2),
        "gap_closed_%": ((fcfs_w - df.mean()) / (fcfs_w - sjf_w) * 100).round(1),
    })
    print(summary.to_string())


if __name__ == "__main__":
    main()
