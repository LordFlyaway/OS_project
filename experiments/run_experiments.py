"""Reproducible experiments used in the final report.

Run from the project root:   python experiments/run_experiments.py
Writes CSV files to experiments/results/.
"""

from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generation import generate_processes  # noqa: E402
from src.prediction import Predictor, compare_models, train_models  # noqa: E402
from src.schedulers import (ai_scheduler, fcfs, priority_scheduling,  # noqa: E402
                            round_robin, sjf)

OUT = os.path.join(os.path.dirname(__file__), "results")
N_BATCHES = 200
BATCH_SIZE = 20
SEED0 = 1000          # batch seeds 1000.. never overlap the training seed (42)
QUANTUM = 4


def max_wait(result) -> int:
    return max(p.waiting_time for p in result.processes)


def run_algorithms(procs, quantum=QUANTUM, aging=0.35):
    c = copy.deepcopy
    return {
        "FCFS": fcfs(c(procs)),
        "SJF": sjf(c(procs)),
        f"Round Robin (q={quantum})": round_robin(c(procs), quantum=quantum),
        "Priority": priority_scheduling(c(procs)),
        "AI Scheduler": ai_scheduler(c(procs), aging_factor=aging),
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    # ---- E1: model quality -------------------------------------------------
    reg, clf, report = train_models(n_rows=4000, seed=42, save=True)
    pd.DataFrame([report.__dict__]).to_csv(os.path.join(OUT, "e1_model_report.csv"), index=False)
    compare_models().to_csv(os.path.join(OUT, "e1_model_comparison.csv"), index=False)
    predictor = Predictor(reg, clf)
    print(report)

    # ---- E2: 200 random batches -------------------------------------------
    rows, per_batch = [], []
    mae_batches = []
    for k in range(N_BATCHES):
        procs = generate_processes(BATCH_SIZE, seed=SEED0 + k)
        predictor.annotate(procs)
        mae_batches.append(np.mean([abs(p.predicted_burst_time - p.actual_burst_time) for p in procs]))
        for name, r in run_algorithms(procs).items():
            rows.append({"batch": k, "algorithm": name,
                         "avg_waiting": r.avg_waiting_time,
                         "avg_turnaround": r.avg_turnaround_time,
                         "avg_response": r.avg_response_time,
                         "max_waiting": max_wait(r),
                         "context_switches": r.context_switches,
                         "cpu_util": r.cpu_utilisation})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e2_all_batches.csv"), index=False)
    summary = df.groupby("algorithm", sort=False).agg(
        avg_waiting=("avg_waiting", "mean"), waiting_std=("avg_waiting", "std"),
        avg_turnaround=("avg_turnaround", "mean"),
        avg_response=("avg_response", "mean"),
        max_waiting=("max_waiting", "mean"),
        context_switches=("context_switches", "mean"),
        cpu_util=("cpu_util", "mean")).round(2)
    summary.to_csv(os.path.join(OUT, "e2_summary.csv"))
    print("\nE2 summary (200 batches x 20 processes)\n", summary.to_string())

    wide = df.pivot(index="batch", columns="algorithm", values="avg_waiting")
    closed = (wide["FCFS"] - wide["AI Scheduler"]) / (wide["FCFS"] - wide["SJF"]).replace(0, np.nan)
    extra = {
        "batch_prediction_MAE_ms": float(np.mean(mae_batches)),
        "AI_beats_FCFS_pct": float((wide["AI Scheduler"] < wide["FCFS"]).mean() * 100),
        "AI_beats_Priority_pct": float((wide["AI Scheduler"] < wide["Priority"]).mean() * 100),
        "AI_beats_RR_pct": float((wide["AI Scheduler"] < wide[f"Round Robin (q={QUANTUM})"]).mean() * 100),
        "AI_matches_or_beats_SJF_pct": float((wide["AI Scheduler"] <= wide["SJF"] + 1e-9).mean() * 100),
        "gap_closed_mean_pct": float(closed.mean() * 100),
        "gap_closed_median_pct": float(closed.median() * 100),
        "AI_minus_SJF_mean_ms": float((wide["AI Scheduler"] - wide["SJF"]).mean()),
    }
    pd.DataFrame([extra]).to_csv(os.path.join(OUT, "e2_extra.csv"), index=False)
    print("\nE2 extra\n", pd.Series(extra).round(2).to_string())

    # ---- E3: aging ablation ------------------------------------------------
    rows = []
    for aging in (0.0, 0.1, 0.35, 0.7, 1.0, 2.0):
        for k in range(N_BATCHES):
            procs = generate_processes(BATCH_SIZE, seed=SEED0 + k)
            predictor.annotate(procs)
            r = ai_scheduler(copy.deepcopy(procs), aging_factor=aging)
            rows.append({"aging": aging, "avg_waiting": r.avg_waiting_time, "max_waiting": max_wait(r)})
    abl = pd.DataFrame(rows).groupby("aging").mean().round(2)
    abl.to_csv(os.path.join(OUT, "e3_aging_ablation.csv"))
    print("\nE3 aging ablation\n", abl.to_string())

    # ---- E4: Round Robin quantum sensitivity ------------------------------
    rows = []
    for q in (1, 2, 4, 8, 16, 32):
        for k in range(N_BATCHES):
            procs = generate_processes(BATCH_SIZE, seed=SEED0 + k)
            r = round_robin(copy.deepcopy(procs), quantum=q)
            rows.append({"quantum": q, "avg_waiting": r.avg_waiting_time,
                         "avg_response": r.avg_response_time,
                         "context_switches": r.context_switches})
    rr = pd.DataFrame(rows).groupby("quantum").mean().round(2)
    rr.to_csv(os.path.join(OUT, "e4_rr_quantum.csv"))
    print("\nE4 Round Robin quantum\n", rr.to_string())

    # ---- E5: robustness to prediction error --------------------------------
    rows = []
    rng = np.random.default_rng(0)
    for noise in (0, 5, 10, 20, 40):
        for k in range(N_BATCHES):
            procs = generate_processes(BATCH_SIZE, seed=SEED0 + k)
            predictor.annotate(procs)
            for p in procs:
                p.predicted_burst_time = max(1.0, p.predicted_burst_time + rng.normal(0, noise))
            r = ai_scheduler(copy.deepcopy(procs))
            rows.append({"extra_noise_sd_ms": noise, "avg_waiting": r.avg_waiting_time})
    nz = pd.DataFrame(rows).groupby("extra_noise_sd_ms").mean().round(2)
    nz.to_csv(os.path.join(OUT, "e5_prediction_noise.csv"))
    print("\nE5 added prediction noise\n", nz.to_string())


if __name__ == "__main__":
    main()
