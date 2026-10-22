"""How does the AI scheduler hold up as the data gets noisier?

Run from the project root:   python experiments/run_noise_sweep.py
Writes experiments/results/e13_noise_sweep.csv and e13_noise_sweep.png

For each noise preset (low, medium, high, extreme) the models are trained on data from that
preset and evaluated on 200 fresh batches of 20 processes drawn from the same preset.
SJF and Priority are the oracle baselines (they are told the realised burst / priority).
The noise levels are assumptions, not measurements of any real system.
"""

from __future__ import annotations

import copy
import os
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generation import NOISE_PRESETS, generate_processes  # noqa: E402
from src.prediction import Predictor, train_models  # noqa: E402
from src.schedulers import (_run_non_preemptive, ai_scheduler, fcfs,  # noqa: E402
                            priority_scheduling, round_robin, sjf)

RES = os.path.join(os.path.dirname(__file__), "results")
N_BATCHES, BATCH = 200, 20


def main() -> None:
    os.makedirs(RES, exist_ok=True)
    rows = []
    for level in NOISE_PRESETS:
        reg, clf, rep = train_models(4000, 42, save=False, noise=level)
        pred = Predictor(reg, clf)
        w = {k: [] for k in ("FCFS", "SJF", "Round Robin", "Priority", "AI", "AI no priority/aging", "max_AI", "max_SJF")}
        err = []
        for b in range(N_BATCHES):
            procs = generate_processes(BATCH, seed=1000 + b, noise=level)
            pred.annotate(procs)
            err.append(np.mean([abs(p.predicted_burst_time - p.actual_burst_time) for p in procs]))
            c = lambda: copy.deepcopy(procs)  # noqa: E731
            w["FCFS"].append(fcfs(c()).avg_waiting_time)
            s = sjf(c())
            w["SJF"].append(s.avg_waiting_time)
            w["max_SJF"].append(max(p.waiting_time for p in s.processes))
            w["Round Robin"].append(round_robin(c(), quantum=4).avg_waiting_time)
            w["Priority"].append(priority_scheduling(c()).avg_waiting_time)
            a = ai_scheduler(c())
            w["AI"].append(a.avg_waiting_time)
            w["max_AI"].append(max(p.waiting_time for p in a.processes))
            w["AI no priority/aging"].append(_run_non_preemptive(c(), "x", key=lambda p, t: p.predicted_burst_time).avg_waiting_time)
        m = {k: float(np.mean(v)) for k, v in w.items()}
        gap = (m["FCFS"] - m["AI"]) / (m["FCFS"] - m["SJF"]) * 100
        gap_plain = (m["FCFS"] - m["AI no priority/aging"]) / (m["FCFS"] - m["SJF"]) * 100
        rows.append({"noise": level, "burst_MAE": rep.burst_mae, "burst_R2": rep.burst_r2,
                     "priority_acc": rep.priority_accuracy, "priority_F1": rep.priority_f1,
                     "batch_pred_MAE": float(np.mean(err)), **{f"wait_{k}": v for k, v in m.items()},
                     "AI_vs_FCFS_%": (m["FCFS"] - m["AI"]) / m["FCFS"] * 100,
                     "gap_closed_%": gap, "gap_closed_plain_SJF_%": gap_plain,
                     "AI_beats_FCFS_%": float(np.mean(np.array(w["AI"]) < np.array(w["FCFS"])) * 100),
                     "AI_beats_Priority_%": float(np.mean(np.array(w["AI"]) < np.array(w["Priority"])) * 100)})
        print(level, {k: round(v, 2) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "e13_noise_sweep.csv"), index=False)

    INK, PANEL, TEXT, MUTED, AMBER = "#0b0f1a", "#121a2b", "#e7ecf7", "#8793b0", "#ffb224"
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#2a3858"})
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), facecolor=INK)
    for a in ax:
        a.set_facecolor(PANEL); a.grid(alpha=.18)
    x = range(len(df))
    ax[0].plot(x, df["wait_FCFS"], "o-", color="#ff7a7a", label="FCFS")
    ax[0].plot(x, df["wait_Priority"], "o-", color="#8fa3ff", label="Priority (oracle)")
    ax[0].plot(x, df["wait_AI"], "o-", color=AMBER, linewidth=3, label="AI scheduler")
    ax[0].plot(x, df["wait_SJF"], "o-", color="#5fd6a0", label="SJF (oracle)")
    ax[0].set_ylabel("average waiting time (ms)"); ax[0].legend(facecolor=PANEL, edgecolor="#2a3858", labelcolor=TEXT, fontsize=9)
    ax[0].set_title("Waiting time by noise level", color=TEXT, loc="left", fontsize=11)
    ax[1].plot(x, df["gap_closed_%"], "o-", color=AMBER, linewidth=3, label="full AI scheduler")
    ax[1].plot(x, df["gap_closed_plain_SJF_%"], "s--", color=MUTED, label="predicted-burst SJF only")
    ax[1].axhline(0, color=MUTED, linewidth=.8); ax[1].set_ylabel("% of the FCFS → SJF gap closed")
    ax[1].legend(facecolor=PANEL, edgecolor="#2a3858", labelcolor=TEXT, fontsize=9)
    ax[1].set_title("How much of the possible gain survives", color=TEXT, loc="left", fontsize=11)
    ax[2].plot(x, df["burst_MAE"], "o-", color="#7ee0e8", label="burst MAE (ms)")
    ax[2].set_ylabel("burst-time error, MAE (ms)", color="#7ee0e8")
    t = ax[2].twinx(); t.plot(x, df["priority_acc"] * 100, "s--", color="#ff8ab3", label="priority accuracy (%)")
    t.set_ylabel("priority accuracy (%)", color="#ff8ab3"); t.tick_params(colors="#ff8ab3")
    ax[2].set_title("Model quality by noise level", color=TEXT, loc="left", fontsize=11)
    for a in ax:
        a.set_xticks(list(x)); a.set_xticklabels(df["noise"])
        a.set_xlabel("noise preset")
    fig.suptitle("AI scheduler vs data noise (models retrained at each level, 200 batches)", color=TEXT, fontsize=13, x=.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(os.path.join(RES, "e13_noise_sweep.png"), dpi=160, facecolor=INK)
    print("saved")


if __name__ == "__main__":
    main()
