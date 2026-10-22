"""How the model-comparison table was produced, and a matplotlib chart of it.

Run from the project root:
    python experiments/run_nonlinear_dataset_benchmark.py            # saves the CSV + PNG
    python experiments/run_nonlinear_dataset_benchmark.py --show     # also opens the chart window
Needs one extra package for the XGBoost row:   pip install xgboost

WHAT IT DOES, STEP BY STEP
  1. Generate a synthetic, NON-LINEAR dataset of processes with four types (CPU-bound, I/O-bound,
     memory-intensive, interactive) using the project's own generator (src/data_generation.py,
     dataset="nonlinear4", noise="low"). The burst-time formula has thresholds and interactions
     that a straight line cannot capture.
  2. Train seven model families on the SAME data with fixed, untuned settings and measure the
     burst-time error (MAE) on held-out processes. Repeat on 5 different data seeds and average,
     so one lucky split cannot decide the ranking.
  3. Put each model inside the scheduler: it predicts every process's burst time and the scheduler
     runs the process with the smallest PREDICTED burst first (shortest-predicted-job-first). Measure
     the average waiting time over 100 random batches of 20 processes.
  4. Express the scheduler result as "share of the FCFS -> SJF gap closed":
         gap closed = (FCFS wait - model wait) / (FCFS wait - SJF wait) * 100
     FCFS is the naive baseline, SJF (told the true bursts) is the best case, so 100% would mean
     "as good as knowing every burst in advance".
  5. Save the table and draw the chart.

All data is synthetic, so the results show how the models compare on this made-up workload, not on
real process logs.
"""

from __future__ import annotations

import copy
import os
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

if "--show" not in sys.argv:
    matplotlib.use("Agg")                       # no window; just write the PNG
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generation import generate_dataset as _generate_dataset  # noqa: E402
from src.data_generation import generate_processes as _generate_processes  # noqa: E402
from src.feature_engineering import build_preprocessor, processes_to_frame  # noqa: E402
from src.process import FEATURE_COLUMNS  # noqa: E402
from src.schedulers import _run_non_preemptive, fcfs, sjf  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "results")
DATA_SEEDS = (42, 43, 44, 45, 46)     # step 2: five different datasets
N_BATCHES, BATCH_SIZE = 100, 20       # step 3: 100 random batches of 20 processes
RF = "Random Forest"

# --------------------------------------------------------------------------------------------
# Step 1: the non-linear, four-type dataset (now part of the main code: src/data_generation.py)
# --------------------------------------------------------------------------------------------
DATASET, NOISE = "nonlinear4", "low"      # "low" = no extra noise: the dataset exactly as designed


def generate_dataset(n_rows: int, seed: int) -> pd.DataFrame:
    return _generate_dataset(n_rows, seed=seed, noise=NOISE, dataset=DATASET)


def generate_processes(n: int, seed: int):
    return _generate_processes(n, seed=seed, noise=NOISE, dataset=DATASET)


# --------------------------------------------------------------------------------------------
# Step 2: the seven models (same fixed, untuned settings for everyone)
# --------------------------------------------------------------------------------------------
def model_zoo() -> dict:
    zoo = {
        "Linear Regression": lambda: LinearRegression(),
        "k-NN": lambda: KNeighborsRegressor(7),
        "Decision Tree": lambda: DecisionTreeRegressor(max_depth=8, random_state=0),
        RF: lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=0, n_jobs=-1),
        "SVM": lambda: SVR(C=30, epsilon=0.5),
        "Neural net": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=0),
    }
    try:
        from xgboost import XGBRegressor
        zoo["XGBoost"] = lambda: XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0, n_jobs=1)
    except ImportError:
        print("xgboost is not installed - the XGBoost row is skipped (pip install xgboost)")
    return zoo


def pipe(estimator) -> Pipeline:
    """Scale the numbers and one-hot encode the process type, then fit the model."""
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def measure_burst_error(zoo: dict) -> pd.DataFrame:
    rows = []
    for seed in DATA_SEEDS:
        df = generate_dataset(4000, seed)
        Xtr, Xte, ytr, yte = train_test_split(df[FEATURE_COLUMNS], df["actual_burst_time"], test_size=0.25, random_state=seed)
        for name, make in zoo.items():
            pred = pipe(make()).fit(Xtr, ytr).predict(Xte)
            rows.append({"model": name, "MAE": mean_absolute_error(yte, pred), "R2": r2_score(yte, pred)})
        print(f"  data seed {seed} done", flush=True)
    g = pd.DataFrame(rows).groupby("model")
    return g.agg(burst_MAE=("MAE", "mean"), burst_MAE_sd=("MAE", "std"), burst_R2=("R2", "mean"))


# --------------------------------------------------------------------------------------------
# Step 3 + 4: put each model inside the scheduler and compare with FCFS and SJF
# --------------------------------------------------------------------------------------------
def measure_scheduler(zoo: dict) -> pd.DataFrame:
    df = generate_dataset(4000, 42)
    Xtr, _, ytr, _ = train_test_split(df[FEATURE_COLUMNS], df["actual_burst_time"], test_size=0.25, random_state=42)
    fitted = {name: pipe(make()).fit(Xtr, ytr) for name, make in zoo.items()}
    waits = {name: [] for name in fitted}
    fcfs_w, sjf_w = [], []
    for b in range(N_BATCHES):
        procs = generate_processes(BATCH_SIZE, seed=1000 + b)
        frame = processes_to_frame(procs)
        fcfs_w.append(fcfs(copy.deepcopy(procs)).avg_waiting_time)
        sjf_w.append(sjf(copy.deepcopy(procs)).avg_waiting_time)          # the oracle: knows the true bursts
        for name, model in fitted.items():
            batch = copy.deepcopy(procs)
            for p, guess in zip(batch, model.predict(frame)):
                p.predicted_burst_time = float(guess)                      # the scheduler only sees this guess
            waits[name].append(_run_non_preemptive(batch, name, key=lambda p, t: p.predicted_burst_time).avg_waiting_time)
    f, s = np.mean(fcfs_w), np.mean(sjf_w)
    out = pd.DataFrame({"avg_waiting_ms": {n: np.mean(v) for n, v in waits.items()}})
    out["gap_closed_pct"] = (f - out["avg_waiting_ms"]) / (f - s) * 100
    print(f"  FCFS {f:.2f} ms, SJF (oracle) {s:.2f} ms")
    return out


# --------------------------------------------------------------------------------------------
# Step 5: the chart
# --------------------------------------------------------------------------------------------
def draw_chart(table: pd.DataFrame, path: str) -> None:
    by_err = table.sort_values("burst_MAE")
    by_gap = table.sort_values("gap_closed_pct", ascending=False)
    ordinal = lambda i: ["1st", "2nd", "3rd"][i] if i < 3 else f"{i + 1}th"        # noqa: E731

    def labels(order: pd.DataFrame) -> list[str]:
        out = []
        for i, name in enumerate(order.index):
            tag = f" ({ordinal(i)})" if name == RF else " (best)" if i == 0 else " (worst)" if i == len(order) - 1 else ""
            out.append(name + tag)
        return out

    colors = lambda order: ["#f2a900" if n == RF else "#8a9bc0" for n in order.index]  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    bars = ax.barh(labels(by_err), by_err["burst_MAE"], color=colors(by_err), edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("burst-time error, MAE (ms)  -  lower is better")
    ax.set_title("Burst-time error\n(mean of 5 data seeds)", fontsize=11, loc="left")
    for bar, v in zip(bars, by_err["burst_MAE"]):
        ax.text(v + 0.06, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, by_err["burst_MAE"].max() * 1.15)

    ax = axes[1]
    bars = ax.barh(labels(by_gap), by_gap["gap_closed_pct"], color=colors(by_gap), edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlim(90, 98)
    ax.set_xlabel("share of the FCFS -> SJF gap closed (%)  -  higher is better   [axis starts at 90%]")
    ax.set_title("In the scheduler\n(100 random batches of 20 processes)", fontsize=11, loc="left")
    for bar, v in zip(bars, by_gap["gap_closed_pct"]):
        ax.text(v + 0.06, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)

    for a in axes:
        a.grid(axis="x", alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Model comparison on the non-linear, four-process-type dataset  (Random Forest highlighted)",
                 fontsize=12.5, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Synthetic data, fixed untuned settings for every model. The scheduler results are nearly tied "
             "because the scheduler only needs a good ordering, not exact bursts.", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path, dpi=160)
    print("saved", path)
    if "--show" in sys.argv:
        plt.show()


def main() -> None:
    os.makedirs(RES, exist_ok=True)
    zoo = model_zoo()
    print("Step 2: burst-time error over 5 data seeds")
    errors = measure_burst_error(zoo)
    print("Step 3-4: each model inside the scheduler")
    sched = measure_scheduler(zoo)
    table = errors.join(sched).sort_values("burst_MAE")
    csv_path = os.path.join(RES, "e14_nonlinear_dataset_models.csv")
    table.reset_index().rename(columns={"index": "model"}).round(4).to_csv(csv_path, index=False)
    print("\n" + table.round(3).to_string())
    print("saved", csv_path)
    draw_chart(table, os.path.join(RES, "e14_nonlinear_dataset_models.png"))


if __name__ == "__main__":
    main()
