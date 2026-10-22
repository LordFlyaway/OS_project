"""Model-choice evidence: which burst-time model is the best *engineering* choice?

Run from the project root:   python experiments/run_model_charts.py
(needs `pip install xgboost` for the XGBoost entries; they are skipped otherwise)

Produces experiments/results/model_choice.png and e8_*.csv. Five questions:
  A  accuracy vs cost        - error against prediction latency and model size
  B  stability               - error spread over 10 different train/test splits
  C  non-linear stress test  - same models when the true burst is NOT linear
  D  how light can RF get    - error and latency against the number of trees
  E  inside the scheduler    - average waiting time when each model drives SJF

Everything uses synthetic data; C is an *additional* synthetic variant written to
stress the models, not evidence about real workloads.
"""

from __future__ import annotations

import copy
import io
import os
import sys
import time
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generation import generate_dataset, generate_processes  # noqa: E402
from src.prediction import RF_TREES  # noqa: E402
from src.feature_engineering import build_preprocessor, processes_to_frame  # noqa: E402
from src.process import FEATURE_COLUMNS  # noqa: E402
from src.schedulers import _run_non_preemptive, fcfs, sjf  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "results")
AMBER, SLATE, INK, PANEL, TEXT, MUTED = "#ffb224", "#7f8fb5", "#0b0f1a", "#121a2b", "#e7ecf7", "#8793b0"


def model_zoo(rf_jobs: int = -1):
    zoo = {
        "Linear Regression": lambda: LinearRegression(),
        "k-NN": lambda: KNeighborsRegressor(7),
        "Decision Tree": lambda: DecisionTreeRegressor(max_depth=8, random_state=0),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=RF_TREES, min_samples_leaf=3, random_state=0, n_jobs=1),
        "SVM": lambda: SVR(C=30, epsilon=0.5),
        "Neural net": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=0),
    }
    try:
        from xgboost import XGBRegressor
        zoo["XGBoost"] = lambda: XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0, n_jobs=1)
    except ImportError:
        print("xgboost not installed - XGBoost skipped")
    return zoo


def pipe_of(est):
    return Pipeline([("prep", build_preprocessor()), ("model", est)])


def latency_ms(pipe, row, reps=60):
    pipe.predict(row)
    t = time.perf_counter()
    for _ in range(reps):
        pipe.predict(row)
    return (time.perf_counter() - t) / reps * 1000


def size_kb(pipe):
    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    return buf.getbuffer().nbytes / 1024


def colors(names):
    return [AMBER if n == "Random Forest" else SLATE for n in names]


def nonlinear_target(df: pd.DataFrame, seed: int = 11) -> pd.Series:
    """A deliberately non-linear burst: thresholds and interactions, plus noise."""
    rng = np.random.default_rng(seed)
    cpu = (df["proc_type"] == "CPU-bound").to_numpy()
    y = (0.35 * df["past_cpu_usage"]
         + 18 * (df["past_cpu_usage"] > 45)
         + 24 * (cpu & (df["memory_mb"] > 500))
         + 3.2 * np.sqrt(df["past_cpu_usage"]) * (~cpu)
         - 0.9 * df["io_operations"]
         + rng.normal(0, 3, len(df)))
    return pd.Series(np.clip(np.round(y), 1, 100), index=df.index)


def main() -> None:
    os.makedirs(RES, exist_ok=True)
    df = generate_dataset(4000, seed=42)
    X, y = df[FEATURE_COLUMNS], df["actual_burst_time"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    zoo = model_zoo()
    names = list(zoo)

    # ---------------- A: accuracy vs cost ----------------
    A = []
    fitted = {}
    for n, mk in zoo.items():
        p = pipe_of(mk()).fit(Xtr, ytr)
        fitted[n] = p
        A.append({"model": n, "MAE": mean_absolute_error(yte, p.predict(Xte)),
                  "latency_ms": latency_ms(p, Xte.iloc[[0]]), "size_kb": size_kb(p)})
    A = pd.DataFrame(A)
    A.to_csv(os.path.join(RES, "e8_accuracy_vs_cost.csv"), index=False)
    print("A\n", A.round(2).to_string(index=False))

    # ---------------- B: stability over splits ----------------
    B = {n: [] for n in names}
    for s in range(10):
        a, b, c, d = train_test_split(X, y, test_size=0.25, random_state=100 + s)
        for n, mk in zoo.items():
            B[n].append(mean_absolute_error(d, pipe_of(mk()).fit(a, c).predict(b)))
    pd.DataFrame(B).to_csv(os.path.join(RES, "e8_stability.csv"), index=False)
    print("B mean±sd\n", {n: f"{np.mean(v):.2f}±{np.std(v):.2f}" for n, v in B.items()})

    # ---------------- C: non-linear stress test ----------------
    ynl = nonlinear_target(df)
    a, b, c, d = train_test_split(X, ynl, test_size=0.25, random_state=42)
    C = {n: mean_absolute_error(d, pipe_of(mk()).fit(a, c).predict(b)) for n, mk in zoo.items()}
    pd.Series(C, name="MAE").to_csv(os.path.join(RES, "e8_nonlinear.csv"))
    print("C", {k: round(v, 2) for k, v in C.items()})

    # ---------------- D: how light can a Random Forest get? ----------------
    D = []
    for k in (1, 2, 5, 10, 20, 30, 50, 100, 200):
        p = pipe_of(RandomForestRegressor(n_estimators=k, min_samples_leaf=3, random_state=0, n_jobs=1)).fit(Xtr, ytr)
        pn = pipe_of(RandomForestRegressor(n_estimators=k, min_samples_leaf=3, random_state=0, n_jobs=1)).fit(a, c)
        D.append({"trees": k, "MAE": mean_absolute_error(yte, p.predict(Xte)),
                  "MAE_nonlinear": mean_absolute_error(d, pn.predict(b)),
                  "latency_ms": latency_ms(p, Xte.iloc[[0]]), "size_kb": size_kb(p)})
    D = pd.DataFrame(D)
    D.to_csv(os.path.join(RES, "e8_rf_trees.csv"), index=False)
    print("D\n", D.round(2).to_string(index=False))

    # ---------------- E: inside the scheduler ----------------
    E = {n: [] for n in names}
    ref = {"FCFS": [], "SJF oracle": []}
    for k in range(100):
        procs = generate_processes(20, seed=1000 + k)
        frame = processes_to_frame(procs)
        ref["FCFS"].append(fcfs(copy.deepcopy(procs)).avg_waiting_time)
        ref["SJF oracle"].append(sjf(copy.deepcopy(procs)).avg_waiting_time)
        for n, p in fitted.items():
            preds = p.predict(frame)
            q = copy.deepcopy(procs)
            for pr, v in zip(q, preds):
                pr.predicted_burst_time = float(v)
            E[n].append(_run_non_preemptive(q, n, key=lambda x, t: x.predicted_burst_time).avg_waiting_time)
    Emean = {n: float(np.mean(v)) for n, v in E.items()}
    pd.Series({**ref_mean(ref), **Emean}, name="avg_waiting").to_csv(os.path.join(RES, "e8_scheduler_waiting.csv"))
    print("E", {k: round(v, 2) for k, v in Emean.items()}, {k: round(float(np.mean(v)), 2) for k, v in ref.items()})

    # ---------------- figure ----------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#2a3858"})
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.2), facecolor=INK)
    for a_ in ax.flat:
        a_.set_facecolor(PANEL)
        a_.grid(alpha=.18)

    # A
    a_ = ax[0, 0]
    for _, r in A.iterrows():
        rf = r["model"] == "Random Forest"
        a_.scatter(r["latency_ms"], r["MAE"], s=60 + r["size_kb"] / 6, c=AMBER if rf else SLATE,
                   edgecolor=INK, linewidth=1.5, zorder=3)
        a_.annotate(r["model"], (r["latency_ms"], r["MAE"]), textcoords="offset points",
                    xytext=(18, -4) if rf else (10, 8),
                    fontsize=9, color=AMBER if rf else TEXT)
    a_.set_xlim(0, A["latency_ms"].max() * 1.6)
    a_.set_ylim(A["MAE"].min() - 0.25 * (A["MAE"].max() - A["MAE"].min()), A["MAE"].max() + 0.25 * (A["MAE"].max() - A["MAE"].min()))
    a_.set_xlabel("time to predict one process (ms)  →  slower")
    a_.set_ylabel("error: MAE (ms)  →  worse")
    a_.set_title("A · Accuracy against cost (lower-left is best; bubble = file size)", color=TEXT, fontsize=11, loc="left")

    # B
    a_ = ax[0, 1]
    bp = a_.boxplot([B[n] for n in names], labels=[n.replace(" ", "\n") for n in names], patch_artist=True,
                    medianprops={"color": INK, "linewidth": 2})
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(AMBER if n == "Random Forest" else SLATE)
        patch.set_edgecolor(INK)
    a_.set_ylabel("MAE over 10 random splits (ms)")
    a_.set_title("B · Stability across train/test splits", color=TEXT, fontsize=11, loc="left")
    a_.tick_params(axis="x", labelsize=8)

    # C
    a_ = ax[0, 2]
    vals = [C[n] for n in names]
    a_.bar(range(len(names)), vals, color=colors(names), edgecolor=INK)
    a_.set_xticks(range(len(names)))
    a_.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    for i, v in enumerate(vals):
        a_.text(i, v + .1, f"{v:.1f}", ha="center", fontsize=9, color=TEXT)
    a_.set_ylabel("MAE (ms)")
    a_.set_title("C · Stress test: non-linear formula, low noise", color=TEXT, fontsize=11, loc="left")

    # D
    a_ = ax[1, 0]
    a_.plot(D["trees"], D["MAE"], "o-", color=AMBER, label="MAE, project data")
    a_.plot(D["trees"], D["MAE_nonlinear"], "s--", color=SLATE, label="MAE, non-linear data")
    a_.set_xscale("log")
    a_.set_xlabel("number of trees in the forest")
    a_.set_ylabel("MAE (ms)")
    t2 = a_.twinx()
    t2.plot(D["trees"], D["latency_ms"], "^:", color="#5fd6a0", label="predict-one latency")
    t2.set_ylabel("latency (ms)", color="#5fd6a0")
    t2.tick_params(colors="#5fd6a0")
    h1, l1 = a_.get_legend_handles_labels()
    h2, l2 = t2.get_legend_handles_labels()
    a_.legend(h1 + h2, l1 + l2, fontsize=8, facecolor=PANEL, edgecolor="#2a3858", labelcolor=TEXT)
    a_.set_title("D · A Random Forest can be made much lighter", color=TEXT, fontsize=11, loc="left")

    # E
    a_ = ax[1, 1]
    order = names + ["SJF oracle", "FCFS"]
    allv = {**Emean, "SJF oracle": float(np.mean(ref["SJF oracle"])), "FCFS": float(np.mean(ref["FCFS"]))}
    cols = [AMBER if n == "Random Forest" else ("#5fd6a0" if n == "SJF oracle" else ("#ff7a7a" if n == "FCFS" else SLATE)) for n in order]
    a_.barh(range(len(order)), [allv[n] for n in order], color=cols, edgecolor=INK)
    a_.set_yticks(range(len(order)))
    a_.set_yticklabels(order, fontsize=9)
    a_.invert_yaxis()
    lo = min(allv.values()) - 20
    a_.set_xlim(lo, max(allv.values()) + 15)
    for i, n in enumerate(order):
        a_.text(allv[n] + 1, i, f"{allv[n]:.1f}", va="center", fontsize=9, color=TEXT)
    a_.set_xlabel("average waiting time (ms), 100 batches (axis starts at %d)" % lo)
    a_.set_title("E · Inside the scheduler: who drives SJF best", color=TEXT, fontsize=11, loc="left")

    # takeaway panel - every sentence is computed from the results above
    a_ = ax[1, 2]
    a_.axis("off")
    rfA = A[A.model == "Random Forest"].iloc[0]
    rank_acc = int((A["MAE"] < rfA.MAE).sum()) + 1
    fastest = A.loc[A["latency_ms"].idxmin()]
    spreads = {n: float(np.std(v)) for n, v in B.items()}
    best_c = min(C, key=C.get)
    d20, dmax = D[D.trees == 20].iloc[0], D.iloc[-1]
    e_vals = list(Emean.values())
    lines = [
        ("What the charts show", AMBER, 13),
        (f"A  Random Forest error is {rfA.MAE:.1f} ms: rank {rank_acc} of {len(A)} on accuracy.\n"
         f"    It takes {rfA.latency_ms:.0f} ms to predict ({fastest.model}: {fastest.latency_ms:.0f} ms).", TEXT, 10),
        (f"B  Run-to-run spread over splits: Random Forest +/-{spreads['Random Forest']:.2f} ms,\n"
         f"    most stable +/-{min(spreads.values()):.2f} ({min(spreads, key=spreads.get)}).", TEXT, 10),
        (f"C  On a non-linear burst, Linear Regression gets {C['Linear Regression']:.1f} ms,\n"
         f"    Random Forest {C['Random Forest']:.1f}; best is {best_c} ({C[best_c]:.1f}).", TEXT, 10),
        (f"D  20 trees: {d20.MAE:.1f} ms error, {d20.latency_ms:.0f} ms; {int(dmax.trees)} trees:\n"
         f"    {dmax.MAE:.1f} ms, {dmax.latency_ms:.0f} ms.", TEXT, 10),
        (f"E  Driving the scheduler, the best and worst model differ by\n"
         f"    only {max(e_vals) - min(e_vals):.1f} ms of waiting time.", TEXT, 10),
    ]
    y0 = .96
    for txt, col, sz in lines:
        a_.text(0.0, y0, txt, color=col, fontsize=sz, va="top", transform=a_.transAxes, linespacing=1.35)
        y0 -= .06 * (txt.count("\n") + 1) + (.08 if sz > 11 else .03)

    fig.suptitle("Which model for burst-time prediction?  (Random Forest highlighted)", color=TEXT, fontsize=15, x=.02, ha="left", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .96))
    out = os.path.join(RES, "model_choice.png")
    fig.savefig(out, dpi=170, facecolor=INK)
    print("saved", out)


def ref_mean(ref):
    return {k: float(np.mean(v)) for k, v in ref.items()}


if __name__ == "__main__":
    main()
