"""Does a bigger forest give more *reliable* predictions?  (and does it give better ones?)

Run from the project root:   python experiments/run_tree_reliability.py
Writes experiments/results/tree_reliability.png and e9_*.csv.

"Reliable" is measured as consistency: train the same forest with different random
seeds (everything else identical) and see how much the result changes.
  seed disagreement   how much predictions for the same process differ across seeds
  MAE spread          standard deviation of test error across seeds
  scheduler spread    standard deviation of the average waiting time across seeds
Accuracy (mean MAE) is reported next to it, because reliability without accuracy
would be meaningless. Datasets: the original synthetic data, a non-linear variant,
a non-linear + extra-noisy variant, and a small training set (500 rows).
All data is synthetic.
"""

from __future__ import annotations

import copy
import os
import sys
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generation import generate_dataset, generate_processes  # noqa: E402
from src.feature_engineering import build_preprocessor, processes_to_frame  # noqa: E402
from src.process import FEATURE_COLUMNS  # noqa: E402
from src.schedulers import _run_non_preemptive  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "results")
TREES = [1, 3, 5, 10, 20, 50, 100, 300]
SEEDS = list(range(15))
AMBER, SLATE, INK, PANEL, TEXT, MUTED = "#ffb224", "#7f8fb5", "#0b0f1a", "#121a2b", "#e7ecf7", "#8793b0"
LINE_COLORS = {"original": AMBER, "non-linear": "#5fd6a0", "non-linear + extra noise": "#ff7a7a", "small data (500 rows)": "#8fa3ff"}


def nonlinear_target(df: pd.DataFrame, seed: int = 11, extra_noise: float = 0.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    cpu = (df["proc_type"] == "CPU-bound").to_numpy()
    y = (0.35 * df["past_cpu_usage"] + 18 * (df["past_cpu_usage"] > 45)
         + 24 * (cpu & (df["memory_mb"] > 500)) + 3.2 * np.sqrt(df["past_cpu_usage"]) * (~cpu)
         - 0.9 * df["io_operations"] + rng.normal(0, 3 + extra_noise, len(df)))
    return pd.Series(np.clip(np.round(y), 1, 100), index=df.index)


def main() -> None:
    os.makedirs(RES, exist_ok=True)
    df = generate_dataset(4000, seed=42)
    X = df[FEATURE_COLUMNS]
    targets = {
        "original": df["actual_burst_time"],
        "non-linear": nonlinear_target(df),
        "non-linear + extra noise": nonlinear_target(df, extra_noise=8.0),
        "small data (500 rows)": df["actual_burst_time"],
    }
    rows = []
    sched_rows = []
    for name, y in targets.items():
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
        if name.startswith("small"):
            Xtr, ytr = Xtr.iloc[:500], ytr.iloc[:500]
        prep = build_preprocessor().fit(Xtr)
        Ztr, Zte = prep.transform(Xtr), prep.transform(Xte)
        for k in TREES:
            preds, maes = [], []
            models = []
            for s in SEEDS:
                rf = RandomForestRegressor(n_estimators=k, min_samples_leaf=3, random_state=s, n_jobs=1).fit(Ztr, ytr)
                p = rf.predict(Zte)
                preds.append(p)
                maes.append(mean_absolute_error(yte, p))
                models.append(rf)
            preds = np.vstack(preds)
            rows.append({"dataset": name, "trees": k, "MAE_mean": np.mean(maes), "MAE_std_across_seeds": np.std(maes),
                         "seed_disagreement_ms": float(preds.std(axis=0).mean())})
            # scheduler-level spread, original data only
            if name == "original":
                waits = []
                for rf in models[:10]:
                    w = []
                    for b in range(50):
                        procs = generate_processes(20, seed=1000 + b)
                        est = rf.predict(prep.transform(processes_to_frame(procs)))
                        q = copy.deepcopy(procs)
                        for pr, v in zip(q, est):
                            pr.predicted_burst_time = float(v)
                        w.append(_run_non_preemptive(q, "x", key=lambda x, t: x.predicted_burst_time).avg_waiting_time)
                    waits.append(np.mean(w))
                sched_rows.append({"trees": k, "wait_mean": np.mean(waits), "wait_std_across_seeds": np.std(waits)})
        print(name, "done", flush=True)
    R = pd.DataFrame(rows)
    S = pd.DataFrame(sched_rows)
    R.to_csv(os.path.join(RES, "e9_tree_reliability.csv"), index=False)
    S.to_csv(os.path.join(RES, "e9_tree_scheduler.csv"), index=False)

    # cost of each size (original data)
    Xtr, Xte, ytr, yte = train_test_split(X, targets["original"], test_size=0.25, random_state=42)
    from sklearn.pipeline import Pipeline
    import io, joblib
    C = []
    for k in TREES:
        pipe = Pipeline([("prep", build_preprocessor()), ("model", RandomForestRegressor(n_estimators=k, min_samples_leaf=3, random_state=0, n_jobs=1))]).fit(Xtr, ytr)
        one = Xte.iloc[[0]]
        pipe.predict(one)
        t = time.perf_counter()
        for _ in range(40):
            pipe.predict(one)
        buf = io.BytesIO(); joblib.dump(pipe, buf)
        C.append({"trees": k, "latency_ms": (time.perf_counter() - t) / 40 * 1000, "size_kb": buf.getbuffer().nbytes / 1024})
    C = pd.DataFrame(C)
    C.to_csv(os.path.join(RES, "e9_tree_cost.csv"), index=False)

    # context: a neural net also depends on its random seed
    prep = build_preprocessor().fit(Xtr)
    Ztr, Zte = prep.transform(Xtr), prep.transform(Xte)
    mlp = []
    for s in range(8):
        mlp.append(mean_absolute_error(yte, MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=s).fit(Ztr, ytr).predict(Zte)))
    mlp_std, mlp_mean = float(np.std(mlp)), float(np.mean(mlp))
    print("MLP seed spread: mean %.2f std %.3f" % (mlp_mean, mlp_std))
    print(R[R.dataset == "original"].round(3).to_string(index=False))
    print(S.round(3).to_string(index=False))
    print(C.round(1).to_string(index=False))

    # ---------------- figure ----------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#2a3858"})
    fig, ax = plt.subplots(2, 3, figsize=(16, 9), facecolor=INK)
    for a in ax.flat:
        a.set_facecolor(PANEL); a.grid(alpha=.18)

    def lines(a, col, ylabel, title, ds=None, log=False):
        for d in (ds or targets):
            r = R[R.dataset == d]
            a.plot(r["trees"], r[col], "o-", color=LINE_COLORS[d], label=d, linewidth=2, markersize=5)
        a.set_xscale("log"); a.set_xlabel("number of trees")
        if log: a.set_yscale("log")
        a.set_ylabel(ylabel); a.set_title(title, color=TEXT, fontsize=11, loc="left")
        for x in (20, 100):
            a.axvline(x, color=MUTED, linestyle=":", linewidth=1)

    lines(ax[0, 0], "seed_disagreement_ms", "seed-to-seed disagreement (ms, log)",
          "1 · Same process, different random seed: predictions agree more", log=True)
    ax[0, 0].legend(fontsize=8, facecolor=PANEL, edgecolor="#2a3858", labelcolor=TEXT)
    lines(ax[0, 1], "MAE_std_across_seeds", "spread of test error across seeds (ms, log)",
          "2 · Test error wobbles less from run to run", log=True)
    a = ax[0, 2]
    a.errorbar(S["trees"], S["wait_mean"], yerr=S["wait_std_across_seeds"], fmt="o-", color=AMBER, capsize=3)
    a.set_xscale("log"); a.set_xlabel("number of trees"); a.set_ylabel("avg waiting time (ms), mean ± s.d. over seeds")
    a.set_title("3 · Scheduler outcome: spread across seeds", color=TEXT, fontsize=11, loc="left")
    for x in (20, 100):
        a.axvline(x, color=MUTED, linestyle=":", linewidth=1)
    lines(ax[1, 0], "MAE_mean", "mean test error (ms)", "4 · …but accuracy stops improving early")
    a = ax[1, 1]
    a.plot(C["trees"], C["latency_ms"], "o-", color="#5fd6a0", label="predict-one latency (ms)")
    a.set_xscale("log"); a.set_xlabel("number of trees"); a.set_ylabel("latency (ms)", color="#5fd6a0")
    t2 = a.twinx(); t2.plot(C["trees"], C["size_kb"] / 1024, "s--", color=SLATE, label="model size (MB)")
    t2.set_ylabel("model size (MB)", color=SLATE); t2.tick_params(colors=SLATE)
    a.set_title("5 · Cost keeps growing with the trees", color=TEXT, fontsize=11, loc="left")
    for x in (20, 100):
        a.axvline(x, color=MUTED, linestyle=":", linewidth=1)

    a = ax[1, 2]; a.axis("off")
    r20 = R[(R.dataset == "original") & (R.trees == 20)].iloc[0]
    r300 = R[(R.dataset == "original") & (R.trees == 300)].iloc[0]
    r1 = R[(R.dataset == "original") & (R.trees == 1)].iloc[0]
    txt = [
        ("Reading the charts", AMBER, 13),
        (f"More trees ⇒ more consistent. Seed disagreement\nfalls from {r1.seed_disagreement_ms:.2f} ms (1 tree) to {r20.seed_disagreement_ms:.2f} (20)\nto {r300.seed_disagreement_ms:.2f} ms (300).", TEXT, 10),
        ("Accuracy does NOT keep improving: it levels off\nby ~20–50 trees (panel 4).", TEXT, 10),
        ("So 'more reliable' is true; 'more accurate' is\nnot. Consistency has diminishing returns too.", TEXT, 10),
        (f"For comparison, a neural net's test error varies\nby ±{mlp_std:.3f} ms across seeds; a 20-tree forest by\n±{r20.MAE_std_across_seeds:.3f}, a 300-tree forest by ±{r300.MAE_std_across_seeds:.3f}.", TEXT, 10),
        ("Linear Regression and SVM have no seed at all\n(they are deterministic), so 'most reliable overall'\nis not a claim this test can support.", MUTED, 10),
    ]
    y0 = .97
    for t_, c_, sz in txt:
        a.text(0, y0, t_, color=c_, fontsize=sz, va="top", transform=a.transAxes, linespacing=1.35)
        y0 -= .062 * (t_.count("\n") + 1) + (.08 if sz > 11 else .035)
    fig.suptitle("Do more trees make a Random Forest more reliable?  (15 seeds per setting; dotted lines mark 20 and 100 trees)",
                 color=TEXT, fontsize=14, x=.02, ha="left", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(os.path.join(RES, "tree_reliability.png"), dpi=170, facecolor=INK)
    print("saved figure")


if __name__ == "__main__":
    main()
