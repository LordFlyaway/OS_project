"""Where does a Random Forest do best?  Four kinds of tabular data, same models, same settings.

Run from the project root:   python experiments/run_wide_data_test.py
(needs `pip install xgboost` for the XGBoost entries)
Writes experiments/results/e12_wide_data.csv and e12_wide_data.png

The regimes were fixed BEFORE running and nothing was tuned per regime:
  A  simple        5 numeric features, linear signal, light noise
  B  non-linear    12 informative features (8 numeric + 4 categorical), thresholds and
                   interactions, light noise
  C  wide + mixed  B plus 30 irrelevant numeric and 10 irrelevant categorical columns
                   (about 250 columns after one-hot encoding), light noise
  D  wide + messy  C with only 1,000 training rows, heavy-tailed noise and 3% outliers
Error is MAE against the NOISE-FREE true signal on 2,000 held-out rows (a clean measure
of predictive quality). Mean over 4 data seeds. All data is synthetic.
"""

from __future__ import annotations

import os
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

warnings.filterwarnings("ignore")
RES = os.path.join(os.path.dirname(__file__), "results")
SEEDS = [0, 1, 2, 3]
TEST_N = 2000


# ------------------------------------------------------------------ data
def make_data(regime: str, n_train: int, seed: int):
    rng = np.random.default_rng(seed)
    n = n_train + TEST_N

    if regime == "A":
        X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
        signal = 3 * X.x0 - 2 * X.x1 + 1.5 * X.x2 + 0.5 * X.x3
        noise = rng.normal(0, 1.0, n)
    else:
        # informative numeric features with different shapes
        num = {
            "n0": rng.uniform(-3, 3, n), "n1": rng.normal(0, 1, n), "n2": rng.normal(0, 2, n),
            "n3": rng.normal(0, 1, n), "n4": rng.uniform(-2, 2, n), "n5": rng.normal(0, 1, n),
            "n6": rng.normal(0, 1.5, n), "n7": rng.lognormal(0, 0.6, n),
        }
        # informative categorical features with different numbers of levels
        cats = {"c0": rng.integers(0, 4, n), "c1": rng.integers(0, 6, n),
                "c2": rng.integers(0, 8, n), "c3": rng.integers(0, 5, n)}
        e0, e1 = rng.normal(0, 2, 4), rng.normal(0, 2, 6)
        signal = (3 * np.sin(num["n0"]) + 2 * (num["n1"] > 0.5) * num["n2"]
                  + 4 * (num["n3"] ** 2 > 2) + 1.5 * num["n4"] * num["n5"]
                  + 2 * np.abs(num["n6"]) - 3 * (np.log(num["n7"]) < -0.5)
                  + e0[cats["c0"]] + e1[cats["c1"]]
                  + np.isin(cats["c2"], [1, 3, 5]) * num["n1"] * 2
                  + (cats["c3"] == 2) * (num["n0"] > 0) * 3)
        X = pd.DataFrame({**num, **{k: v.astype(str) for k, v in cats.items()}})
        if regime in ("C", "D"):
            for i in range(30):   # irrelevant numeric columns on mixed scales, some heavy-tailed
                kind = i % 3
                X[f"z{i}"] = (rng.normal(0, 1 + i % 5, n) if kind == 0
                              else rng.lognormal(0, 1, n) if kind == 1 else rng.standard_t(3, n))
            for i in range(10):   # irrelevant high-cardinality categoricals
                X[f"k{i}"] = rng.integers(0, 20, n).astype(str)
        noise = rng.normal(0, 1.0, n)
        if regime == "D":
            noise = rng.standard_t(2.5, n) * 1.5
            out = rng.random(n) < 0.03
            noise[out] += rng.choice([-1, 1], out.sum()) * rng.uniform(20, 40, out.sum())

    y_noisy = np.asarray(signal) + noise
    Xtr, Xte = X.iloc[:n_train], X.iloc[n_train:]
    return Xtr, y_noisy[:n_train], Xte, np.asarray(signal)[n_train:]


def preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = [c for c in X.columns if X[c].dtype != object]
    cat = [c for c in X.columns if X[c].dtype == object]
    parts = [("num", StandardScaler(), num)]
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return ColumnTransformer(parts)


def models():
    def scaled(est):
        return TransformedTargetRegressor(regressor=est, transformer=StandardScaler())
    z = {
        "Linear Regression": lambda: LinearRegression(),
        "k-NN": lambda: KNeighborsRegressor(10),
        "Decision Tree": lambda: DecisionTreeRegressor(min_samples_leaf=5, random_state=0),
        "Random Forest (100)": lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features=0.33, random_state=0, n_jobs=-1),
        "Random Forest (20)": lambda: RandomForestRegressor(n_estimators=20, min_samples_leaf=3, max_features=0.33, random_state=0, n_jobs=-1),
        "SVM": lambda: scaled(SVR(C=3)),
        "Neural net": lambda: scaled(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, early_stopping=True, random_state=0)),
    }
    try:
        from xgboost import XGBRegressor
        z["XGBoost"] = lambda: XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                                            colsample_bytree=0.8, random_state=0, n_jobs=1)
    except ImportError:
        print("xgboost not installed - XGBoost skipped")
    return z


REGIMES = {
    "A": ("A · Simple\n5 numeric, linear", 3000),
    "B": ("B · Non-linear\n12 mixed features", 3000),
    "C": ("C · Wide + mixed\n+40 irrelevant columns", 3000),
    "D": ("D · Wide + messy\nheavy noise, 1k rows", 1000),
}


def main() -> None:
    os.makedirs(RES, exist_ok=True)
    rows = []
    for key, (label, n_train) in REGIMES.items():
        for seed in SEEDS:
            Xtr, ytr, Xte, ytrue = make_data(key, n_train, seed)
            width = preprocessor(Xtr).fit(Xtr).transform(Xtr[:5]).shape[1]
            for name, mk in models().items():
                pipe = Pipeline([("prep", preprocessor(Xtr)), ("model", mk())]).fit(Xtr, ytr)
                rows.append({"regime": key, "label": label.replace("\n", " "), "seed": seed, "model": name,
                             "MAE_vs_truth": mean_absolute_error(ytrue, pipe.predict(Xte)), "columns_after_encoding": width})
                print(key, seed, name, round(rows[-1]["MAE_vs_truth"], 3), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "e12_wide_data.csv"), index=False)

    M = df.groupby(["regime", "model"])["MAE_vs_truth"].agg(["mean", "std"]).reset_index()
    pv = M.pivot(index="model", columns="regime", values="mean")
    print("\nMean MAE against the true signal (lower is better)\n", pv.round(2).to_string())
    print("\nRank of each model (1 = best)\n", pv.rank().astype(int).to_string())

    # ------------------------------------------------------------ figure
    INK, PANEL, TEXT, MUTED, AMBER = "#0b0f1a", "#121a2b", "#e7ecf7", "#8793b0", "#ffb224"
    colors = {"Random Forest (100)": AMBER, "Random Forest (20)": "#ffd27a", "Linear Regression": "#6ea8ff", "k-NN": "#a58bff",
              "Decision Tree": "#5fd6a0", "SVM": "#ff8ab3", "Neural net": "#7ee0e8", "XGBoost": "#c3c9d8"}
    names = list(pv.index)
    order = [n for n in colors if n in names]
    rel = pv.div(pv.min(axis=0), axis=1)          # 1.0 = best model in that regime
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#2a3858"})
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.4), facecolor=INK, gridspec_kw={"width_ratios": [1.5, 1]})
    a = ax[0]; a.set_facecolor(PANEL); a.grid(axis="y", alpha=.18)
    w = 0.8 / len(order)
    for i, m in enumerate(order):
        a.bar(np.arange(4) + (i - len(order) / 2 + .5) * w, [rel.loc[m, k] for k in REGIMES], w * .92,
              color=colors[m], edgecolor=INK, label=m)
    a.set_xticks(range(4)); a.set_xticklabels([REGIMES[k][0] for k in REGIMES], fontsize=9)
    a.set_ylabel("error relative to the best model in that data type  (1.0 = best, lower is better)")
    a.set_ylim(0.9, min(rel.max().max() * 1.05, 4)); a.axhline(1, color=MUTED, linewidth=.8)
    a.legend(fontsize=8, facecolor=PANEL, edgecolor="#2a3858", labelcolor=TEXT, ncol=2)
    a.set_title("Where does Random Forest do best?  (bars clipped at 4×)", color=TEXT, fontsize=12, loc="left")

    a = ax[1]; a.set_facecolor(PANEL); a.axis("off")
    ranks = pv.rank().astype(int)
    lines = [("Random Forest rank, by data type", AMBER, 12)]
    for k in REGIMES:
        rf100, rf20 = ranks.loc["Random Forest (100)", k], ranks.loc["Random Forest (20)", k]
        best = pv[k].idxmin()
        lines.append((f"{REGIMES[k][0].splitlines()[0]}:  RF(100) #{rf100}, RF(20) #{rf20}  of {len(names)}   (best: {best})", TEXT, 9.5))
    y0 = .93
    for t, c, sz in lines:
        a.text(0, y0, t, color=c, fontsize=sz, va="top", transform=a.transAxes, wrap=True)
        y0 -= .085 if sz > 11 else .1
    fig.suptitle("Random Forest across four kinds of tabular data (same settings everywhere, 4 seeds)", color=TEXT, fontsize=13, x=.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(os.path.join(RES, "e12_wide_data.png"), dpi=160, facecolor=INK)
    print("saved figure")


if __name__ == "__main__":
    main()
