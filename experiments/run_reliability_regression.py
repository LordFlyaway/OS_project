"""Multi-variable regression: what makes a Random Forest's predictions unstable?

Run from the project root:   python experiments/run_reliability_regression.py
Writes experiments/results/e11_grid.csv and e11_regression.json

Design: a full grid of  trees x training rows x target noise.  For every cell the forest
is trained with 5 different random seeds, and we record
    seed disagreement  - how much predictions for the same process differ across seeds
    MAE                - mean test error
Then two ordinary-least-squares regressions on log-scaled predictors:
    log(seed disagreement) ~ log(trees) + log(rows) + noise
    MAE                    ~ log(trees) + log(rows) + noise
Predictors are standardised so the coefficients are comparable ("how many standard
deviations does the outcome move per standard deviation of the factor").
Confidence intervals come from a 300-sample bootstrap over the grid cells.
All data is synthetic.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generation import generate_dataset  # noqa: E402
from src.feature_engineering import build_preprocessor  # noqa: E402
from src.process import FEATURE_COLUMNS  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "results")
TREES = [5, 10, 20, 50, 100, 300]
ROWS = [500, 1000, 2000, 4000]
NOISE = [0.0, 4.0, 8.0]
SEEDS = list(range(5))
TEST_ROWS = 1000


def ols(X: np.ndarray, y: np.ndarray):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    ss_res, ss_tot = ((y - pred) ** 2).sum(), ((y - y.mean()) ** 2).sum()
    return beta, 1 - ss_res / ss_tot, pred


def standardise(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(axis=0)) / a.std(axis=0)


def regress(G: pd.DataFrame, target: str, rng: np.random.Generator) -> dict:
    Xraw = np.column_stack([np.log(G["trees"]), np.log(G["rows"]), G["noise"]])
    y = np.log(G[target].to_numpy()) if target == "seed_disagreement" else G[target].to_numpy()
    Xz, yz = standardise(Xraw), standardise(y.reshape(-1, 1)).ravel()
    beta, r2, pred = ols(Xz, yz)
    boots = []
    for _ in range(300):
        idx = rng.integers(0, len(G), len(G))
        boots.append(ols(Xz[idx], yz[idx])[0][1:])
    boots = np.array(boots)
    names = ["trees (log)", "training rows (log)", "target noise"]
    return {
        "target": target, "R2": float(r2), "n_cells": int(len(G)),
        "factors": [{"name": n, "coef": float(beta[i + 1]),
                     "ci_low": float(np.percentile(boots[:, i], 2.5)),
                     "ci_high": float(np.percentile(boots[:, i], 97.5))} for i, n in enumerate(names)],
        "actual": [float(v) for v in yz], "predicted": [float(v) for v in pred],
    }


def main() -> None:
    os.makedirs(RES, exist_ok=True)
    grid_path = os.path.join(RES, "e11_grid.csv")
    if "--reuse" in sys.argv and os.path.exists(grid_path):
        # skip the slow forest grid and only redo the regression on the saved results
        G = pd.read_csv(grid_path)
        rng = np.random.default_rng(0)
        out = {"consistency": regress(G, "seed_disagreement", rng), "accuracy": regress(G, "MAE", rng)}
        with open(os.path.join(RES, "e11_regression.json"), "w", encoding="utf-8") as f:
            json.dump(out, f)
        for k, v in out.items():
            print(k, "R2=%.3f" % v["R2"], [(f["name"], round(f["coef"], 2)) for f in v["factors"]])
        return
    df = generate_dataset(max(ROWS) + TEST_ROWS, seed=21)
    # rows come sorted by arrival time; shuffle so train and test are drawn alike
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    train_all, test = df.iloc[:max(ROWS)], df.iloc[max(ROWS):]
    rng_noise = np.random.default_rng(5)
    cells = []
    for noise in NOISE:
        # extra Gaussian noise on top of the generator's own noise, applied to train and test targets
        y_train = np.clip(train_all["actual_burst_time"].to_numpy() + rng_noise.normal(0, noise, len(train_all)), 1, 100)
        y_test = np.clip(test["actual_burst_time"].to_numpy() + rng_noise.normal(0, noise, len(test)), 1, 100)
        for rows in ROWS:
            prep = build_preprocessor().fit(train_all.iloc[:rows][FEATURE_COLUMNS])
            Ztr = prep.transform(train_all.iloc[:rows][FEATURE_COLUMNS])
            Zte = prep.transform(test[FEATURE_COLUMNS])
            for trees in TREES:
                preds = []
                for s in SEEDS:
                    rf = RandomForestRegressor(n_estimators=trees, min_samples_leaf=3, random_state=s, n_jobs=1)
                    preds.append(rf.fit(Ztr, y_train[:rows]).predict(Zte))
                P = np.vstack(preds)
                cells.append({"trees": trees, "rows": rows, "noise": noise,
                              "seed_disagreement": float(P.std(axis=0).mean()),
                              "MAE": float(np.mean([mean_absolute_error(y_test, p) for p in preds]))})
        print("noise", noise, "done", flush=True)
    G = pd.DataFrame(cells)
    G.to_csv(os.path.join(RES, "e11_grid.csv"), index=False)
    rng = np.random.default_rng(0)
    out = {"consistency": regress(G, "seed_disagreement", rng), "accuracy": regress(G, "MAE", rng)}
    with open(os.path.join(RES, "e11_regression.json"), "w", encoding="utf-8") as f:
        json.dump(out, f)
    for k, v in out.items():
        print(k, "R2=%.3f" % v["R2"], [(f["name"], round(f["coef"], 2)) for f in v["factors"]])


if __name__ == "__main__":
    main()
