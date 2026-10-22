"""Benchmark of seven model families for burst time and priority class.

Run from the project root:   python experiments/run_model_benchmark.py
Needs one extra package for the XGBoost rows:   pip install xgboost
Writes experiments/results/e7_model_benchmark.csv

Same data and split as `python main.py compare` (4000 rows, seed 42, 75/25 split).
Every model uses fixed, reasonable settings; none was tuned, and there is a single
split (no cross-validation), so differences of a few hundredths are within noise.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generation import generate_dataset  # noqa: E402
from src.prediction import RF_TREES  # noqa: E402
from src.feature_engineering import build_preprocessor  # noqa: E402
from src.process import FEATURE_COLUMNS  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e7_model_benchmark.csv")


def main() -> None:
    try:
        from xgboost import XGBClassifier, XGBRegressor
        have_xgb = True
    except ImportError:
        have_xgb = False
        print("xgboost not installed - skipping the XGBoost rows (pip install xgboost)")

    df = generate_dataset(n_rows=4000, seed=42)
    X, yb, yp = df[FEATURE_COLUMNS], df["actual_burst_time"], df["actual_priority"]
    Xtr, Xte, ybtr, ybte, yptr, ypte = train_test_split(X, yb, yp, test_size=0.25, random_state=42)

    regs = [
        ("Linear Regression", LinearRegression()),
        ("k-NN (k=7)", KNeighborsRegressor(7)),
        ("Decision Tree (depth 8)", DecisionTreeRegressor(max_depth=8, random_state=0)),
        ("Random Forest (deployed)", RandomForestRegressor(n_estimators=RF_TREES, min_samples_leaf=3, random_state=0, n_jobs=1)),
        ("SVM / SVR (RBF)", SVR(C=30, epsilon=0.5)),
        ("Neural net (64,32)", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=0)),
    ]
    clfs = [
        ("k-NN (k=7)", KNeighborsClassifier(7), 0),
        ("Decision Tree (depth 8)", DecisionTreeClassifier(max_depth=8, random_state=0), 0),
        ("Random Forest (deployed)", RandomForestClassifier(n_estimators=RF_TREES, min_samples_leaf=3, random_state=0, n_jobs=1), 0),
        ("SVM / SVC (RBF)", SVC(C=10), 0),
        ("Neural net (64,32)", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=0), 0),
    ]
    if have_xgb:
        regs.append(("XGBoost", XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0, n_jobs=1)))
        clfs.append(("XGBoost", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0, n_jobs=1), 1))

    def fit_time(est, ytr):
        pipe = Pipeline([("prep", build_preprocessor()), ("model", est)])
        t = time.perf_counter()
        pipe.fit(Xtr, ytr)
        train_s = time.perf_counter() - t
        one = Xte.iloc[[0]]
        pipe.predict(one)
        t = time.perf_counter()
        for _ in range(100):
            pipe.predict(one)
        return pipe, train_s, (time.perf_counter() - t) / 100 * 1000

    rows = []
    for name, est in regs:
        pipe, tr, inf = fit_time(est, ybtr)
        pr = pipe.predict(Xte)
        rows.append({"task": "burst", "model": name, "MAE_ms": round(mean_absolute_error(ybte, pr), 2),
                     "R2": round(r2_score(ybte, pr), 3), "train_s": round(tr, 2), "predict_one_ms": round(inf, 1)})
    for name, est, zero_based in clfs:
        shift = 1 if zero_based else 0          # XGBoost needs labels starting at 0
        pipe, tr, inf = fit_time(est, yptr - shift)
        pr = pipe.predict(Xte) + shift
        rows.append({"task": "priority", "model": name, "accuracy": round(accuracy_score(ypte, pr), 3),
                     "macro_F1": round(f1_score(ypte, pr, average="macro"), 3),
                     "train_s": round(tr, 2), "predict_one_ms": round(inf, 1)})

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.fillna("").to_string(index=False))


if __name__ == "__main__":
    main()
