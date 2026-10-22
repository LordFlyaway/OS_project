"""Scalability of each model family: error, training time and prediction time as the
training set grows from 500 to 16,000 rows.

Run from the project root:   python experiments/run_scalability.py
(needs `pip install xgboost` for the XGBoost rows)
Writes experiments/results/e10_scalability.csv
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generation import generate_dataset  # noqa: E402
from src.feature_engineering import build_preprocessor  # noqa: E402
from src.prediction import RF_TREES  # noqa: E402
from src.process import FEATURE_COLUMNS  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e10_scalability.csv")
SIZES = [500, 1000, 2000, 4000, 8000, 16000]
TEST_ROWS = 2000


def zoo():
    z = {
        "Linear Regression": lambda: LinearRegression(),
        "k-NN": lambda: KNeighborsRegressor(7),
        "Decision Tree": lambda: DecisionTreeRegressor(max_depth=8, random_state=0),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=RF_TREES, min_samples_leaf=3, random_state=0, n_jobs=1),
        "SVM": lambda: SVR(C=30, epsilon=0.5),
        "Neural net": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=0),
    }
    try:
        from xgboost import XGBRegressor
        z["XGBoost"] = lambda: XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0, n_jobs=1)
    except ImportError:
        print("xgboost not installed - XGBoost skipped")
    return z


def main() -> None:
    df = generate_dataset(SIZES[-1] + TEST_ROWS, seed=7)
    # generate_dataset returns rows sorted by arrival time; shuffle so train and test are drawn alike
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    train_all, test = df.iloc[:SIZES[-1]], df.iloc[SIZES[-1]:]
    Xte, yte = test[FEATURE_COLUMNS], test["actual_burst_time"]
    rows = []
    for n in SIZES:
        tr = train_all.iloc[:n]
        for name, mk in zoo().items():
            pipe = Pipeline([("prep", build_preprocessor()), ("model", mk())])
            t = time.perf_counter()
            pipe.fit(tr[FEATURE_COLUMNS], tr["actual_burst_time"])
            train_s = time.perf_counter() - t
            t = time.perf_counter()
            pred = pipe.predict(Xte)
            pred_ms_per_1000 = (time.perf_counter() - t) / TEST_ROWS * 1000 * 1000
            rows.append({"model": name, "train_rows": n, "MAE": round(mean_absolute_error(yte, pred), 3),
                         "train_s": round(train_s, 3), "predict_ms_per_1000": round(pred_ms_per_1000, 1)})
            print(f"{n:>6} {name:18s} MAE {rows[-1]['MAE']:.2f}  train {train_s:6.2f}s", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()
