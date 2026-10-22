"""Run-to-run consistency of each model: how much does test error change when only the
random seed changes?  (Used by the scorecard on the dashboard's Graphs tab.)

Run from the project root:   python experiments/run_seed_context.py
Writes experiments/results/e9_seed_context.csv

Linear Regression, k-NN, Decision Tree, SVM and XGBoost (with the settings used here)
are deterministic for a fixed training set, so their seed spread is exactly 0.
Random Forest (deployed size) and the neural net genuinely depend on the seed.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generation import generate_dataset  # noqa: E402
from src.feature_engineering import build_preprocessor  # noqa: E402
from src.prediction import RF_TREES  # noqa: E402
from src.process import FEATURE_COLUMNS  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "e9_seed_context.csv")


def main() -> None:
    df = generate_dataset(4000, seed=42)
    Xtr, Xte, ytr, yte = train_test_split(df[FEATURE_COLUMNS], df["actual_burst_time"], test_size=0.25, random_state=42)
    prep = build_preprocessor().fit(Xtr)
    Ztr, Zte = prep.transform(Xtr), prep.transform(Xte)
    rf = [mean_absolute_error(yte, RandomForestRegressor(n_estimators=RF_TREES, min_samples_leaf=3, random_state=s, n_jobs=1)
                              .fit(Ztr, ytr).predict(Zte)) for s in range(15)]
    mlp = [mean_absolute_error(yte, MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=s)
                               .fit(Ztr, ytr).predict(Zte)) for s in range(8)]
    rows = [{"model": m, "mae_std_across_seeds": 0.0} for m in
            ("Linear Regression", "k-NN", "Decision Tree", "SVM", "XGBoost")]
    rows += [{"model": "Random Forest", "mae_std_across_seeds": float(np.std(rf))},
             {"model": "Neural net", "mae_std_across_seeds": float(np.std(mlp))}]
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
