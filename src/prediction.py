"""Module 3 - Prediction.

Two scikit-learn pipelines behind one small interface:

    * a regressor   -> predicted CPU burst time  (continuous)
    * a classifier  -> predicted priority class   (1 / 2 / 3)

The scheduler in Module 4 only ever touches `Predictor.predict(process)`, so the
models can be swapped or retrained without changing scheduling code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data_generation import DEFAULT_NOISE, generate_dataset
from .feature_engineering import build_preprocessor, processes_to_frame
from .process import FEATURE_COLUMNS, Process

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
BURST_MODEL_PATH = os.path.join(MODEL_DIR, "burst_regressor.joblib")
PRIORITY_MODEL_PATH = os.path.join(MODEL_DIR, "priority_classifier.joblib")

# Size of the deployed Random Forests. 20 trees keep ~99% of the 100-tree accuracy with a
# roughly 5x smaller model (see experiments/run_model_charts.py, panel D); n_jobs=1 avoids
# the thread start-up cost that dominates a single small prediction.
RF_TREES = 20


@dataclass
class TrainingReport:
    n_train: int
    n_test: int
    burst_mae: float
    burst_r2: float
    priority_accuracy: float
    priority_f1: float

    def __str__(self) -> str:
        return (
            "Model training report\n"
            "---------------------\n"
            f"  training rows          : {self.n_train}\n"
            f"  test rows              : {self.n_test}\n"
            f"  burst-time  MAE        : {self.burst_mae:.2f} ms\n"
            f"  burst-time  R^2        : {self.burst_r2:.3f}\n"
            f"  priority    accuracy   : {self.priority_accuracy:.3f}\n"
            f"  priority    macro-F1   : {self.priority_f1:.3f}"
        )


def _build_regressor() -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestRegressor(n_estimators=RF_TREES, min_samples_leaf=3, random_state=0, n_jobs=1)),
    ])


def _build_classifier() -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=RF_TREES, min_samples_leaf=3, random_state=0, n_jobs=1)),
    ])


MODEL_META_PATH = os.path.join(MODEL_DIR, "meta.json")


def train_models(n_rows: int = 4000, seed: int = 42, save: bool = True,
                 noise=None) -> tuple[Pipeline, Pipeline, TrainingReport]:
    """Generate data, fit both pipelines, evaluate on a held-out split."""
    df = generate_dataset(n_rows=n_rows, seed=seed, noise=noise)
    X = df[FEATURE_COLUMNS]
    y_burst = df["actual_burst_time"]
    y_prio = df["actual_priority"]

    X_tr, X_te, yb_tr, yb_te, yp_tr, yp_te = train_test_split(
        X, y_burst, y_prio, test_size=0.25, random_state=seed
    )

    regressor = _build_regressor().fit(X_tr, yb_tr)
    classifier = _build_classifier().fit(X_tr, yp_tr)

    yb_pred = regressor.predict(X_te)
    yp_pred = classifier.predict(X_te)

    report = TrainingReport(
        n_train=len(X_tr),
        n_test=len(X_te),
        burst_mae=float(mean_absolute_error(yb_te, yb_pred)),
        burst_r2=float(r2_score(yb_te, yb_pred)),
        priority_accuracy=float(accuracy_score(yp_te, yp_pred)),
        priority_f1=float(f1_score(yp_te, yp_pred, average="macro")),
    )

    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(regressor, BURST_MODEL_PATH)
        joblib.dump(classifier, PRIORITY_MODEL_PATH)
        with open(MODEL_META_PATH, "w", encoding="utf-8") as f:      # what these models were trained on
            json.dump({"noise": noise if isinstance(noise, str) else DEFAULT_NOISE, "rows": n_rows, "trees": RF_TREES}, f)

    return regressor, classifier, report


class Predictor:
    """Loads the trained pipelines and annotates Process objects in place."""

    def __init__(self, regressor: Pipeline | None = None,
                 classifier: Pipeline | None = None):
        self.regressor = regressor
        self.classifier = classifier

    # ---- construction -------------------------------------------------------
    @classmethod
    def load(cls) -> "Predictor":
        if not (os.path.exists(BURST_MODEL_PATH) and os.path.exists(PRIORITY_MODEL_PATH)):
            raise FileNotFoundError(
                "Models not found. Run `python main.py train` first."
            )
        return cls(joblib.load(BURST_MODEL_PATH), joblib.load(PRIORITY_MODEL_PATH))

    @classmethod
    def train_and_get(cls, **kwargs) -> "Predictor":
        reg, clf, _ = train_models(**kwargs)
        return cls(reg, clf)

    # ---- inference --------------------------------------------------------
    def predict(self, process: Process) -> tuple[float, int]:
        """Return (predicted_burst, predicted_priority) for a single process."""
        row = pd.DataFrame([process.features()], columns=FEATURE_COLUMNS)
        burst = float(np.clip(self.regressor.predict(row)[0], 1.0, None))
        priority = int(self.classifier.predict(row)[0])
        return burst, priority

    def annotate(self, processes: list[Process]) -> list[Process]:
        """Batch version - fills predicted_* on every process and returns them."""
        frame = processes_to_frame(processes)
        bursts = np.clip(self.regressor.predict(frame), 1.0, None)
        priorities = self.classifier.predict(frame)
        for p, b, pr in zip(processes, bursts, priorities):
            p.predicted_burst_time = float(b)
            p.predicted_priority = int(pr)
        return processes


def compare_models(n_rows: int = 4000, seed: int = 42, noise=None) -> pd.DataFrame:
    """Report objective 2: evaluate the candidate scikit-learn models."""
    df = generate_dataset(n_rows=n_rows, seed=seed, noise=noise)
    X, yb, yp = df[FEATURE_COLUMNS], df["actual_burst_time"], df["actual_priority"]
    X_tr, X_te, yb_tr, yb_te, yp_tr, yp_te = train_test_split(
        X, yb, yp, test_size=0.25, random_state=seed)

    rows = []
    regs = {"Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=0),
            "Random Forest": RandomForestRegressor(n_estimators=RF_TREES, min_samples_leaf=3,
                                                   random_state=0, n_jobs=1)}
    for name, m in regs.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", m)]).fit(X_tr, yb_tr)
        pred = pipe.predict(X_te)
        rows.append({"task": "burst (regression)", "model": name,
                     "MAE": round(mean_absolute_error(yb_te, pred), 3),
                     "R2": round(r2_score(yb_te, pred), 3)})
    clfs = {"k-Nearest Neighbours": KNeighborsClassifier(n_neighbors=7),
            "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=0),
            "Random Forest": RandomForestClassifier(n_estimators=RF_TREES, min_samples_leaf=3,
                                                    random_state=0, n_jobs=1)}
    for name, m in clfs.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", m)]).fit(X_tr, yp_tr)
        pred = pipe.predict(X_te)
        rows.append({"task": "priority (classification)", "model": name,
                     "accuracy": round(accuracy_score(yp_te, pred), 3),
                     "macro_F1": round(f1_score(yp_te, pred, average="macro"), 3)})
    return pd.DataFrame(rows)
