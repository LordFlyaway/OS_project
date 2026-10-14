"""Module 2 - Feature Engineering.

Turns raw process attributes into a numeric matrix a scikit-learn estimator can
consume: one-hot encode the categorical `proc_type`, standard-scale the numeric
columns. The transformer is returned as an sklearn object so it can be dropped
straight into a Pipeline in Module 3 (fit on train data only -> no leakage).
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .process import FEATURE_COLUMNS, Process

NUMERIC_FEATURES = ["arrival_time", "memory_mb", "io_operations", "past_cpu_usage"]
CATEGORICAL_FEATURES = ["proc_type"]


def build_preprocessor() -> ColumnTransformer:
    """Fresh, unfitted preprocessing transformer."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def processes_to_frame(processes: list[Process]) -> pd.DataFrame:
    """Feature matrix (no labels) for a batch of live processes."""
    return pd.DataFrame([p.features() for p in processes], columns=FEATURE_COLUMNS)
