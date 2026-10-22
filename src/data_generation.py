"""Module 1 - Data Collection / Simulation.

A real deployment would log these numbers from the OS over time. For the project
we synthesise them with a fixed, documented relationship between the *known*
attributes and the *actual* burst time / priority, plus noise. The ML models in
Module 3 then have to recover that relationship from data alone - they never see
the formulas below.

TWO DATASETS (choose with `dataset=` or the environment variable SCHEDULER_DATASET)
  nonlinear4  (default)  four process types (CPU-bound, I/O-bound, memory-intensive,
                         interactive) and a burst formula with thresholds, feature
                         interactions and a smooth non-linear term, so a straight line
                         cannot represent it. Tree ensembles are expected to beat linear
                         models here.
  classic                two process types and a nearly linear burst formula (the
                         project's original dataset).

NOISE PRESETS (choose with `noise=` or SCHEDULER_NOISE: low, medium, high, extreme)
Real scheduling data is noisy, so each preset adds the same kinds of imperfection, in
increasing amounts, on top of the dataset's own noise:

  * run-to-run variation   the same program does not take the same CPU time twice
                           (multiplicative, log-normal, so it grows with the burst)
  * hidden load            an unobserved factor (cache, contention) scales the burst
  * outliers               a few processes run 2-5x longer than their attributes suggest
  * measurement error      the recorded `past_cpu_usage` is itself a noisy estimate
  * label noise            some priority classes are recorded wrongly

The default noise is "high". `noise="low"` adds nothing, so (dataset="nonlinear4",
noise="low") is exactly the non-linear dataset as originally designed, and
(dataset="classic", noise="low") is exactly the original project data. The noise
levels are assumptions, not measurements of any real system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .process import Process


# ---------------------------------------------------------------------------
# Noise presets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NoiseConfig:
    add_sigma: float = 3.0            # additive Gaussian noise on the burst (ms); datasets may add an offset
    run_sigma: float = 0.0            # log-normal run-to-run variation (relative)
    load_range: tuple = (1.0, 1.0)    # hidden multiplicative load factor
    outlier_rate: float = 0.0         # share of processes that run far longer
    outlier_scale: tuple = (2.0, 4.0)
    feature_sigma: float = 0.0        # relative measurement error on past_cpu_usage
    label_flip: float = 0.0           # chance a priority class is recorded wrongly
    burst_cap: int = 100              # longest burst allowed (ms)

    @property
    def is_clean(self) -> bool:
        return (self.run_sigma == 0 and self.load_range == (1.0, 1.0) and self.outlier_rate == 0
                and self.feature_sigma == 0 and self.label_flip == 0)


NOISE_PRESETS = {
    "low": NoiseConfig(),   # nothing extra: the dataset exactly as designed
    "medium": NoiseConfig(add_sigma=4.0, run_sigma=0.20, load_range=(0.90, 1.20), outlier_rate=0.02,
                          outlier_scale=(2.0, 3.0), feature_sigma=0.10, label_flip=0.05, burst_cap=300),
    "high": NoiseConfig(add_sigma=5.0, run_sigma=0.35, load_range=(0.85, 1.30), outlier_rate=0.04,
                        outlier_scale=(2.0, 4.0), feature_sigma=0.20, label_flip=0.10, burst_cap=300),
    "extreme": NoiseConfig(add_sigma=7.0, run_sigma=0.55, load_range=(0.75, 1.50), outlier_rate=0.08,
                           outlier_scale=(2.0, 5.0), feature_sigma=0.35, label_flip=0.20, burst_cap=300),
}
DEFAULT_NOISE = os.environ.get("SCHEDULER_NOISE", "high").lower()


def resolve_noise(noise) -> NoiseConfig:
    if noise is None:
        noise = DEFAULT_NOISE
    if isinstance(noise, NoiseConfig):
        return noise
    if noise not in NOISE_PRESETS:
        raise ValueError(f"unknown noise preset {noise!r}; choose from {sorted(NOISE_PRESETS)}")
    return NOISE_PRESETS[noise]


# ---------------------------------------------------------------------------
# The "hidden" ground-truth models. Kept in one place so the report can quote them.
# ---------------------------------------------------------------------------
def _signal_classic(proc_type: str, memory_mb: float, io_operations: int, past_cpu_usage: float) -> float:
    """Predictable part of the burst (ms): nearly linear in the attributes."""
    base = 0.55 * past_cpu_usage
    base += 16.0 if proc_type == "CPU-bound" else 3.0
    base += 0.015 * memory_mb          # bigger footprint -> a bit more compute
    base -= 0.9 * io_operations        # I/O-heavy work spends less time on CPU
    return base


def _priority_classic(rng, proc_type: str, burst: int, io_operations: int, memory_mb: float,
                      past_cpu_usage: float) -> int:
    """Priority class 1..3 (1 = most urgent): short interactive I/O work first, long batch work last."""
    if proc_type == "IO-bound" and burst <= 14 and io_operations >= 4:
        return 1
    if burst >= 42:
        return 3
    return 2


def _signal_nonlinear4(proc_type: str, memory_mb: float, io_operations: int, past_cpu_usage: float) -> float:
    """Predictable part of the burst (ms) with thresholds, interactions and a smooth non-linear term."""
    burst = 7.0 + 0.20 * past_cpu_usage + 0.006 * memory_mb
    burst += 0.35 * np.sqrt(io_operations + 1)
    burst += {"CPU-bound": 8.0, "IO-bound": -1.0, "Memory-intensive": 4.0, "Interactive": -3.0}[proc_type]
    if past_cpu_usage > 58:                                    # threshold effect
        burst += 10.0 + 0.08 * (past_cpu_usage - 58)
    if proc_type == "Memory-intensive" and memory_mb > 900:    # type x memory interaction
        burst += 13.0
    if proc_type == "IO-bound" and io_operations >= 8:
        burst -= 0.75 * (io_operations - 7)
    if past_cpu_usage > 45 and memory_mb > 700:                # two-feature interaction
        burst += 7.0
    if proc_type == "Interactive" and io_operations >= 5:
        burst -= 4.0
    if memory_mb > 1500 and past_cpu_usage > 65:
        burst += 8.0
    burst += 2.2 * np.sin(past_cpu_usage / 10.0)               # smooth non-linearity
    return float(burst)


def _priority_nonlinear4(rng, proc_type: str, burst: int, io_operations: int, memory_mb: float,
                         past_cpu_usage: float) -> int:
    """Priority class 1..3 (1 = most urgent) from workload behaviour."""
    score = 0.025 * past_cpu_usage + 0.004 * memory_mb - 0.20 * io_operations
    score += {"CPU-bound": 1.8, "IO-bound": -1.4, "Memory-intensive": 1.0, "Interactive": -2.0}[proc_type]
    if proc_type == "Interactive" and past_cpu_usage < 35:
        score -= 1.2
    if memory_mb > 1000 and past_cpu_usage > 60:
        score += 1.0
    if burst >= 55:                                            # long bursts are batch work
        score += 0.7
    score += rng.normal(0.0, 0.35)
    return 1 if score < 0.65 else 2 if score < 2.65 else 3


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    proc_types: tuple
    type_probs: tuple
    memory_range: tuple
    io_lambda: dict
    io_clip: int | None
    sigma_offset: float               # added to the noise preset's additive sigma
    signal: object = field(repr=False, default=None)
    priority: object = field(repr=False, default=None)


DATASETS = {
    "classic": DatasetSpec("classic", ("CPU-bound", "IO-bound"), (0.55, 0.45), (16, 1024),
                           {"CPU-bound": 2.0, "IO-bound": 6.0}, None, 0.0, _signal_classic, _priority_classic),
    "nonlinear4": DatasetSpec("nonlinear4", ("CPU-bound", "IO-bound", "Memory-intensive", "Interactive"),
                              (0.36, 0.30, 0.20, 0.14), (16, 2048),
                              {"CPU-bound": 2.0, "IO-bound": 9.0, "Memory-intensive": 4.0, "Interactive": 6.0},
                              25, 0.5, _signal_nonlinear4, _priority_nonlinear4),
}
DEFAULT_DATASET = os.environ.get("SCHEDULER_DATASET", "nonlinear4").lower()


def resolve_dataset(dataset) -> DatasetSpec:
    if dataset is None:
        dataset = DEFAULT_DATASET
    if isinstance(dataset, DatasetSpec):
        return dataset
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")
    return DATASETS[dataset]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_processes(n: int, seed: int | None = None, max_arrival: int | None = None,
                       noise=None, dataset=None) -> list[Process]:
    """Return a list of `n` fully-populated Process objects."""
    cfg, spec = resolve_noise(noise), resolve_dataset(dataset)
    rng = np.random.default_rng(seed)
    # Separate stream for the extra noise: the base attributes of a given seed are
    # identical whatever the noise level, so presets can be compared like for like.
    nrng = np.random.default_rng() if seed is None else np.random.default_rng([seed, 101])
    if max_arrival is None:
        max_arrival = max(1, n * 2)

    processes: list[Process] = []
    for pid in range(1, n + 1):
        proc_type = str(rng.choice(spec.proc_types, p=spec.type_probs))
        arrival = int(rng.integers(0, max_arrival + 1))
        memory_mb = float(round(rng.uniform(*spec.memory_range), 1))
        io_operations = int(rng.poisson(spec.io_lambda[proc_type]))
        if spec.io_clip is not None:
            io_operations = int(np.clip(io_operations, 0, spec.io_clip))
        past_cpu_usage = float(round(rng.uniform(2, 90), 1))

        burst_f = spec.signal(proc_type, memory_mb, io_operations, past_cpu_usage)
        if not cfg.is_clean:                      # always draw in the same order
            load = nrng.uniform(*cfg.load_range)
            run = float(np.exp(nrng.normal(0.0, cfg.run_sigma) - cfg.run_sigma ** 2 / 2))
            is_outlier = nrng.random() < cfg.outlier_rate
            out_scale = nrng.uniform(*cfg.outlier_scale)
            feat_err = float(np.exp(nrng.normal(0.0, cfg.feature_sigma)))
            flip = nrng.random() < cfg.label_flip
            flip_dir = 1 if nrng.random() < 0.5 else -1
            burst_f *= load * run * (out_scale if is_outlier else 1.0)
        burst_f += rng.normal(0.0, cfg.add_sigma + spec.sigma_offset)
        burst = int(np.clip(round(burst_f), 1, cfg.burst_cap))
        priority = spec.priority(rng, proc_type, burst, io_operations, memory_mb, past_cpu_usage)

        observed_past = past_cpu_usage
        if not cfg.is_clean:
            observed_past = float(round(np.clip(past_cpu_usage * feat_err, 1.0, 300.0), 1))
            if flip:
                priority = int(np.clip(priority + flip_dir, 1, 3))

        processes.append(Process(
            pid=pid,
            arrival_time=arrival,
            memory_mb=memory_mb,
            io_operations=io_operations,
            past_cpu_usage=observed_past,
            proc_type=proc_type,
            actual_burst_time=burst,
            actual_priority=priority,
        ))

    processes.sort(key=lambda p: (p.arrival_time, p.pid))
    return processes


def generate_dataset(n_rows: int = 4000, seed: int = 42, noise=None, dataset=None) -> pd.DataFrame:
    """A flat training table: one row per historical process."""
    procs = generate_processes(n_rows, seed=seed, max_arrival=n_rows, noise=noise, dataset=dataset)
    rows = []
    for p in procs:
        row = p.features()
        row["actual_burst_time"] = p.actual_burst_time
        row["actual_priority"] = p.actual_priority
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":  # quick sanity check
    for ds in DATASETS:
        for level in NOISE_PRESETS:
            df = generate_dataset(2000, seed=1, noise=level, dataset=ds)
            print(f"{ds:10s} {level:8s} burst mean {df.actual_burst_time.mean():5.1f}  sd {df.actual_burst_time.std():5.1f}  "
                  f"max {df.actual_burst_time.max():3d}  priority mix "
                  f"{df.actual_priority.value_counts(normalize=True).sort_index().round(2).to_dict()}")
