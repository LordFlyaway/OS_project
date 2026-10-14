"""Module 1 - Data Collection / Simulation.

A real deployment would log these numbers from the OS over time. For the project
we synthesise them with a fixed, documented relationship between the *known*
attributes and the *actual* burst time / priority, plus random noise. The ML
models in Module 3 then have to recover that relationship from data alone - they
never see the formula below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .process import PROC_TYPES, Process

# ---------------------------------------------------------------------------
# The "hidden" ground-truth model. Kept in one place so the report can quote it.
# ---------------------------------------------------------------------------


def _true_burst_time(rng: np.random.Generator, proc_type: str, memory_mb: float,
                     io_operations: int, past_cpu_usage: float) -> int:
    """How long the process WILL actually run on the CPU (ms)."""
    base = 0.55 * past_cpu_usage
    base += 16.0 if proc_type == "CPU-bound" else 3.0
    base += 0.015 * memory_mb          # bigger footprint -> a bit more compute
    base -= 0.9 * io_operations        # I/O-heavy work spends less time on CPU
    base += rng.normal(0.0, 3.0)       # irreducible noise
    return int(np.clip(round(base), 1, 100))


def _true_priority(proc_type: str, burst: int, io_operations: int) -> int:
    """Priority class 1..3 (1 = most urgent). Short, interactive, I/O-driven work
    is treated as high priority; long CPU-bound work as low priority."""
    if proc_type == "IO-bound" and burst <= 14 and io_operations >= 4:
        return 1
    if burst >= 42:
        return 3
    return 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_processes(n: int, seed: int | None = None,
                       max_arrival: int | None = None) -> list[Process]:
    """Return a list of `n` fully-populated Process objects."""
    rng = np.random.default_rng(seed)
    if max_arrival is None:
        max_arrival = max(1, n * 2)

    processes: list[Process] = []
    for pid in range(1, n + 1):
        proc_type = str(rng.choice(PROC_TYPES, p=[0.55, 0.45]))
        arrival = int(rng.integers(0, max_arrival + 1))
        memory_mb = float(round(rng.uniform(16, 1024), 1))
        # I/O-bound processes issue more I/O operations on average.
        io_lambda = 6 if proc_type == "IO-bound" else 2
        io_operations = int(rng.poisson(io_lambda))
        past_cpu_usage = float(round(rng.uniform(2, 90), 1))

        burst = _true_burst_time(rng, proc_type, memory_mb, io_operations,
                                 past_cpu_usage)
        priority = _true_priority(proc_type, burst, io_operations)

        processes.append(Process(
            pid=pid,
            arrival_time=arrival,
            memory_mb=memory_mb,
            io_operations=io_operations,
            past_cpu_usage=past_cpu_usage,
            proc_type=proc_type,
            actual_burst_time=burst,
            actual_priority=priority,
        ))

    processes.sort(key=lambda p: (p.arrival_time, p.pid))
    return processes


def generate_dataset(n_rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    """A flat training table: one row per historical process."""
    procs = generate_processes(n_rows, seed=seed, max_arrival=n_rows)
    rows = []
    for p in procs:
        row = p.features()
        row["actual_burst_time"] = p.actual_burst_time
        row["actual_priority"] = p.actual_priority
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":  # quick sanity check
    df = generate_dataset(10)
    print(df.to_string(index=False))
