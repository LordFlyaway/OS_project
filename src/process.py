"""The Process object.

This is the simulator's equivalent of an OS Process Control Block (PCB): it holds
the process identity, the attributes known to the OS at submission time, the
"ground truth" that the OS does NOT know in advance (actual burst time / priority),
the values predicted by the ML models, and the bookkeeping fields filled in while
the discrete-event simulation runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Attributes the OS genuinely knows when a process is submitted. These are the
# only fields the ML models are allowed to see as inputs.
FEATURE_COLUMNS = [
    "arrival_time",
    "memory_mb",
    "io_operations",
    "past_cpu_usage",
    "proc_type",
]

PROC_TYPES = ("CPU-bound", "IO-bound")

# Human-readable names for the three predicted priority classes (1 = most urgent).
PRIORITY_NAMES = {1: "High (interactive)", 2: "Medium", 3: "Low (batch)"}


@dataclass
class Process:
    # ---- identity -------------------------------------------------------------
    pid: int

    # ---- attributes known to the OS at submission (ML inputs) ----------------
    arrival_time: int = 0
    memory_mb: float = 0.0          # requested memory
    io_operations: int = 0          # number of I/O bursts requested
    past_cpu_usage: float = 0.0     # mean CPU time this program used historically
    proc_type: str = "CPU-bound"    # CPU-bound | IO-bound

    # ---- ground truth: hidden from the scheduler, used only for evaluation ---
    actual_burst_time: int = 1
    actual_priority: int = 2

    # ---- values produced by the Prediction Module ---------------------------
    predicted_burst_time: float | None = None
    predicted_priority: int | None = None

    # ---- runtime bookkeeping (reset before every simulation) ----------------
    remaining_time: int = field(default=0)
    start_time: int | None = None       # first instant this process holds the CPU
    completion_time: int | None = None

    def __post_init__(self) -> None:
        self.remaining_time = self.actual_burst_time

    # ------------------------------------------------------------------ helpers
    def reset_runtime(self) -> None:
        """Clear per-simulation state so the same process list can be reused by
        every scheduling algorithm on an equal footing."""
        self.remaining_time = self.actual_burst_time
        self.start_time = None
        self.completion_time = None

    # Derived OS scheduling metrics. Only valid once the process has completed.
    @property
    def turnaround_time(self) -> int:
        assert self.completion_time is not None
        return self.completion_time - self.arrival_time

    @property
    def waiting_time(self) -> int:
        return self.turnaround_time - self.actual_burst_time

    @property
    def response_time(self) -> int:
        assert self.start_time is not None
        return self.start_time - self.arrival_time

    def features(self) -> dict:
        """The row handed to the ML models."""
        return {c: getattr(self, c) for c in FEATURE_COLUMNS}

    def __repr__(self) -> str:
        return (
            f"P{self.pid}(arr={self.arrival_time}, burst={self.actual_burst_time}, "
            f"prio={self.actual_priority}, type={self.proc_type})"
        )
