"""Module 4 - Scheduler.

Every algorithm consumes the SAME list of Process objects and produces a
ScheduleResult (a Gantt trace + per-process timing). Classical algorithms are
handed the ground-truth burst / priority, exactly as a textbook simulation would
give them; the AI policy is only allowed to look at `predicted_*`.

All algorithms here are non-preemptive except Round Robin, which is the classic
preemptive time-sliced version.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .process import Process


@dataclass
class GanttSegment:
    pid: int
    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass
class ScheduleResult:
    algorithm: str
    gantt: list[GanttSegment] = field(default_factory=list)
    processes: list[Process] = field(default_factory=list)
    context_switches: int = 0

    # ---- aggregate OS metrics -------------------------------------------
    @property
    def makespan(self) -> int:
        return max((s.end for s in self.gantt), default=0)

    @property
    def busy_time(self) -> int:
        return sum(s.duration for s in self.gantt)

    @property
    def avg_waiting_time(self) -> float:
        return _mean(p.waiting_time for p in self.processes)

    @property
    def avg_turnaround_time(self) -> float:
        return _mean(p.turnaround_time for p in self.processes)

    @property
    def avg_response_time(self) -> float:
        return _mean(p.response_time for p in self.processes)

    @property
    def cpu_utilisation(self) -> float:
        return 100.0 * self.busy_time / self.makespan if self.makespan else 0.0

    @property
    def throughput(self) -> float:
        return len(self.processes) / self.makespan if self.makespan else 0.0

    def metrics(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "avg_waiting": round(self.avg_waiting_time, 2),
            "avg_turnaround": round(self.avg_turnaround_time, 2),
            "avg_response": round(self.avg_response_time, 2),
            "cpu_util_%": round(self.cpu_utilisation, 2),
            "throughput": round(self.throughput, 4),
            "context_switches": self.context_switches,
        }


def _mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _prepare(processes: list[Process]) -> list[Process]:
    for p in processes:
        p.reset_runtime()
    return sorted(processes, key=lambda p: (p.arrival_time, p.pid))


# ---------------------------------------------------------------------------
# Generic non-preemptive engine
# ---------------------------------------------------------------------------


def _run_non_preemptive(processes: list[Process], algorithm: str, key) -> ScheduleResult:
    """`key(process, now)` -> sort key; the ready process with the smallest key
    runs next and runs to completion."""
    procs = _prepare(processes)
    n = len(procs)
    result = ScheduleResult(algorithm=algorithm, processes=procs)

    time = 0
    done = 0
    ready: list[Process] = []
    i = 0  # index into arrival-sorted procs

    while done < n:
        while i < n and procs[i].arrival_time <= time:
            ready.append(procs[i])
            i += 1

        if not ready:
            time = procs[i].arrival_time
            continue

        ready.sort(key=lambda p: (key(p, time), p.arrival_time, p.pid))
        current = ready.pop(0)

        start = time
        if current.start_time is None:
            current.start_time = start
        if result.gantt:
            result.context_switches += 1

        time += current.actual_burst_time
        current.completion_time = time
        done += 1
        result.gantt.append(GanttSegment(current.pid, start, time))

    return result


# ---------------------------------------------------------------------------
# Classical algorithms
# ---------------------------------------------------------------------------


def fcfs(processes: list[Process]) -> ScheduleResult:
    return _run_non_preemptive(processes, "FCFS", key=lambda p, now: p.arrival_time)


def sjf(processes: list[Process]) -> ScheduleResult:
    """Non-preemptive Shortest Job First using the true burst time (oracle)."""
    return _run_non_preemptive(processes, "SJF", key=lambda p, now: p.actual_burst_time)


def priority_scheduling(processes: list[Process]) -> ScheduleResult:
    """Lower priority number = more urgent. Uses the true priority class."""
    return _run_non_preemptive(processes, "Priority",
                               key=lambda p, now: p.actual_priority)


def round_robin(processes: list[Process], quantum: int = 4) -> ScheduleResult:
    """Classic preemptive Round Robin with a fixed time quantum."""
    procs = _prepare(processes)
    n = len(procs)
    result = ScheduleResult(algorithm=f"Round Robin (q={quantum})", processes=procs)

    time = 0
    i = 0
    queue: list[Process] = []
    done = 0

    while i < n and procs[i].arrival_time <= time:
        queue.append(procs[i]); i += 1

    while done < n:
        if not queue:
            time = procs[i].arrival_time
            while i < n and procs[i].arrival_time <= time:
                queue.append(procs[i]); i += 1
            continue

        current = queue.pop(0)
        if current.start_time is None:
            current.start_time = time
        if result.gantt:
            result.context_switches += 1

        slice_len = min(quantum, current.remaining_time)
        start = time
        time += slice_len
        current.remaining_time -= slice_len
        result.gantt.append(GanttSegment(current.pid, start, time))

        # Admit everything that arrived during this slice BEFORE requeueing
        # the current process (standard RR convention).
        while i < n and procs[i].arrival_time <= time:
            queue.append(procs[i]); i += 1

        if current.remaining_time > 0:
            queue.append(current)
        else:
            current.completion_time = time
            done += 1

    return result


# ---------------------------------------------------------------------------
# The AI-assisted policy
# ---------------------------------------------------------------------------


def ai_scheduler(processes: list[Process], aging_factor: float = 0.35) -> ScheduleResult:
    """Predicted-SJF with a predicted-priority tie-break and aging.

    Selection key for a ready process at time `now`:

        key = predicted_burst
              + 0.5 * predicted_priority      # nudge low-priority work back
              - aging_factor * (now - arrival) # aging: the longer it waits,
                                               #        the smaller the key
    Requires `predicted_burst_time` / `predicted_priority` to be populated by the
    Prediction Module.
    """
    for p in processes:
        if p.predicted_burst_time is None or p.predicted_priority is None:
            raise ValueError(
                f"P{p.pid} has no predictions - call Predictor.annotate() first."
            )

    def key(p: Process, now: int) -> float:
        wait = now - p.arrival_time
        return (p.predicted_burst_time
                + 0.5 * p.predicted_priority
                - aging_factor * wait)

    return _run_non_preemptive(processes, "AI Scheduler", key=key)


# ---------------------------------------------------------------------------
# Convenience: run the whole comparison
# ---------------------------------------------------------------------------


def run_all(processes: list[Process], quantum: int = 4) -> list[ScheduleResult]:
    """Runs every algorithm on independent copies of the process list."""
    import copy

    def fresh():
        return copy.deepcopy(processes)

    return [
        fcfs(fresh()),
        sjf(fresh()),
        round_robin(fresh(), quantum=quantum),
        priority_scheduling(fresh()),
        ai_scheduler(fresh()),
    ]
