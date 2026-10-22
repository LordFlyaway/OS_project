"""Local web dashboard for the AI Task Scheduler.

Run from the project root:   python dashboard/app.py
Then open http://127.0.0.1:5050  (a browser tab opens automatically).

The server only wraps the existing src/ package: it generates a batch of
processes, asks the trained Random Forests for predictions, runs all five
schedulers, and returns everything as JSON for the page to animate.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import threading
import time
import webbrowser

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data_generation import generate_processes  # noqa: E402
from src.prediction import BURST_MODEL_PATH, Predictor, train_models  # noqa: E402
from src.schedulers import (ai_scheduler, fcfs, priority_scheduling,  # noqa: E402
                            round_robin, sjf)

PORT = int(os.environ.get("SCHEDULER_PORT", "5050"))
RESULTS_DIR = os.path.join(ROOT, "experiments", "results")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        try:
            _predictor = Predictor.load()
        except FileNotFoundError:
            print("[dashboard] no saved models - training a fresh pair...")
            reg, clf, _ = train_models()
            _predictor = Predictor(reg, clf)
    return _predictor


def _model_meta() -> dict:
    """What the saved models were trained on (written by train_models)."""
    try:
        with open(os.path.join(ROOT, "outputs", "models", "meta.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _clamp(value, lo, hi, cast):
    try:
        return max(lo, min(hi, cast(value)))
    except (TypeError, ValueError):
        return lo


def _result_json(result) -> dict:
    return {
        "name": result.algorithm,
        "metrics": result.metrics(),
        "makespan": result.makespan,
        "gantt": [{"pid": s.pid, "start": s.start, "end": s.end} for s in result.gantt],
        "per_process": {
            str(p.pid): {
                "start": p.start_time,
                "completion": p.completion_time,
                "waiting": p.waiting_time,
                "turnaround": p.turnaround_time,
                "response": p.response_time,
            } for p in result.processes
        },
    }


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/healthz")
def healthz():
    return "ok"


@app.get("/api/simulate")
def simulate():
    n = _clamp(request.args.get("n", 8), 3, 30, int)
    seed = _clamp(request.args.get("seed", 7), 0, 999999, int)
    quantum = _clamp(request.args.get("quantum", 4), 1, 32, int)
    aging = _clamp(request.args.get("aging", 0.35), 0.0, 3.0, float)

    started = time.perf_counter()
    processes = generate_processes(n, seed=seed)
    get_predictor().annotate(processes)

    fresh = lambda: copy.deepcopy(processes)  # noqa: E731
    trace: list = []
    results = [
        fcfs(fresh()),
        sjf(fresh()),
        round_robin(fresh(), quantum=quantum),
        priority_scheduling(fresh()),
        ai_scheduler(fresh(), aging_factor=aging, trace=trace),
    ]
    keys = ["fcfs", "sjf", "rr", "priority", "ai"]
    algos = {}
    for k, r in zip(keys, results):
        algos[k] = _result_json(r)

    by_pid = {p.pid: p for p in processes}
    decisions = []
    for step in trace:
        t = step["time"]
        rows = []
        for q in step["queue"]:
            p = by_pid[q["pid"]]
            wait = t - p.arrival_time
            rows.append({
                "pid": p.pid,
                "score": round(q["score"], 3),
                "predicted_burst": round(p.predicted_burst_time, 2),
                "priority_term": round(0.5 * p.predicted_priority, 2),
                "aging_term": round(-aging * wait, 2),
                "waited": wait,
            })
        decisions.append({"time": t, "queue": rows})

    try:
        model_time = os.path.getmtime(BURST_MODEL_PATH)
        model_trained = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(model_time))
    except OSError:
        model_trained = None

    return jsonify({
        "server": {
            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "compute_ms": round((time.perf_counter() - started) * 1000, 1),
            "model_trained": model_trained,
            "model_noise": _model_meta().get("noise"),
            "model_dataset": _model_meta().get("dataset"),
        },
        "params": {"n": n, "seed": seed, "quantum": quantum, "aging": aging},
        "processes": [{
            "pid": p.pid,
            "arrival": p.arrival_time,
            "memory_mb": p.memory_mb,
            "io_operations": p.io_operations,
            "past_cpu_usage": p.past_cpu_usage,
            "type": p.proc_type,
            "burst": p.actual_burst_time,
            "priority": p.actual_priority,
            "predicted_burst": round(p.predicted_burst_time, 2),
            "predicted_priority": p.predicted_priority,
        } for p in sorted(processes, key=lambda x: x.pid)],
        "algorithms": algos,
        "decisions": decisions,
    })


def _records(name: str):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).astype(object)
    return df.where(df.notna(), None).to_dict(orient="records")


@app.get("/api/last-run")
def last_run():
    """Parameters of the batch most recently simulated by main.py (run/demo)."""
    defaults = {"n": 8, "seed": 7, "quantum": 3, "aging": 0.35, "source": "default", "time": None}
    path = os.path.join(ROOT, "outputs", "last_run.json")
    try:
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        return jsonify({**defaults, **saved})
    except (OSError, ValueError):
        return jsonify(defaults)


def _read_csv(name: str):
    path = os.path.join(RESULTS_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def _rows(df):
    df = df.astype(object)
    return df.where(df.notna(), None).to_dict(orient="records")


@app.get("/api/nonlinear-dataset")
def nonlinear_dataset():
    """Model comparison on the non-linear, four-process-type dataset (Graphs tab)."""
    df = _read_csv("e14_nonlinear_dataset_models.csv")
    if df is None:
        return jsonify({"available": False, "missing": ["e14_nonlinear_dataset_models.csv"]})
    return jsonify({"available": True, "rows": _rows(df)})


@app.get("/api/model-graphs")
def model_graphs():
    """Data behind the Graphs tab: model comparison, scalability, tree-count reliability, regression."""
    files = {
        "cost": "e8_accuracy_vs_cost.csv", "stability": "e8_stability.csv", "nonlinear": "e8_nonlinear.csv",
        "scheduler": "e8_scheduler_waiting.csv", "reliability": "e9_tree_reliability.csv",
        "tree_scheduler": "e9_tree_scheduler.csv", "tree_cost": "e9_tree_cost.csv",
        "seed_context": "e9_seed_context.csv", "scalability": "e10_scalability.csv",
    }
    frames = {k: _read_csv(v) for k, v in files.items()}
    reg_path = os.path.join(RESULTS_DIR, "e11_regression.json")
    missing = [files[k] for k, v in frames.items() if v is None]
    if not os.path.exists(reg_path):
        missing.append("e11_regression.json")
    if missing:
        return jsonify({"available": False, "missing": missing})

    nl = frames["nonlinear"].rename(columns={frames["nonlinear"].columns[0]: "model"})
    sch = frames["scheduler"].rename(columns={frames["scheduler"].columns[0]: "name"})
    out = {k: _rows(v) for k, v in frames.items() if k not in ("stability", "nonlinear", "scheduler")}
    out["nonlinear"] = _rows(nl)
    out["scheduler"] = _rows(sch)
    out["stability"] = {c: [float(x) for x in frames["stability"][c]] for c in frames["stability"].columns}
    with open(reg_path, encoding="utf-8") as f:
        out["regression"] = json.load(f)
    out["available"] = True
    return jsonify(out)


@app.get("/api/experiments")
def experiments():
    files = {
        "model_report": "e1_model_report.csv",
        "model_comparison": "e1_model_comparison.csv",
        "batches": "e2_summary.csv",
        "extra": "e2_extra.csv",
        "aging": "e3_aging_ablation.csv",
        "quantum": "e4_rr_quantum.csv",
        "noise": "e5_prediction_noise.csv",
    }
    out = {k: _records(v) for k, v in files.items()}
    out["available"] = all(v is not None for v in out.values())
    return jsonify(out)


def _port_in_use() -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def main() -> None:
    url = f"http://127.0.0.1:{PORT}"
    if _port_in_use():
        print(f"\n  The dashboard is already running at {url} - opening it.\n")
        if "--no-browser" not in sys.argv:
            webbrowser.open(url)
        return
    get_predictor()
    if "--no-browser" not in sys.argv:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"\n  AI Task Scheduler dashboard  ->  {url}\n  (press Ctrl+C to stop)\n")
    app.run(host="127.0.0.1", port=PORT, debug=False)


if __name__ == "__main__":
    main()
