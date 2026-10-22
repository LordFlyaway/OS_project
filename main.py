"""AI Task Scheduler - command-line entry point.

Usage
-----
  python main.py train                 # train + save the ML models, print MAE / accuracy
  python main.py run  [--n 12] [--seed 7] [--quantum 4] [--no-plots]
  python main.py compare               # LR / DT / RF / KNN comparison
  python main.py demo                  # fixed 8-process showcase with a printed Gantt
  python main.py dataset [--rows 20]   # peek at the synthesised training data
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from datetime import datetime

from src.data_generation import generate_dataset, generate_processes
from src.evaluation import OUTPUT_DIR, print_report, save_all
from src.prediction import Predictor, compare_models, train_models
from src.schedulers import ai_scheduler, run_all

LAST_RUN_PATH = os.path.join(OUTPUT_DIR, "last_run.json")


def _save_last_run(args: argparse.Namespace) -> None:
    """Remember which batch was just simulated so the dashboard can replay it."""
    aging = inspect.signature(ai_scheduler).parameters["aging_factor"].default
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LAST_RUN_PATH, "w", encoding="utf-8") as f:
        json.dump({"n": args.n, "seed": args.seed, "quantum": args.quantum, "aging": aging,
                   "source": args.command, "time": datetime.now().isoformat(timespec="seconds")}, f)


def _load_or_train_predictor() -> Predictor:
    try:
        return Predictor.load()
    except FileNotFoundError:
        print("[i] No saved models found - training a fresh pair...\n")
        reg, clf, report = train_models()
        print(report, "\n")
        return Predictor(reg, clf)


def cmd_train(args: argparse.Namespace) -> None:
    _, _, report = train_models(n_rows=args.rows, seed=args.seed, save=True)
    print(report)
    print("\nSaved to outputs/models/")


def cmd_compare(args):
    print(compare_models(seed=args.seed).fillna('').to_string(index=False))


def cmd_dataset(args: argparse.Namespace) -> None:
    df = generate_dataset(n_rows=args.rows, seed=args.seed)
    print(df.to_string(index=False))
    print(f"\n{len(df)} rows | priority distribution: "
          f"{df['actual_priority'].value_counts().sort_index().to_dict()}")


def cmd_run(args: argparse.Namespace) -> None:
    predictor = _load_or_train_predictor()
    processes = generate_processes(args.n, seed=args.seed)
    predictor.annotate(processes)

    print(f"\nSubmitted batch of {args.n} processes (seed={args.seed}):\n")
    print(f"{'PID':>4} {'arr':>4} {'burst':>6} {'prio':>5} {'type':>10} "
          f"{'pred_burst':>11} {'pred_prio':>10}")
    for p in sorted(processes, key=lambda x: x.pid):
        print(f"{p.pid:>4} {p.arrival_time:>4} {p.actual_burst_time:>6} "
              f"{p.actual_priority:>5} {p.proc_type:>10} "
              f"{p.predicted_burst_time:>11.1f} {p.predicted_priority:>10}")

    results = run_all(processes, quantum=args.quantum)
    print_report(results)
    _save_last_run(args)

    if not args.no_plots:
        paths = save_all(results)
        print("Figures written:")
        for pth in paths:
            print("  -", pth)


def cmd_demo(args: argparse.Namespace) -> None:
    args.n, args.seed, args.quantum, args.no_plots = 8, 7, 3, False
    cmd_run(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI Task Scheduler")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train and save the ML models")
    t.add_argument("--rows", type=int, default=4000)
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("compare", help="compare candidate ML models")
    c.add_argument("--seed", type=int, default=42)
    c.set_defaults(func=cmd_compare)

    d = sub.add_parser("dataset", help="print synthesised training data")
    d.add_argument("--rows", type=int, default=20)
    d.add_argument("--seed", type=int, default=42)
    d.set_defaults(func=cmd_dataset)

    r = sub.add_parser("run", help="simulate a batch under every algorithm")
    r.add_argument("--n", type=int, default=12)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--quantum", type=int, default=4)
    r.add_argument("--no-plots", action="store_true")
    r.set_defaults(func=cmd_run)

    dm = sub.add_parser("demo", help="fixed 8-process showcase")
    dm.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
