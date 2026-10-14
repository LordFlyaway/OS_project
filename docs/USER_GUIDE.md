# AI Task Scheduler – User Guide

This guide covers (1) how to install and use the project, and (2) how the
project works and why it is built the way it is.

---

## Part 1 – How to use it

### 1.1 Requirements

- Python 3.10 or newer
- Packages in `requirements.txt` (NumPy, Pandas, scikit-learn, Matplotlib, joblib)

```bash
git clone https://github.com/LordFlyaway/OS_project.git
cd OS_project
python -m venv .venv            # optional but recommended
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

### 1.2 Commands

All commands are run from the repository root.

| Command | What it does |
|---|---|
| `python main.py train` | Generates a training set, trains the burst-time regressor and priority classifier, prints MAE / R² / accuracy / F1, saves the models to `outputs/models/`. Options: `--rows 4000`, `--seed 42`. |
| `python main.py compare` | Trains Linear Regression, Decision Tree, Random Forest (burst) and kNN, Decision Tree, Random Forest (priority) on the same split and prints a comparison table. |
| `python main.py demo` | Fixed 8-process example (seed 7, quantum 3). Prints the batch, the comparison table, and saves the charts. |
| `python main.py run` | Same as `demo` with your own settings: `--n 20` (number of processes), `--seed 5` (which random batch), `--quantum 4` (Round Robin time slice), `--no-plots` (skip chart files). |
| `python main.py dataset --rows 15` | Prints rows of the synthetic training data so you can see what the models learn from. |
| `python -m unittest discover -s tests` | Runs the invariant tests. |

If you run `demo` or `run` before `train`, the models are trained automatically.

### 1.3 Typical session

```bash
python main.py train
python main.py compare
python main.py run --n 20 --seed 5 --quantum 4
```

### 1.4 Reading the output

**Process table.** One row per process: arrival time, the *actual* burst and
priority (hidden from the AI scheduler), the type, and what the models
*predicted*. Comparing the actual and predicted columns shows how good the
predictions are.

**Comparison table.** One row per algorithm, lower is better except CPU
utilisation and throughput:

| Column | Meaning |
|---|---|
| `avg_waiting` | Average time spent in the ready queue (ms). |
| `avg_turnaround` | Average arrival-to-completion time (ms) = waiting + burst. |
| `avg_response` | Average arrival-to-first-CPU time (ms). |
| `cpu_util_%` | Share of the timeline the CPU was busy. Below 100% only if the CPU sat idle waiting for the first arrival. |
| `throughput` | Processes completed per ms. |
| `context_switches` | Number of times the CPU changed process. |

The line `AI vs SJF (oracle) gap` shows how far the AI scheduler is from SJF and
what percentage of the FCFS-to-SJF gap it closes.

**Charts** (written to `outputs/`):

- `metrics_comparison.png` – grouped bars of waiting, turnaround and response
  time for each algorithm.
- `gantt_charts.png` – one timeline per algorithm showing which process holds
  the CPU when. Each process has the same colour in every row.

### 1.5 Using your own processes

Build `Process` objects directly (see `src/process.py`), then annotate and run:

```python
from src.process import Process
from src.prediction import Predictor
from src.schedulers import run_all
from src.evaluation import print_report

procs = [
    Process(pid=1, arrival_time=0, memory_mb=300, io_operations=1,
            past_cpu_usage=60, proc_type="CPU-bound", actual_burst_time=45, actual_priority=3),
    Process(pid=2, arrival_time=2, memory_mb=40, io_operations=8,
            past_cpu_usage=5, proc_type="IO-bound", actual_burst_time=6, actual_priority=1),
]

predictor = Predictor.load()        # run `python main.py train` first
predictor.annotate(procs)           # fills predicted_burst_time / predicted_priority
print_report(run_all(procs, quantum=4))
```

---

## Part 2 – How the project works

### 2.1 The problem

Classical CPU scheduling algorithms each have a weakness:

- **FCFS** – short jobs get stuck behind long ones (convoy effect).
- **SJF** – gives the lowest average waiting time, but needs each process's
  burst time in advance, which a real OS does not know.
- **Round Robin** – fair, but results depend heavily on the chosen time quantum.
- **Priority** – low-priority processes can starve without aging.

This project **learns** the missing information (burst time and priority) from
past process data, then schedules with the predictions.

### 2.2 The five modules

```
 Module 1            Module 2              Module 3            Module 4          Module 5
 data_generation --> feature_engineering --> prediction  -->  schedulers  -->  evaluation
 (training data)     (encode + scale)       (train + predict)  (5 algorithms)   (metrics + charts)
```

1. **Data generation** (`src/data_generation.py`). Produces historical process
   records: arrival time, memory, number of I/O operations, past CPU usage, type
   (CPU-bound or I/O-bound), and the process's real burst time and priority class.
   The real values come from a hidden formula plus random noise (CPU-bound jobs
   run longer, more I/O means a shorter CPU burst, and so on). The models never
   see the formula; they must recover the pattern from examples.
2. **Feature engineering** (`src/feature_engineering.py`). A scikit-learn
   `ColumnTransformer`: standard-scales the numeric columns and one-hot encodes
   the process type. It is part of each model's `Pipeline`, so it is fitted on
   training data only.
3. **Prediction** (`src/prediction.py`). Two pipelines:
   - `RandomForestRegressor` predicts burst time (a continuous value).
   - `RandomForestClassifier` predicts priority class 1, 2 or 3 (1 = most urgent).

   They are scored on a held-out 25% of the data (MAE and R² for the regressor,
   accuracy and macro-F1 for the classifier). `Predictor.predict(process)` and
   `Predictor.annotate(processes)` are the only functions the scheduler needs,
   so the model can be swapped without touching scheduling code.
4. **Scheduler** (`src/schedulers.py`). FCFS, SJF, Round Robin, Priority and the
   AI scheduler, all on one discrete-event simulation. Each run produces a Gantt
   trace and per-process timings.
5. **Evaluation** (`src/evaluation.py`). Averages the metrics into a table and
   draws the bar chart and Gantt charts.

### 2.3 The AI scheduling rule

Whenever the CPU becomes free, every ready process gets a score and the
**smallest** score runs next (non-preemptive):

```
score = predicted_burst + 0.5 * predicted_priority - 0.35 * time_already_waited
```

- `predicted_burst` – behaves like SJF: shorter predicted jobs go first.
- `0.5 * predicted_priority` – a small nudge that pushes batch (low-priority)
  work back and acts as a tie-break.
- `- 0.35 * time_already_waited` – **aging**. The longer a process waits, the
  lower its score becomes, so a process that is always predicted "long" is
  eventually picked. Nothing starves.

The `0.35` is the `aging_factor` argument of `ai_scheduler()`.

### 2.4 Keeping the comparison fair

Every `Process` has two sets of numbers:

| Set | Used for |
|---|---|
| **Predicted** burst and priority | The AI scheduler's decisions only. |
| **Actual** burst and priority | Running the simulation and computing all timings. |

The AI scheduler never reads the actual values. If a prediction is wrong, the
schedule is worse and the metrics show it. The classical SJF and Priority
algorithms are given the actual values, as in a textbook problem, which makes
them "oracle" baselines: the best the AI scheduler could hope to match.

### 2.5 Interpreting results

- **AI close to SJF, far better than FCFS** – the predictions are good enough
  to recover most of SJF's benefit.
- **Round Robin has the worst waiting and turnaround but the best response
  time** – slicing every job delays completion but gives everyone the CPU
  quickly. It also has far more context switches (94 vs 7 in the demo).
- **Non-preemptive algorithms have response = waiting** – once a process
  starts, it runs to completion.
- A swap between AI and SJF ordering (for example P8 before P3 in the demo)
  is a misprediction: the model guessed the order wrong, at a small cost.

### 2.6 Testing

`tests/test_schedulers.py` checks that, for every algorithm:

- all processes complete,
- no waiting time is negative,
- total CPU busy time equals the sum of all bursts,

and that SJF is no worse than FCFS and the AI scheduler lands much closer to
SJF than to FCFS.

### 2.7 Limitations and future work

- Training data is **synthetic**; results demonstrate the approach, not
  real-world accuracy. Real process logs would be the next step.
- Single CPU core, no memory-aware scheduling.
- The AI scheduler is non-preemptive.
- Future work: reinforcement learning instead of predict-then-schedule,
  multi-core scheduling, a Linux `nice`/`cgroups` prototype, and online
  retraining from live execution logs.

### 2.8 Process model

The project follows the **Incremental model**: (1) classical simulator and
metrics as the test harness, (2) data and feature pipeline, (3) ML models,
(4) AI policy integration and charts.
