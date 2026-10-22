# AI Task Scheduler

An intelligent, machine-learning-driven CPU process scheduling simulator.
*BCSE303L – Operating Systems project.*

A scikit-learn model predicts each process's **CPU burst time** and **priority
class** before it runs, and the scheduler orders the ready queue using those
predictions. This approximates Shortest Job First (SJF) without needing burst
times to be known in advance. Five algorithms (FCFS, SJF, Round Robin, Priority,
and the AI scheduler) are simulated side by side on the same process set.

| | |
|---|---|
| Language | Python 3.10+ |
| Libraries | NumPy, Pandas, scikit-learn, Matplotlib, joblib, Flask |
| Team | Asher Mathew (24BDE0128), Krishiv Rajkumar (24BDE0132), Snigdha Muralidharan (24BDE0133) |

## Quick start

```bash
pip install -r requirements.txt

python main.py train     # train + save the ML models
python main.py demo      # 8-process comparison, writes charts to outputs/output_N
python dashboard/app.py  # interactive dashboard at http://127.0.0.1:5050
```

On Windows, double-click `run_all.bat` (setup, train, compare, run, tests, experiments, then the
dashboard opens). Full usage is in **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)**.

## Synthetic data: two datasets, four noise levels

All data is synthetic (`src/data_generation.py`). You choose a **dataset** and a **noise level**:

| Setting | Options | Default |
|---|---|---|
| Dataset (`SCHEDULER_DATASET`) | `nonlinear4`: four process types (CPU-bound, I/O-bound, memory-intensive, interactive) and a burst formula with thresholds and interactions. `classic`: two types, nearly linear. | `nonlinear4` |
| Noise (`SCHEDULER_NOISE`) | `low` (nothing added), `medium`, `high`, `extreme` | `high` |

Each noise level adds run-to-run variation in the burst, a hidden load factor, outliers, measurement error
on `past_cpu_usage`, and wrongly recorded priority labels. `dataset=nonlinear4, noise=low` is the non-linear
dataset exactly as originally designed; `dataset=classic, noise=low` is the project's original data. For
example: `SCHEDULER_NOISE=medium python main.py train`. The noise levels are assumptions, not measurements of
any real system.

## Results (default: `nonlinear4`, `high` noise)

200 random batches of 20 processes, average waiting time (lower is better):

| Algorithm | Avg waiting (ms) |
|---|---|
| FCFS | 355.18 |
| SJF (oracle: told every true burst) | 225.80 |
| Round Robin (q = 4) | 442.74 |
| Priority (oracle: told the true priority) | 314.01 |
| **AI scheduler** | **274.91** |

- The AI scheduler waits about **23% less than FCFS** and closes about **60%** of the FCFS-to-SJF gap. It beats
  FCFS in 97% of batches, Round Robin in 100% and the oracle Priority scheduler in 90%. It never matches SJF.
- Model quality on held-out data is modest because the data is noisy: burst-time MAE about 17.2 ms (R² 0.28),
  priority accuracy about 87% (macro-F1 0.78).
- The benefit shrinks as noise grows (share of the FCFS-to-SJF gap closed): low 95%, medium 81%, high 62%,
  extreme 39% (`experiments/results/e13_noise_sweep.png`).
- The learned priority class and the aging term do not measurably help average waiting time
  (`experiments/run_extra_baselines.py`).
- On a single small batch the AI scheduler can lose: in the 8-process demo below it is slightly worse than FCFS.
- The project reproduces a well-known idea (predict burst time with machine learning, then run SJF); it is not a
  new scheduling method.

8-process demo (`python main.py demo`, seed 7):

| Algorithm | Avg waiting (ms) | Avg turnaround (ms) | Avg response (ms) | Context switches |
|---|---|---|---|---|
| FCFS | 70.38 | 93.25 | 70.38 | 7 |
| SJF (oracle) | 61.38 | 84.25 | 61.38 | 7 |
| Round Robin (q=3) | 103.38 | 126.25 | 8.88 | 60 |
| Priority | 65.12 | 88.00 | 65.12 | 7 |
| **AI Scheduler** | **71.12** | **94.00** | **71.12** | **7** |

![Metrics comparison](docs/images/metrics_comparison.png)
![Gantt charts](docs/images/gantt_charts.png)

### Which model? (`experiments/run_nonlinear_dataset_benchmark.py`)

On the non-linear dataset with no extra noise, seven model families compared (fixed, untuned settings, mean of
5 data seeds):

| Model | Burst error, MAE (ms) | Share of FCFS→SJF gap closed |
|---|---|---|
| XGBoost | **2.99** | 97.1% |
| **Random Forest** | **3.10** | **97.2%** |
| Neural net | 3.25 | 97.1% |
| Decision Tree | 3.42 | 96.2% |
| SVM | 3.56 | 96.1% |
| k-NN | 3.81 | 95.1% |
| Linear Regression | 5.69 | 92.8% |

Tree-based models beat a straight line on error, but the scheduler results are nearly tied. With the default
heavy noise the models are much closer: Linear Regression, SVM and the neural net match or beat the forest on
burst error (`experiments/results/e7_model_benchmark.csv`).

## Interactive dashboard

```bash
python dashboard/app.py        # then open http://127.0.0.1:5050
```

- **Simulator:** animated Gantt lanes for all five policies on the same batch (hover, zoom, pan, click a
  process to follow it across every policy), a decision inspector showing the AI scheduler's score for each
  waiting process, a scoreboard, and predicted-versus-true burst times.
- **Results:** the 200-batch experiments.
- **Graphs:** the model comparison above.

### Deploying it online

`render.yaml` is a [Render](https://render.com) blueprint. Create a free Render account, choose
**New > Blueprint**, and point it at this repository. The build trains the models; the start command
runs the Flask app with gunicorn. The free tier sleeps when idle, so the first load after a pause can
take about a minute.

## Experiments

All experiments are reproducible scripts in `experiments/`, with results in `experiments/results/`:
`run_experiments.py` (200 batches, aging, quantum, noise robustness), `run_noise_sweep.py` (data noise
levels), `run_extra_baselines.py` (ablations and naive baselines), `run_nonlinear_dataset_benchmark.py`
(the table above), `run_model_benchmark.py` and `run_model_charts.py` (seven model families),
`run_scalability.py`, `run_tree_reliability.py`, `run_reliability_regression.py`, and `run_wide_data_test.py`.
`pip install xgboost` is needed for the XGBoost rows.

## Repository layout

```
main.py                      command-line entry point
dashboard/                   Flask server + web page (animated Gantt, results, graphs)
experiments/                 reproducible experiments + result CSVs and figures
render.yaml                  one-click deploy blueprint (Render)
run_all.bat                  Windows: setup, run everything, open dashboard
src/
  process.py                 Process object (the simulator's PCB)
  data_generation.py         Module 1: synthetic data (two datasets, four noise levels)
  feature_engineering.py     Module 2: encoding + scaling
  prediction.py              Module 3: ML models + Predictor interface
  schedulers.py              Module 4: FCFS, SJF, RR, Priority, AI scheduler
  evaluation.py              Module 5: metrics table + charts
tests/test_schedulers.py     invariant tests
docs/USER_GUIDE.md           how to use it + how it works
```

## Tests

```bash
python -m unittest discover -s tests
```

## Note on the data

All data is synthetic, generated from documented formulas plus noise. The results show that the pipeline works
and how it degrades with noise; they are not evidence of performance on real process logs, which is listed as
future scope.
