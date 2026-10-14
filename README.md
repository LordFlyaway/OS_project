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
| Libraries | NumPy, Pandas, scikit-learn, Matplotlib, joblib |
| Team | Asher Mathew (24BDE0128), Krishiv Rajkumar (24BDE0132), Snigdha Muralidharan (24BDE0133) |

## Quick start

```bash
git clone https://github.com/LordFlyaway/OS_project.git
cd OS_project
pip install -r requirements.txt

python main.py train     # train + save the ML models
python main.py demo      # 8-process comparison, writes charts to outputs/
```

Full usage, every command and the project explanation are in
**[docs/USER_GUIDE.md](docs/USER_GUIDE.md)**.

## Sample result

8-process demo (`python main.py demo`), lower is better:

| Algorithm | Avg waiting (ms) | Avg turnaround (ms) | Avg response (ms) | Context switches |
|---|---|---|---|---|
| FCFS | 116.00 | 150.38 | 116.00 | 7 |
| SJF (oracle) | 97.38 | 131.75 | 97.38 | 7 |
| Round Robin (q=3) | 161.50 | 195.88 | 8.62 | 94 |
| Priority | 99.88 | 134.25 | 99.88 | 7 |
| **AI Scheduler** | **97.75** | **132.12** | **97.75** | **7** |

The AI scheduler closes about 98% of the FCFS-to-SJF waiting-time gap here
without being told any burst time. Model quality on held-out data: burst-time
MAE about 2.7 ms, priority accuracy about 92%.

![Metrics comparison](docs/images/metrics_comparison.png)
![Gantt charts](docs/images/gantt_charts.png)

## Repository layout

```
main.py                      command-line entry point
src/
  process.py                 Process object (the simulator's PCB)
  data_generation.py         Module 1: synthetic historical process data
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

Training data is synthetic, generated from a documented formula in
`src/data_generation.py`. The results show that the pipeline and the idea work;
validating on real process logs is listed as future scope.
