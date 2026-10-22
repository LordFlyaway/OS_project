"""Sanity tests - `python -m unittest` from the ai_task_scheduler/ folder."""

import copy
import unittest

from src.data_generation import generate_processes
from src.prediction import Predictor
from src.schedulers import ai_scheduler, fcfs, round_robin, sjf


class SchedulerInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.procs = generate_processes(15, seed=1)
        # save=False: the tests must never overwrite the real trained models
        Predictor.train_and_get(n_rows=1500, save=False).annotate(cls.procs)

    def test_all_processes_complete(self):
        for algo in (fcfs, sjf, round_robin, ai_scheduler):
            r = algo(copy.deepcopy(self.procs))
            self.assertEqual(len(r.processes), 15)
            self.assertTrue(all(p.completion_time is not None for p in r.processes))

    def test_no_negative_waiting(self):
        for algo in (fcfs, sjf, round_robin, ai_scheduler):
            r = algo(copy.deepcopy(self.procs))
            self.assertTrue(all(p.waiting_time >= 0 for p in r.processes))

    def test_work_conserving(self):
        # total CPU busy time must equal the sum of every burst
        total_burst = sum(p.actual_burst_time for p in self.procs)
        for algo in (fcfs, sjf, round_robin, ai_scheduler):
            r = algo(copy.deepcopy(self.procs))
            self.assertEqual(r.busy_time, total_burst)

    def test_sjf_beats_fcfs_on_waiting(self):
        f = fcfs(copy.deepcopy(self.procs))
        s = sjf(copy.deepcopy(self.procs))
        self.assertLessEqual(s.avg_waiting_time, f.avg_waiting_time)

    def test_ai_close_to_sjf(self):
        s = sjf(copy.deepcopy(self.procs))
        a = ai_scheduler(copy.deepcopy(self.procs))
        f = fcfs(copy.deepcopy(self.procs))
        # AI should land much nearer SJF than the naive FCFS baseline
        self.assertLess(a.avg_waiting_time - s.avg_waiting_time,
                        0.5 * (f.avg_waiting_time - s.avg_waiting_time) + 1)


if __name__ == "__main__":
    unittest.main()
