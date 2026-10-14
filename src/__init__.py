"""AI Task Scheduler - an ML-driven CPU process scheduling simulator.

Modules
-------
process              : the Process object (mirrors an OS Process Control Block)
data_generation      : Module 1 - simulates the historical process dataset
feature_engineering  : Module 2 - cleans / encodes / scales features for sklearn
prediction           : Module 3 - trains + serves the burst-time and priority models
schedulers           : Module 4 - FCFS, SJF, Round Robin, Priority and the AI policy
evaluation           : Module 5 - OS scheduling metrics + Gantt / bar-chart plots
"""

__version__ = "1.0.0"
