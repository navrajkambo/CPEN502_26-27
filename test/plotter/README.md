# Part 2 report plots

All Part 2 plotting implementations live in this folder. Run commands from the repository root.

Regenerate all ten report figures:

```powershell
venv/Scripts/python.exe -B -m test.plotter
```

The default destination is `doc/Part 2/figures/`. To use another directory:

```powershell
venv/Scripts/python.exe -B -m test.plotter --output tmp/plot-review
```

Plot a single saved battle history (the same plot that CapPeeler opens automatically):

```powershell
venv/Scripts/python.exe -B -m test.plotter.training "training/qlearning/cappeeler.csv" --show
```

Use `--window 100` to change its rolling window, or `--output path.png` to choose a destination. Without `--output`, this command saves beside its source CSV.

Compare arbitrary saved histories:

```powershell
venv/Scripts/python.exe -B -m test.plotter.compare --run "Q-learning=training/qlearning/cappeeler.csv" --run "SARSA=training/sarsa/cappeeler.csv" --output tmp/comparison.png
```

## Files

| Module | Responsibility |
| --- | --- |
| `common.py` | Read CSV/checkpoint metadata, reconstruct the epsilon used in each round, calculate rolling means and periodic greedy-window statistics |
| `training.py` | Per-round reward, rolling reward/win rate, and recorded epsilon |
| `report.py` | Policy comparison, reward comparison, exploration comparison/summary, and regeneration of the full report figure set |
| `initialization.py` | Zero versus random initialization, including the early-training view |
| `greedy.py` | Original versus kLinearPerf comparison and greedy-window-only plots |
| `compare.py` | Configurable comparison of arbitrary histories |

The existing commands in `src/plot_learning.py`, `src/analyze_experiments.py`, and `src/compare_learning.py`, and `LUTInterface.plotLearningPerformance`, delegate to this package. The bot still opens its saved training plot automatically.

## Sources and conventions

The exploration comparison uses `training/0 epsilon (random init)/` for its zero-epsilon curve, matching the report. `training/0 epsilon/` is the separate zero-initialized run. The direct initialization plot compares both. The periodic greedy experiment comes from `training/qlearning (online performance measurment)/` (folder spelling retained).

Raw training plots use trailing 50-round averages, including partial windows at the start. Report comparison curves use complete 200-round windows. Greedy-only points use each measurement block specified in its checkpoint; they exclude the permanent zero-epsilon phase after the cutoff. The CSV records epsilon after each round, so greedy-window selection uses the schedule at `episode - 1`.

The generator writes PNG figures and `measured_summary.json`. It does not edit Word/PDF documents, checkpoints, or training CSV files. Figures regenerated from CSV may differ in image encoding from an older saved training PNG.
