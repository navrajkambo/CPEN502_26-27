# CapPeeler Q-learning

## Assignment (2): state and action LUT

The observation is quantized only when indexing the table:

| State variable | Quantization | Bins |
| --- | --- | --- |
| Enemy distance | [0,150), [150,400), [400,infinity) | 3 |
| Relative bearing | front [-45,45), right [-135,-45), back [-180,-135) or [135,180), left [45,135) degrees | 4 |
| Own energy | [0,30), [30,70), [70,infinity) | 3 |
| Enemy energy | [0,30), [30,70), [70,infinity) | 3 |

Bearings are normalized to [-180,180). There are 3*4*3*3 = **108 states**.
The six discrete actions are forward, backward, left, right, fire, and face enemy.
They are held for five ticks; radar scanning and gun aiming run independently.
Thus the state-row representation has **108 rows and 6 action columns**. If each
row represents a complete state-action pair, there are **648 rows/Q-values**.
The actual NumPy array has shape (3,4,3,3,6). Actions are a predefined discrete
set, rather than a quantization of all possible continuous control combinations.

## Assignment (3a): measured learning progress

The existing `training/cappeeler.csv` contains 8,513 consecutive completed episodes
at the time of this analysis. `training/cappeeler_progress.png` plots this measured
history. The metric is rolling win rate over 50 rounds, alongside mean reward.

| Window | Win rate | Mean episode reward |
| --- | ---: | ---: |
| First 50 episodes | 16% | -98.16 |
| Final 50 episodes | 96% | 128.64 |
| Episodes 7,514-8,013 | 87.6% | 97.562 |
| Episodes 8,014-8,513 | 94% | 111.308 |

These results show improved recorded performance. They do not establish formal
convergence: the recent blocks still differ, alpha remains constant at 0.1, and
epsilon remains at 0.05. This is one training history, including exploratory
actions. The CSV does not record opponent, battle seed, or arena settings, so
consistency of those conditions cannot be verified retrospectively. Do not infer
causality or a statistically established plateau from this curve alone.

## Assignments (3b), (3c), (4a): controlled comparisons

The robot now supports the following settings, set **before launching the GUI**
so its child bot processes inherit them:

| Environment variable | Values | Default |
| --- | --- | --- |
| `CAPPEELER_ALGORITHM` | `qlearning` (off-policy), `sarsa` (on-policy) | `qlearning` |
| `CAPPEELER_REWARDS` | `intermediate`, `terminal` | `intermediate` |
| `CAPPEELER_EPSILON` | Fixed probability in [0,1], including 0 | Unset: existing decay schedule |
| `CAPPEELER_SEED` | Integer action-selection random seed | Unset |
| `CAPPEELER_QTABLE` | Unique absolute filename for each run | Existing CapPeeler table |

Q-learning bootstraps from max_a Q(s',a). SARSA bootstraps from Q(s',a'), where
a' is selected with the current epsilon-greedy policy **and executed next**.
Both use reward alone on terminal transitions.

Terminal-only mode ignores bullet-hit, damage, wall, and collision rewards;
it retains +100 for a win, -100 for death, and 0 terminal bonus for a draw.
Zero-reward intermediate transitions still perform TD updates. The default mode
also uses +10 for bullet hits, -10 for being hit, -5 for walls, and -3 for collisions.
Terminal-only feedback is expected to provide less immediate guidance, but its
measured effect on this robot remains to be determined.

For example, configure a fresh SARSA run in PowerShell:

```powershell
$env:CAPPEELER_ALGORITHM = 'sarsa'
$env:CAPPEELER_REWARDS = 'intermediate'
$env:CAPPEELER_EPSILON = '0.1'
$env:CAPPEELER_SEED = '1'
$env:CAPPEELER_QTABLE = Join-Path (Get-Location) 'training/sarsa_seed1.npz'
```

Then launch the GUI from this shell and run the battle. Use a fresh, unique table
path for every condition and seed; do not reuse the existing trained table or
switch configurations within one history. Explicitly set both algorithm and
reward mode for each condition. An existing table automatically resumes training.
The seed above controls action selection, not battle randomness.

For (3b), train Q-learning and SARSA from zero tables with the same fixed epsilon
(e.g. 0.1), intermediate rewards, opponent, arena, and episode budget.
For (3c), compare intermediate and terminal rewards with the same algorithm and
epsilon. Use **win rate** as the common performance metric: their reward totals
have different definitions and cannot be compared directly. Also inspect episode
duration and battle replays to support claims about movement or firing behaviour.
For (4a), train independent Q-learning runs at fixed epsilon values
**0, 0.1, 0.3, 0.5, 1.0**, holding the other settings constant.
Epsilon=0 disables explicit random exploration; greedy ties are still broken
randomly. A fixed zero stays zero at round end.

Use equal budgets (e.g. 2,000 episodes per run) and repeat with several independent
seeds. Keep the opponent and battle settings recorded separately. Compare rolling
win rate against completed training episodes, and final-window win rate against
epsilon. Training win rate includes the immediate cost of exploration; separate
greedy evaluation battles are needed to measure learned-policy quality alone.

Generate the comparison figures from the actual CSVs after completing battles:

```powershell
.venv/Scripts/python.exe src/compare_learning.py --run Q-learning=training/qlearning_seed1.csv --run SARSA=training/sarsa_seed1.csv --output training/policy_comparison.png
.venv/Scripts/python.exe src/compare_learning.py --run Intermediate=training/intermediate_seed1.csv --run Terminal=training/terminal_seed1.csv --output training/reward_comparison.png
.venv/Scripts/python.exe src/compare_learning.py --run e0=training/e0_seed1.csv --run e0.1=training/e01_seed1.csv --run e0.3=training/e03_seed1.csv --run e0.5=training/e05_seed1.csv --run e1=training/e1_seed1.csv --output training/exploration_curves.png --epsilon-summary training/exploration_summary.png
```

Each line is one seed's comparison; repeat for other seeds to check variability.
The epsilon summary requires equal episode counts and constant epsilon in each
CSV. It plots win rate over the final 50 episodes by default (`--window` changes
this). It is descriptive and does not provide confidence intervals.

**Data status:** only the existing Q-learning/intermediate-reward/decaying-epsilon
history is currently available. Comparative battle measurements and conclusions
for (3b), (3c), and (4a) are pending; no synthetic battle results are substituted.

After the configured rounds finish, a plot window opens automatically with
round reward, average reward and win rate over the last 50 rounds, and epsilon
against completed training rounds. Early averages use the available rounds.
The figure includes the full saved training history and is also saved as
`training/cappeeler.png` (or beside your custom CSV). These are training results
including exploration, rather than separate zero-epsilon evaluation results.
The epsilon curve shows the recorded value after each round.

To view the plot later without running a battle, use this command from the
repository root:

    .venv/Scripts/python.exe src/plot_learning.py --show

Use `--window 100` to change the averaging window, or pass a custom CSV filename.
`LUTInterface.plotLearningPerformance(filename, window=50, show=False)` saves the
figure and returns `(figure, axes)` for use from Python. The bot launches the
viewer in a separate process at game end so the plot does not block bot events.

CapPeeler trains automatically while fighting. Add `src/bots` as a bot directory
in Tank Royale, select CapPeeler and an opponent (start with Tracker), and run a
battle with many rounds. Install the root requirements into `.venv` first; the
bot launchers use that environment. Each round is one episode. No separate battle
controller is required: choose the round count in the GUI.

The algorithm follows the Q-learning algorithm discussed in the lecture notes:

    Q(s,a) += alpha * (reward + gamma * max(Q(next_state)) - Q(s,a))

Terminal transitions use reward alone. Defaults: alpha=0.1, gamma=0.95,
epsilon=1.0, decaying by 0.995 each completed round to a minimum of 0.05.
Greedy action selection breaks ties randomly.

`src/LUTInterface.py` implements Q-learning, the `State` dataclass, six actions,
and the abstract `BaseInterface` methods `saveWeights` and `loadWeights`. These
persist Q-values, epsilon and episode count in the existing `.npz` format.
`bots/CapPeeler/CapPeeler.py` is my robot.
`State` stores raw distance, bearing and energies as an immutable observation.
Its `distance_bin`, `bearing_bin`, `own_energy_bin`, and `enemy_energy_bin`
properties discretize those measurements. The LUT uses `state.to_tuple()` only
when indexing the NumPy table; action selection and updates accept `State` objects.
The existing neural-network interfaces from assignment #1 are independent and unchanged.

State order: distance, bearing, own energy, enemy energy (3 x 4 x 3 x 3).
Distance bins are [0,150), [150,400), [400,inf). Energy bins are [0,30), [30,70), [70,inf).
Bearing bins are front [-45,45), right [-135,-45), left [45,135), and back.
Positive bearings are ccw. There are 648 Q-values.
These numerical thresholds are starting choices; The hope is to optimize these.

Actions last five game ticks: forward (+8 speed), backward (-8), left (+10
degrees/tick), right (-10), fire (power 3 when aimed within 10 degrees and energy >=3), 
or face enemy. Radar scans continuously and the gun aims independently.
Actions use nonblocking commands; callbacks never call blocking movement methods.
Tick decisions run after scan and reward callbacks. The latest scanned enemy
position is used between scans; after 30 ticks aim/fire stop and any new target
can be acquired. This initial state space has no explicit unknown-target state.

Rewards: 
- bullet hit +10
- hit by bullet -10
- wall -5
- bot collision -3
- win +100
- death -100

A surviving round without a win/death event is a draw with
no terminal bonus. Rewards accumulate when events arrive, including delayed
bullet hits; they update the active transition, without retroactive bullet credit
assignment or extra energy shaping. Terminal callbacks collect rewards; after final events are flushed, the next round start or game end
performs one final update, including rounds shorter than five ticks.

Training outputs (saved at the next round start or game end):

- `training/cappeeler.npz`: Q-values, epsilon and completed episode count;
  loaded automatically on the next launch.
- `training/cappeeler.csv`: episode reward, outcome, turns, action count and
  epsilon. Compute win rate and average reward across a window of these rows.

To evaluate, set `CAPPEELER_TRAINING=0` in the environment **before starting the
GUI/bot process**. Evaluation chooses greedy actions and does not update Q-values,
decay epsilon or write training files. Training is enabled by default.
`CAPPEELER_QTABLE` overrides the table location (use an absolute path); the CSV
uses the same filename stem. Use a new filename to start fresh. Run one training
CapPeeler process per table path to avoid concurrent writes.

Run the focused checks from the repository root:

    .venv/Scripts/python.exe -B -m unittest discover -s test -v
