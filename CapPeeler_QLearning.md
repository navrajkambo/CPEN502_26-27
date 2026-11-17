# CapPeeler Q-learning

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
