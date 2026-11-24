import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from enum import IntEnum, Enum
from dataclasses import dataclass

from BaseInterface import BaseInterface

class Action(IntEnum):
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    FIRE = 4
    FACE_ENEMY = 5


@dataclass(frozen=True)
class State: # an observation snapshot, converted to discrete bins only for LUT indexing
    distance: int | float
    bearing: int | float
    own_energy: int | float
    enemy_energy: int | float

    def _energy_bin(self, energy: int | float) -> int:
        return 0 if energy < 30 else 1 if energy < 70 else 2

    @property
    def distance_bin(self) -> int:
        return 0 if self.distance < 150 else 1 if self.distance < 400 else 2

    @property
    def bearing_bin(self) -> int:
        bearing = (self.bearing + 180) % 360 - 180
        return 0 if -45 <= bearing < 45 else (3 if 45 <= bearing < 135 else 1 if -135 <= bearing < -45 else 2)

    @property
    def own_energy_bin(self) -> int:
        return self._energy_bin(self.own_energy)

    @property
    def enemy_energy_bin(self) -> int:
        return self._energy_bin(self.enemy_energy)

    def to_tuple(self) -> tuple[int, int, int, int]:
        return self.distance_bin, self.bearing_bin, self.own_energy_bin, self.enemy_energy_bin


@dataclass(frozen=True)
class EpsilonSchedule:
    class Decay(Enum):
        kLinear = "Linear"
        kStep = "Step"

    start: float
    zero_at_battle: int
    mode: Decay = Decay.kStep

    def __post_init__(self):
        # check ranges and make sure that decay type is defined
        if not 0 <= self.start <= 1:
            raise ValueError("Starting epsilon must be in [0, 1]")
        if (not isinstance(self.zero_at_battle, (int, np.integer))
                or isinstance(self.zero_at_battle, bool) or self.zero_at_battle < 0):
            raise ValueError("zero_at_battle must be a nonnegative integer")
        if not isinstance(self.mode, self.Decay):
            raise ValueError("Epsilon decay mode must be Linear or Step")

    def value(self, battle: int) -> float:
        if battle >= self.zero_at_battle: # zero epsilon value after max battle
            return 0.0
        if self.mode == self.Decay.kStep: # keep epsilon constant
            return self.start
        return self.start * (1 - battle / self.zero_at_battle) # reduce epsilon proportional to current battle / total battles

class LUTInterface(BaseInterface):
    class Algorithm(Enum):
        kQLearning = "qlearning"
        kSarsa = "sarsa"

    class RewardMode(Enum):
        kIntermediate = "intermediate"
        kTerminal = "terminal"

    def __init__(self, state_shape: tuple = (3, 4, 3, 3), n_actions: int = 6, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 1.0, seed=None):
        super().__init__()
        if not all(0 <= value <= 1 for value in (alpha, gamma, epsilon)):
            raise ValueError("alpha, gamma and epsilon must be in [0, 1]")
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon # hyper params
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((*state_shape, n_actions), dtype=float)
        self.update_counts = np.zeros(self.Q.shape, dtype=np.int64) # Count training updates (discussed in lecture)
        self.episodes: int = 0
        self.epsilon_schedule: EpsilonSchedule | None = None

    @property
    def state_update_counts(self) -> np.ndarray:
        return self.update_counts.sum(axis=-1) # used for part 3 - remove under-visited states

    def choose_action(self, state: State, explore: bool = True) -> int:
        if explore and (self.rng.random() < self.epsilon): # if epsilon > random number between (0,1) (epsilon is percentage of random exploratory moves)
            return int(self.rng.integers(self.Q.shape[-1])) # random number between 0 and 5
        values = self.Q[state.to_tuple()] # choose largest value (highest probability)
        return int(self.rng.choice(np.flatnonzero(values == values.max()))) # choose randomly if there is a tie for largest value

    def update(self, state: State, action: Action | int, reward: int | float, next_state: State | None = None, done: bool = False, next_action: Action | int | None = None) -> float:
        """Q-learning by default; supply the actual next action for SARSA."""
        if not done and next_state is None:
            raise ValueError("next_state is required for a non-terminal update")
        index = state.to_tuple() + (action,) # append action to end of state tuple
        current = self.Q[index] # search Q-table
        if done:
            target = reward
        else:
            values = self.Q[next_state.to_tuple()]
            bootstrap = values.max() if next_action is None else values[next_action]
            target = reward + self.gamma * bootstrap
        td_error = float(target - current)
        self.Q[index] += self.alpha * td_error # add difference of target and current (scaled by alpha) to Q value
        self.update_counts[index] += 1
        return td_error

    def configure_epsilon(self, start: float, zero_at_battle: int, mode: EpsilonSchedule.Decay = EpsilonSchedule.Decay.kLinear) -> None:
        self.epsilon_schedule = EpsilonSchedule(start, zero_at_battle, mode)
        self.epsilon = self.epsilon_schedule.value(self.episodes)

    def decay_epsilon(self) -> None:
        if self.epsilon_schedule is not None:
            self.epsilon = self.epsilon_schedule.value(self.episodes)

####################################### CHATGPT METHODS BELOW ###################################################################

    def saveWeights(self, filename: str = "weights.npz") -> None:
        # save Q-values, update counts,  exploration rate and completed training episodes (ChatGPT)
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        schedule_data = {} if self.epsilon_schedule is None else {
            "epsilon_start": self.epsilon_schedule.start,
            "epsilon_zero_at_battle": self.epsilon_schedule.zero_at_battle,
            "epsilon_decay_mode": self.epsilon_schedule.mode.value,
        }
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, Q=self.Q, update_counts=self.update_counts,
                                epsilon=self.epsilon, episodes=self.episodes,
                                **schedule_data)
        # Windows readers (for example an editor or scanner) may briefly prevent
        # replacing an open checkpoint. Retry only that transient access error;
        # preserve the old checkpoint and completed temporary file on failure.
        for attempt in range(5):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * 2 ** attempt)

    def loadWeights(self, filename: str = "weights.npz") -> None:
        # load training state, validating it before replacing the current table (ChatGPT)
        with np.load(filename, allow_pickle=False) as data:
            values = np.array(data["Q"], dtype=float, copy=True)
            epsilon = float(data["epsilon"])
            episodes = int(data["episodes"]) if "episodes" in data else 0
            # Older files do not contain historical update counts. Start tracking
            # from zero while retaining their learned Q-values and metadata.
            counts = (np.array(data["update_counts"], copy=True)
                      if "update_counts" in data else np.zeros(values.shape, dtype=np.int64))
            schedule = (EpsilonSchedule(float(data["epsilon_start"]),
                                        data["epsilon_zero_at_battle"].item(),
                                        EpsilonSchedule.Decay(str(data["epsilon_decay_mode"].item()).capitalize()))
                        if "epsilon_zero_at_battle" in data else None)
        if values.shape != self.Q.shape or not np.isfinite(values).all():
            raise ValueError("Saved Q-table has incompatible shape or invalid values")
        if not 0 <= epsilon <= 1 or episodes < 0:
            raise ValueError("Invalid saved training metadata")
        if (counts.shape != self.Q.shape or counts.dtype.kind not in "iu"
                or np.any(counts < 0) or np.any(counts > np.iinfo(np.int64).max)):
            raise ValueError("Saved update counts must be nonnegative integers matching the Q-table shape")
        self.Q, self.epsilon, self.episodes = values, epsilon, episodes
        self.update_counts = counts.astype(np.int64)
        self.epsilon_schedule = schedule

    def plotLearningPerformance(self, filename: str, window: int = 50, output_path: str | None = None, show: bool = False) -> tuple: # CHATGPT
        """Plot training rewards, win rate and epsilon from the round history.

        Rolling statistics include up to `window` rounds, including the current
        round. Early points use all rounds available. Epsilon is the recorded
        value after each round; these are training results including exploration.
        """
        if not isinstance(window, int) or window < 1:
            raise ValueError("window must be a positive integer")
        path = Path(filename)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError("No completed rounds to plot")
        episodes = np.array([int(row["episode"]) for row in rows])
        rewards = np.array([float(row["reward"]) for row in rows])
        wins = np.array([row["outcome"] == "win" for row in rows], dtype=float)
        epsilon = np.array([float(row["epsilon"]) for row in rows])

        def rolling_mean(values):
            totals = np.concatenate(([0.0], np.cumsum(values)))
            ends = np.arange(1, len(values) + 1)
            starts = np.maximum(0, ends - window)
            return (totals[ends] - totals[starts]) / (ends - starts)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                                 constrained_layout=True)
        fig.suptitle("CapPeeler training performance (includes exploration)")
        axes[0].plot(episodes, rewards, color="lightsteelblue", alpha=0.6,
                     label="Round reward")
        axes[0].plot(episodes, rolling_mean(rewards), color="tab:blue",
                     label=f"Average over last {window} rounds")
        axes[0].set_ylabel("Reward")
        axes[0].legend(loc="best")
        axes[1].plot(episodes, 100 * rolling_mean(wins), color="tab:red")
        axes[1].set_ylabel(f"Win rate (%)\nLast {window} rounds")
        axes[1].set_ylim(0, 100)
        axes[2].plot(episodes, epsilon, color="tab:green")
        axes[2].set_ylabel("Epsilon after round")
        axes[2].set_ylim(0, 1.05)
        axes[2].set_xlabel("Completed training rounds (battles / episodes)")
        for ax in axes:
            ax.grid(True, alpha=0.25)
        destination = Path(output_path) if output_path else path.with_suffix(".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=150)
        if show:
            plt.show()
        return fig, axes
