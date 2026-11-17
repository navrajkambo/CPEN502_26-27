import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from enum import IntEnum
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

class LUTInterface(BaseInterface):
    def __init__(self, state_shape: tuple = (3, 4, 3, 3), n_actions: int = 6, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 1.0, seed=None):
        super().__init__()
        if not all(0 <= value <= 1 for value in (alpha, gamma, epsilon)):
            raise ValueError("alpha, gamma and epsilon must be in [0, 1]")
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((*state_shape, n_actions), dtype=float)
        self.episodes: int = 0

    def choose_action(self, state: State, explore: bool = True) -> int:
        if explore and (self.rng.random() < self.epsilon): # if epsilon > random number between (0,1) (epsilon is percentage of random exploratory moves)
            return int(self.rng.integers(self.Q.shape[-1])) # random number between 0 and 5
        values = self.Q[state.to_tuple()] # choose largest value (highest probability)
        return int(self.rng.choice(np.flatnonzero(values == values.max()))) # choose randomly if there is a tie for largest value

    def update(self, state: State, action: Action | int, reward: float, next_state: State | None = None, done: bool = False) -> float:
        if not done and next_state is None:
            raise ValueError("next_state is required for a non-terminal update")
        index = state.to_tuple() + (action,) # append action to end of state tuple
        current = self.Q[index] # search Q-table
        target = reward if done else reward + self.gamma * self.Q[next_state.to_tuple()].max()
        td_error = float(target - current)
        self.Q[index] += self.alpha * td_error # add difference of target and current (scaled by alpha) to Q value
        return td_error

    def decay_epsilon(self, decay: float = 0.995, minimum: float = 0.05) -> None:
        self.epsilon = max(minimum, self.epsilon * decay)

    def saveWeights(self, filename: str = "weights.npz") -> None:
        # save Q-values, exploration rate and completed training episodes
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, Q=self.Q, epsilon=self.epsilon, episodes=self.episodes)
        temporary.replace(path)

    def loadWeights(self, filename: str = "weights.npz") -> None:
        # load training state, validating it before replacing the current table
        with np.load(filename, allow_pickle=False) as data:
            values = np.array(data["Q"], dtype=float, copy=True)
            epsilon = float(data["epsilon"])
            episodes = int(data["episodes"]) if "episodes" in data else 0
        if values.shape != self.Q.shape or not np.isfinite(values).all():
            raise ValueError("Saved Q-table has incompatible shape or invalid values")
        if not 0 <= epsilon <= 1 or episodes < 0:
            raise ValueError("Invalid saved training metadata")
        self.Q, self.epsilon, self.episodes = values, epsilon, episodes

    def plotLearningPerformance(self, filename: str, window: int = 50,
                                output_path: str | None = None, show: bool = False) -> tuple:
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
