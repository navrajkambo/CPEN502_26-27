from pathlib import Path
import csv
from BaseInterface import BaseInterface
import numpy as np
import matplotlib.pyplot as plt

class LUTInterface(BaseInterface):
    def __init__(self, state_shape=(3, 4, 3, 3), n_actions: int = 6, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 1.0, seed=None):
        super().__init__()
        if not all(0 <= value <= 1 for value in (alpha, gamma, epsilon)):
            raise ValueError("alpha, gamma and epsilon must be in [0, 1]")
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((*state_shape, n_actions), dtype=float)
        self.episodes: int = 0

    def choose_action(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.Q.shape[-1]))
        values = self.Q[state]
        return int(self.rng.choice(np.flatnonzero(values == values.max())))

    def update(self, state, action, reward, next_state=None, done=False) -> float:
        current = self.Q[state + (action,)]
        target = reward if done else reward + self.gamma * self.Q[next_state].max()
        td_error = float(target - current)
        self.Q[state + (action,)] += self.alpha * td_error
        return td_error

    def decay_epsilon(self, decay=0.995, minimum=0.05):
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

    def plotLearningPerformance(self, filename, window: int = 50,
                                output_path=None, show: bool = False):
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
