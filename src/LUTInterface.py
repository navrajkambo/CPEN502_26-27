from pathlib import Path
from BaseInterface import BaseInterface
import numpy as np

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
