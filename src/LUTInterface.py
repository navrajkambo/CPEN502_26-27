import time
import numpy as np
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
        kLinearPerf = "LinearPerf"
        kStep = "Step"

    start: float
    zero_at_battle: int
    mode: Decay = Decay.kStep
    perf_interval: int = 200
    perf_rounds: int = 10

    def __post_init__(self):
        # check ranges and make sure that decay type is defined
        if not 0 <= self.start <= 1:
            raise ValueError("Starting epsilon must be in [0, 1]")
        if (not isinstance(self.zero_at_battle, (int, np.integer))
                or isinstance(self.zero_at_battle, bool) or self.zero_at_battle < 0):
            raise ValueError("zero_at_battle must be a nonnegative integer")
        if not isinstance(self.mode, self.Decay):
            raise ValueError("Epsilon decay mode must be Linear, LinearPerf or Step")
        if (isinstance(self.perf_interval, bool)
                or not isinstance(self.perf_interval, (int, np.integer))
                or self.perf_interval < 1):
            raise ValueError("Performance interval must be a positive integer")
        if (isinstance(self.perf_rounds, bool)
                or not isinstance(self.perf_rounds, (int, np.integer))
                or not 1 <= self.perf_rounds < self.perf_interval):
            raise ValueError("Performance rounds must be positive and less than the interval")

    def is_performance_round(self, battle: int) -> bool:
        return (self.mode == self.Decay.kLinearPerf
                and self.perf_interval <= battle < self.zero_at_battle
                and battle % self.perf_interval < self.perf_rounds)

    def value(self, battle: int) -> float:
        if battle >= self.zero_at_battle: # zero epsilon value after max battle
            return 0.0
        if self.mode == self.Decay.kStep: # keep epsilon constant
            return self.start
        if self.is_performance_round(battle):
            return 0.0
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

    def initialize_random(self, low: float = -0.5, high: float = 0.5) -> None:
        low, high = float(low), float(high)
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError("Random initialization bounds must be finite and low < high")
        self.Q[...] = self.rng.uniform(low, high, size=self.Q.shape)
        self.update_counts.fill(0)

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

    def configure_epsilon(self, start: float, zero_at_battle: int,
                          mode: EpsilonSchedule.Decay = EpsilonSchedule.Decay.kLinear,
                          perf_interval: int = 200, perf_rounds: int = 10) -> None:
        self.epsilon_schedule = EpsilonSchedule(start, zero_at_battle, mode,
                                                perf_interval, perf_rounds)
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
            "epsilon_perf_interval": self.epsilon_schedule.perf_interval,
            "epsilon_perf_rounds": self.epsilon_schedule.perf_rounds,
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
            schedule = None
            if "epsilon_zero_at_battle" in data:
                # Match legacy lowercase strings as well as multiword modes.
                saved_mode = str(data["epsilon_decay_mode"].item()).casefold()
                modes = {mode.value.casefold(): mode for mode in EpsilonSchedule.Decay}
                if saved_mode not in modes:
                    raise ValueError(f"Unknown saved epsilon decay mode: {saved_mode}")
                schedule = EpsilonSchedule(
                    float(data["epsilon_start"]),
                    data["epsilon_zero_at_battle"].item(), modes[saved_mode],
                    data["epsilon_perf_interval"].item() if "epsilon_perf_interval" in data else 200,
                    data["epsilon_perf_rounds"].item() if "epsilon_perf_rounds" in data else 10)
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

    # CHATGPT
    def plotLearningPerformance(self, filename: str, window: int = 50, output_path: str | None = None, show: bool = False) -> tuple:
        """Compatibility method; the implementation lives in test/plotter/training.py."""
        import sys
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        from test.plotter.training import plot_learning_performance
        return plot_learning_performance(filename, window, output_path, show)
