"""CapPeeler: five-tick actions trained with a persistent Q-learning LUT."""
import csv
import os
from pathlib import Path
import sys
import subprocess
import traceback

# robocode launches this file from the bot directory.
SOURCE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_DIR))

from LUTInterface import LUTInterface, Action, State
from BotDiagnostics import BotDiagnostics
from robocode_tank_royale.bot_api import Color
from robocode_tank_royale.bot_api.bot import Bot
# from robocode_tank_royale.bot_api.bot_info import BotInfo
from robocode_tank_royale.bot_api.events import (
    BotDeathEvent,
    BulletHitBotEvent,
    DeathEvent,
    GameEndedEvent,
    HitBotEvent,
    HitByBulletEvent,
    HitWallEvent,
    RoundEndedEvent,
    RoundStartedEvent,
    ScannedBotEvent,
    TickEvent,
    WonRoundEvent,
)


class CapPeeler(Bot):
    ACTION_TICKS = 5
    TARGET_TIMEOUT = 30

    def __init__(self, epsilon_start: float = 0.8,
                 epsilon_zero_at_battle: int | None = None,
                 epsilon_decay: str = "linear"):
        super().__init__()
        self.training = os.getenv("CAPPEELER_TRAINING", "1") != "0"
        self.table_path = Path(os.getenv("CAPPEELER_QTABLE", str(SOURCE_DIR.parent / "training" / "cappeeler.npz")))
        self.metrics_path = self.table_path.with_suffix(".csv")
        self.qtable = LUTInterface()
        self.algorithm = os.getenv("CAPPEELER_ALGORITHM", "qlearning").lower()
        self.reward_mode = os.getenv("CAPPEELER_REWARDS", "intermediate").lower()
        if self.algorithm not in ("qlearning", "sarsa"):
            raise ValueError("CAPPEELER_ALGORITHM must be qlearning or sarsa")
        if self.reward_mode not in ("intermediate", "terminal"):
            raise ValueError("CAPPEELER_REWARDS must be intermediate or terminal")
        self.fixed_epsilon = os.getenv("CAPPEELER_EPSILON")
        if self.fixed_epsilon is not None:
            self.fixed_epsilon = float(self.fixed_epsilon)
            if not 0 <= self.fixed_epsilon <= 1:
                raise ValueError("CAPPEELER_EPSILON must be in [0, 1]")
        seed = os.getenv("CAPPEELER_SEED")
        if seed is not None:
            import numpy as np
            self.qtable.rng = np.random.default_rng(int(seed))
        if self.table_path.exists():
            self.qtable.loadWeights(self.table_path)
        elif not self.training:
            raise FileNotFoundError(f"Train a table before evaluation: {self.table_path}")
        cutoff = os.getenv("CAPPEELER_EPSILON_ZERO_AT")
        if cutoff is not None:
            epsilon_zero_at_battle = int(cutoff)
        if epsilon_zero_at_battle is not None:
            self.qtable.configure_epsilon(
                start=float(os.getenv("CAPPEELER_EPSILON_START", str(epsilon_start))),
                zero_at_battle=epsilon_zero_at_battle,
                mode=os.getenv("CAPPEELER_EPSILON_DECAY", epsilon_decay).lower(),
            )
        if self.fixed_epsilon is not None:
            self.qtable.epsilon = self.fixed_epsilon
        # Process scans, damage and death before each learning decision.
        self.set_event_priority(TickEvent, 0)
        self._reset_episode()
        self.diagnostics = None
        if os.getenv("CAPPEELER_DIAGNOSTICS", "1") != "0":
            self.diagnostics = BotDiagnostics(self.table_path.with_suffix(".diagnostics.log"))

    def _reset_episode(self):
        self._target = None
        self._state: State | None = None
        self._action = None
        self._action_turn = 0
        self._reward = 0.0
        self._total_reward = 0.0
        self._steps = 0
        self._outcome = None
        self._finished = False
        self._round_end = None

    def on_round_started(self, e: RoundStartedEvent) -> None:
        self._heartbeat(f"starting round {e.round_number}")
        self._finish_episode()
        self._reset_episode()

    def _heartbeat(self, phase):
        diagnostics = getattr(self, "diagnostics", None)
        if diagnostics is not None:
            diagnostics.heartbeat(phase)

    def run(self):
        self.body_color = Color.from_rgb(255, 0, 255) # purple
        self.turret_color = Color.from_rgb(255, 255, 255)
        self.adjust_gun_for_body_turn = True
        self.adjust_radar_for_body_turn = True
        self.adjust_radar_for_gun_turn = True
        while self.running:
            self.go() # this is a baseclass inherited method, dispatches callbacks (nonblocking commands only)

    def on_scanned_bot(self, e: ScannedBotEvent) -> None:
        if self._outcome is not None or self._finished or self._round_end is not None:
            return
        if (self._target is None or e.scanned_bot_id == self._target[0]
                or e.turn_number - self._target[4] > self.TARGET_TIMEOUT):
            self._target = (e.scanned_bot_id, e.x, e.y,
                            e.energy, e.turn_number)

    def _observation(self) -> State:
        _, x, y, energy, _ = self._target
        return State(distance=self.distance_to(x, y), bearing=self.bearing_to(x, y),
                     own_energy=self.energy, enemy_energy=energy)

    def on_tick(self, e: TickEvent) -> None:
        if self._outcome is not None or self._finished or self._round_end is not None:
            return
        self._heartbeat(f"tick {e.turn_number}")
        self.radar_turn_rate = 45
        if self._target is None:
            self.target_speed = 0
            self.turn_rate = 0
            self.set_fire(0)
            return
        state = self._observation()
        if self._state is None or e.turn_number - self._action_turn >= self.ACTION_TICKS:
            # SARSA must bootstrap from the action actually executed next.
            sarsa = getattr(self, "algorithm", "qlearning") == "sarsa"
            next_action = self.qtable.choose_action(state, explore=self.training) if sarsa else None
            if self._state is not None:
                if self.training:
                    self.qtable.update(self._state, self._action, self._reward, state,
                                       next_action=next_action)
            self._reward = 0.0
            self._state = state
            self._action = next_action if sarsa else self.qtable.choose_action(state, explore=self.training)
            self._action_turn = e.turn_number
            self._steps += 1
        self._execute_action(e.turn_number)

    def _execute_action(self, turn):
        # Reset previous commands so actions do not leak into the next interval.
        self.target_speed = 0
        self.turn_rate = 0
        self.set_fire(0)
        _, x, y, _, seen = self._target
        fresh = turn - seen <= self.TARGET_TIMEOUT
        self.gun_turn_rate = self.gun_bearing_to(x, y) if fresh else 0
        action = Action(self._action)
        if action == Action.FORWARD:
            self.target_speed = 8
        elif action == Action.BACKWARD:
            self.target_speed = -8
        elif action == Action.LEFT:
            self.turn_rate = 10
        elif action == Action.RIGHT:
            self.turn_rate = -10
        elif action == Action.FACE_ENEMY and fresh:
            self.turn_rate = self.bearing_to(x, y)
        elif action == Action.FIRE and fresh:
            if abs(self.gun_bearing_to(x, y)) <= 10 and self.energy > 3:
                self.set_fire(3)

    def _add_reward(self, reward: int, terminal: bool = False):
        if getattr(self, "reward_mode", "intermediate") == "terminal" and not terminal:
            return
        if not self._finished:
            self._total_reward += reward
            if self._state is not None:
                self._reward += reward

    def on_bullet_hit(self, e: BulletHitBotEvent) -> None:
        self._add_reward(10)

    def on_hit_by_bullet(self, e: HitByBulletEvent) -> None:
        self._add_reward(-10)

    def on_hit_wall(self, e: HitWallEvent) -> None:
        self._add_reward(-5)

    def on_hit_bot(self, e: HitBotEvent) -> None:
        self._add_reward(-3)

    def on_bot_death(self, e: BotDeathEvent) -> None:
        if self._target is not None and e.victim_id == self._target[0]:
            # Keep the last observation for the pending transition; allow retargeting.
            self._target = (self._target[0], *self._target[1:4], -self.TARGET_TIMEOUT - 1)

    def on_death(self, e: DeathEvent) -> None:
        diagnostics = getattr(self, "diagnostics", None)
        if diagnostics is not None:
            # Continue monitoring: a stall can happen while waiting for the next
            # round or saving the preceding one, even after the bot has died.
            diagnostics.heartbeat(f"died on turn {e.turn_number}; waiting for round end")
            diagnostics.record(f"died on turn {e.turn_number}; waiting for round end")
        if self._outcome is None:
            self._outcome = "loss"
            self._add_reward(-100, terminal=True)

    def on_won_round(self, e: WonRoundEvent) -> None:
        if self._outcome is None:
            self._outcome = "win"
            self._add_reward(100, terminal=True)

    def on_round_ended(self, e: RoundEndedEvent) -> None:
        self._heartbeat(f"round {e.round_number} ended; waiting for next round or game end")
        self._round_end = e # API 1.4.0 flushes final events AFTER this callback returns.

    def on_game_ended(self, e: GameEndedEvent) -> None:
        diagnostics = getattr(self, "diagnostics", None)
        if diagnostics is not None:
            diagnostics.idle(f"game ended after {e.number_of_rounds} rounds")
        self._finish_episode()
        self._plot_performance()

    def _plot_performance(self):
        if (not self.training or not self.metrics_path.exists()
                or getattr(self, "_plotted_episode", None) == self.qtable.episodes):
            return
        # A separate process keeps the plotting GUI on its own main thread and
        # leaves the bot free to receive future game/server events.
        # Do not inherit the booter's pipes: a plot window can outlive the bot.
        # Keep errors in a file instead of sending them to the battle console.
        try:
            with self.metrics_path.with_suffix(".plot.log").open("a") as plot_log:
                subprocess.Popen(
                    [sys.executable, str(SOURCE_DIR / "plot_learning.py"),
                     str(self.metrics_path.resolve()), "--show"],
                    stdin=subprocess.DEVNULL, stdout=plot_log, stderr=subprocess.STDOUT,
                    close_fds=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
        except OSError as error:
            diagnostics = getattr(self, "diagnostics", None)
            if diagnostics is not None:
                diagnostics.record(f"Could not launch plot: {error}")
            return
        self._plotted_episode = self.qtable.episodes

    def on_game_aborted(self, e: None) -> None:
        diagnostics = getattr(self, "diagnostics", None)
        if diagnostics is not None:
            diagnostics.idle("game aborted")
        # Tank Royale supplies None for this callback, rather than an event object.
        self._finish_episode()

    def _finish_episode(self):
        e: RoundEndedEvent | None = self._round_end
        if e is None or self._finished:
            return
        # Finalize here, after all terminal-turn rewards have been collected.
        if self.training and self._state is not None:
            self.qtable.update(self._state, self._action, self._reward, done=True)
        self._finished = True
        if self.training:
            self.qtable.episodes += 1
            if getattr(self, "fixed_epsilon", None) is None:
                self.qtable.decay_epsilon()
            try:
                self.qtable.saveWeights(self.table_path)
            except OSError:
                diagnostics = getattr(self, "diagnostics", None)
                if diagnostics is not None:
                    diagnostics.record(f"Checkpoint save failed at episode {self.qtable.episodes}:\n"
                                       + traceback.format_exc())
                raise
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            header = not self.metrics_path.exists()
            with self.metrics_path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                if header:
                    writer.writerow(("episode", "round", "outcome", "reward",
                                     "turns", "actions", "epsilon"))
                writer.writerow((self.qtable.episodes, e.round_number,
                                 self._outcome or "draw", self._total_reward,
                                 e.turn_number, self._steps, self.qtable.epsilon))
        print(f"Round {e.round_number}: {self._outcome or 'draw'}, "
              f"reward={self._total_reward:g}, epsilon={self.qtable.epsilon:.3f}")


def main():
    bot = CapPeeler(
        epsilon_start=0.8,
        epsilon_zero_at_battle=7000,
        epsilon_decay="linear",  # or "step"
    )
    try:
        bot.start()
    finally:
        if bot.diagnostics is not None:
            bot.diagnostics.close()


if __name__ == "__main__":
    main()
