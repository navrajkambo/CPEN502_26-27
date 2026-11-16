"""CapPeeler: five-tick actions trained with a persistent Q-learning LUT."""
import csv
import os
from pathlib import Path
import sys
import subprocess

# robocode launches this file from the bot directory.
SOURCE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_DIR))

from LUTInterface import LUTInterface, Action, encode_state
from robocode_tank_royale.bot_api import Color
from robocode_tank_royale.bot_api.bot import Bot
# from robocode_tank_royale.bot_api.bot_info import BotInfo
from robocode_tank_royale.bot_api.events import TickEvent, ScannedBotEvent, HitBotEvent, WonRoundEvent, RoundEndedEvent, DeathEvent


class CapPeeler(Bot):
    ACTION_TICKS = 5
    TARGET_TIMEOUT = 30

    def __init__(self):
        super().__init__()
        self.training = os.getenv("CAPPEELER_TRAINING", "1") != "0"
        self.table_path = Path(os.getenv("CAPPEELER_QTABLE", str(SOURCE_DIR.parent / "training" / "cappeeler.npz")))
        self.metrics_path = self.table_path.with_suffix(".csv")
        self.qtable = LUTInterface()
        if self.table_path.exists():
            self.qtable.loadWeights(self.table_path)
        elif not self.training:
            raise FileNotFoundError(f"Train a table before evaluation: {self.table_path}")
        # Process scans, damage and death before each learning decision.
        self.set_event_priority(TickEvent, 0)
        self._reset_episode()

    def _reset_episode(self):
        self._target = None
        self._state = None
        self._action = None
        self._action_turn = 0
        self._reward = 0.0
        self._total_reward = 0.0
        self._steps = 0
        self._outcome = None
        self._finished = False
        self._round_end = None

    def on_round_started(self, event):
        self._finish_episode()
        self._reset_episode()

    def run(self):
        self.body_color = Color.from_rgb(255, 0, 255) # purple
        self.turret_color = Color.from_rgb(255, 255, 255)
        self.adjust_gun_for_body_turn = True
        self.adjust_radar_for_body_turn = True
        self.adjust_radar_for_gun_turn = True
        while self.running:
            self.go() # this is a baseclass inherited method, dispatches callbacks (nonblocking commands only)

    def on_scanned_bot(self, event: ScannedBotEvent):
        if self._outcome is not None or self._finished or self._round_end is not None:
            return
        if (self._target is None or event.scanned_bot_id == self._target[0]
                or event.turn_number - self._target[4] > self.TARGET_TIMEOUT):
            self._target = (event.scanned_bot_id, event.x, event.y,
                            event.energy, event.turn_number)

    def _observation(self):
        _, x, y, energy, _ = self._target
        return encode_state(self.distance_to(x, y), self.bearing_to(x, y),
                            self.energy, energy)

    def on_tick(self, event):
        if self._outcome is not None or self._finished or self._round_end is not None:
            return
        self.radar_turn_rate = 45
        if self._target is None:
            self.target_speed = 0
            self.turn_rate = 0
            self.set_fire(0)
            return
        state = self._observation()
        if self._state is None or event.turn_number - self._action_turn >= self.ACTION_TICKS:
            if self._state is not None:
                if self.training:
                    self.qtable.update(self._state, self._action, self._reward, state)
            self._reward = 0.0
            self._state = state
            self._action = self.qtable.choose_action(state, explore=self.training)
            self._action_turn = event.turn_number
            self._steps += 1
        self._execute_action(event.turn_number)

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

    def _add_reward(self, reward):
        if not self._finished:
            self._total_reward += reward
            if self._state is not None:
                self._reward += reward

    def on_bullet_hit(self, event):
        self._add_reward(10)

    def on_hit_by_bullet(self, event):
        self._add_reward(-10)

    def on_hit_wall(self, event):
        self._add_reward(-5)

    def on_hit_bot(self, event):
        self._add_reward(-3)

    def on_bot_death(self, event):
        if self._target is not None and event.victim_id == self._target[0]:
            # Keep the last observation for the pending transition; allow retargeting.
            self._target = (self._target[0], *self._target[1:4], -self.TARGET_TIMEOUT - 1)

    def on_death(self, event):
        if self._outcome is None:
            self._outcome = "loss"
            self._add_reward(-100)

    def on_won_round(self, event):
        if self._outcome is None:
            self._outcome = "win"
            self._add_reward(100)

    def on_round_ended(self, event):
        self._round_end = event # API 1.4.0 flushes final events AFTER this callback returns.

    def on_game_ended(self, event):
        self._finish_episode()
        self._plot_performance()

    def _plot_performance(self):
        if (not self.training or not self.metrics_path.exists()
                or getattr(self, "_plotted_episode", None) == self.qtable.episodes):
            return
        # A separate process keeps the plotting GUI on its own main thread and
        # leaves the bot free to receive future game/server events.
        subprocess.Popen(
            [sys.executable, str(SOURCE_DIR / "plot_learning.py"),
             str(self.metrics_path.resolve()), "--show"],
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self._plotted_episode = self.qtable.episodes

    def on_game_aborted(self, event):
        self._finish_episode()

    def _finish_episode(self):
        event = self._round_end
        if event is None or self._finished:
            return
        # Finalize here, after all terminal-turn rewards have been collected.
        if self.training and self._state is not None:
            self.qtable.update(self._state, self._action, self._reward, done=True)
        self._finished = True
        if self.training:
            self.qtable.episodes += 1
            self.qtable.decay_epsilon()
            self.qtable.saveWeights(self.table_path)
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            header = not self.metrics_path.exists()
            with self.metrics_path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                if header:
                    writer.writerow(("episode", "round", "outcome", "reward",
                                     "turns", "actions", "epsilon"))
                writer.writerow((self.qtable.episodes, event.round_number,
                                 self._outcome or "draw", self._total_reward,
                                 event.turn_number, self._steps, self.qtable.epsilon))
        print(f"Round {event.round_number}: {self._outcome or 'draw'}, "
              f"reward={self._total_reward:g}, epsilon={self.qtable.epsilon:.3f}")


def main():
    CapPeeler().start()


if __name__ == "__main__":
    main()
