import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as Event
import unittest
from unittest.mock import Mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from LUTInterface import LUTInterface
from BaseInterface import BaseInterface
from CapPeelerRL import encode_state
from bots.CapPeeler.CapPeeler import CapPeeler


class LearningTests(unittest.TestCase):
    def test_lut_implements_base_interface(self):
        self.assertIsInstance(LUTInterface(), BaseInterface)
        with self.assertRaises(TypeError):
            BaseInterface()
    def test_bellman_update_uses_maximum_next_action(self):
        table = LUTInterface(alpha=0.1, gamma=0.95)
        s, nxt = (0, 0, 0, 0), (1, 1, 1, 1)
        table.Q[s + (2,)] = 2
        table.Q[nxt] = [-4, 8, 1, 2, 0, 3]
        error = table.update(s, 2, 10, nxt)
        self.assertAlmostEqual(error, 15.6)
        self.assertAlmostEqual(table.Q[s + (2,)], 3.56)
        table.update(s, 2, -100, done=True)
        self.assertAlmostEqual(table.Q[s + (2,)], -6.796)

    def test_policy_and_decay(self):
        table = LUTInterface(epsilon=1, seed=42)
        state = (0, 0, 0, 0)
        table.Q[state + (3,)] = 10
        self.assertEqual(table.choose_action(state, explore=False), 3)
        self.assertGreater(len({table.choose_action(state) for _ in range(100)}), 1)
        table.epsilon = 0.05
        table.decay_epsilon()
        self.assertEqual(table.epsilon, 0.05)
        table.Q[:] = 0
        self.assertGreater(len({table.choose_action(state, explore=False)
                                for _ in range(100)}), 1)

    def test_persistence(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'table.npz'
            table = LUTInterface(epsilon=0.3)
            table.Q[1, 2, 1, 2, 4] = 13
            table.episodes = 17
            table.saveWeights(path)
            restored = LUTInterface()
            restored.loadWeights(path)
            np.testing.assert_array_equal(table.Q, restored.Q)
            self.assertEqual(restored.episodes, 17)
            self.assertEqual(restored.epsilon, 0.3)
            np.savez(path, Q=np.zeros((2, 2)), epsilon=0.2)
            with self.assertRaises(ValueError):
                restored.loadWeights(path)
            self.assertEqual(restored.Q[1, 2, 1, 2, 4], 13)

    def test_encoder_boundaries(self):
        self.assertEqual(encode_state(149, 0, 29, 69), (0, 0, 0, 1))
        self.assertEqual(encode_state(150, -90, 30, 70), (1, 1, 1, 2))
        self.assertEqual(encode_state(400, 90, 70, 100), (2, 3, 2, 2))
        self.assertEqual(encode_state(400, 180, 70, 100), (2, 2, 2, 2))

    def bot(self):
        # No network or game session: exercise actual learning callbacks.
        bot = object.__new__(CapPeeler)
        bot.qtable = LUTInterface(epsilon=0, seed=42)
        bot.training = True
        bot._reset_episode()
        bot._observation = Mock(return_value=(0, 0, 0, 0))
        bot._execute_action = Mock()
        bot._plot_performance = Mock()
        bot._target = (1, 100, 100, 100, 1)
        return bot

    def test_action_duration_and_terminal_reward_once(self):
        bot = self.bot()
        # Avoid network-backed radar property in this callback-only harness.
        bot._bot_internals = Event(radar_turn_rate=0)
        bot.on_tick(Event(turn_number=1))
        bot.on_bullet_hit(Event())
        bot.on_tick(Event(turn_number=5))
        self.assertEqual(bot._steps, 1)
        bot.on_tick(Event(turn_number=6))
        self.assertEqual(bot._steps, 2)
        self.assertEqual(bot.qtable.Q.sum(), 1)
        bot.on_death(Event())
        bot.on_death(Event())
        bot.on_hit_by_bullet(Event())
        with TemporaryDirectory() as directory:
            bot.table_path = Path(directory) / 'table.npz'
            bot.metrics_path = Path(directory) / 'table.csv'
            bot.on_round_ended(Event(round_number=1, turn_number=7))
            bot.on_game_ended(Event())
            values = bot.qtable.Q.copy()
            bot.on_round_ended(Event(round_number=1, turn_number=7))
            bot.on_game_ended(Event())
            np.testing.assert_array_equal(values, bot.qtable.Q)
            self.assertEqual(bot.qtable.episodes, 1)
            self.assertEqual(bot._total_reward, -100)
            self.assertEqual(len(bot.metrics_path.read_text().splitlines()), 2)

    def test_final_events_after_round_end_are_saved_before_reset(self):
        bot = self.bot()
        bot._bot_internals = Event(radar_turn_rate=0)
        bot.on_tick(Event(turn_number=1))
        with TemporaryDirectory() as directory:
            bot.table_path = Path(directory) / 'table.npz'
            bot.metrics_path = Path(directory) / 'table.csv'
            bot.on_round_ended(Event(round_number=1, turn_number=2))
            bot.on_won_round(Event())
            bot.on_bullet_hit(Event())
            self.assertFalse(bot.table_path.exists())
            bot.on_round_started(Event())
            self.assertAlmostEqual(bot.qtable.Q.sum(), 11)
            self.assertEqual(bot.qtable.episodes, 1)
            self.assertIsNone(bot._state)
            self.assertIsNone(bot._target)
            self.assertIn('win,110.0', bot.metrics_path.read_text())

    def test_evaluation_leaves_training_state_unchanged(self):
        bot = self.bot()
        bot.training = False
        bot._bot_internals = Event(radar_turn_rate=0)
        bot.qtable.epsilon = 1
        bot.qtable.Q[0, 0, 0, 0, 4] = 5
        before = bot.qtable.Q.copy()
        bot.on_tick(Event(turn_number=1))
        self.assertEqual(bot._action, 4)
        bot.on_tick(Event(turn_number=6))
        bot.on_round_ended(Event(round_number=1, turn_number=8))
        bot.on_won_round(Event())
        bot.on_game_ended(Event())
        np.testing.assert_array_equal(before, bot.qtable.Q)
        self.assertEqual(bot.qtable.epsilon, 1)
        self.assertEqual(bot.qtable.episodes, 0)


if __name__ == '__main__':
    unittest.main()
