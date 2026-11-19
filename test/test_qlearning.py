import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as Event
import unittest
from unittest.mock import Mock, PropertyMock, patch
from dataclasses import FrozenInstanceError
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from LUTInterface import LUTInterface, State, EpsilonSchedule
from BaseInterface import BaseInterface
from bots.CapPeeler.CapPeeler import CapPeeler

# CHATGPT

class LearningTests(unittest.TestCase):
    def test_checkpoint_retries_temporary_windows_file_lock(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'table.npz'
            table = LUTInterface()
            original_replace = Path.replace
            attempts = []

            def transient_lock(source, target):
                attempts.append(source)
                if len(attempts) < 3:
                    raise PermissionError('checkpoint temporarily open')
                return original_replace(source, target)

            with patch.object(Path, 'replace', transient_lock), patch('LUTInterface.time.sleep'):
                table.saveWeights(path)
            self.assertEqual(len(attempts), 3)
            restored = LUTInterface()
            restored.loadWeights(path)
            np.testing.assert_array_equal(restored.Q, table.Q)

    def test_checkpoint_persistent_lock_preserves_previous_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'table.npz'
            table = LUTInterface()
            table.saveWeights(path)
            previous = path.read_bytes()
            table.episodes = 1
            with patch.object(Path, 'replace', side_effect=PermissionError('locked')) as replace:
                with patch('LUTInterface.time.sleep'):
                    with self.assertRaises(PermissionError):
                        table.saveWeights(path)
            self.assertEqual(replace.call_count, 5)
            self.assertEqual(path.read_bytes(), previous)
            self.assertTrue(path.with_suffix('.npz.tmp').exists())

    def test_checkpoint_failure_is_logged(self):
        bot = self.bot()
        bot.diagnostics = Mock()
        with TemporaryDirectory() as directory:
            bot.table_path = Path(directory) / 'table.npz'
            bot.qtable.saveWeights = Mock(side_effect=PermissionError('locked'))
            bot.on_round_ended(Event(round_number=1, turn_number=1))
            with self.assertRaises(PermissionError):
                bot._finish_episode()
        self.assertIn('PermissionError', bot.diagnostics.record.call_args.args[0])

    def test_linear_epsilon_schedule_reaches_exact_zero(self):
        table = LUTInterface()
        table.configure_epsilon(start=.8, zero_at_battle=7000, mode=EpsilonSchedule.Decay.kLinear)
        for battle, expected in ((0, .8), (3500, .4), (6999, .8 / 7000),
                                 (7000, 0), (7001, 0)):
            table.episodes = battle
            table.decay_epsilon()
            self.assertAlmostEqual(table.epsilon, expected)

    def test_step_epsilon_schedule_and_immediate_cutoff(self):
        table = LUTInterface()
        table.configure_epsilon(start=.8, zero_at_battle=7000, mode=EpsilonSchedule.Decay.kStep)
        for battle, expected in ((0, .8), (3500, .8), (6999, .8), (7000, 0), (7001, 0)):
            table.episodes = battle
            table.decay_epsilon()
            self.assertEqual(table.epsilon, expected)
        table.episodes = 0
        table.configure_epsilon(start=.8, zero_at_battle=0)
        self.assertEqual(table.epsilon, 0)

    def test_invalid_epsilon_schedule_leaves_current_setting_unchanged(self):
        table = LUTInterface(epsilon=.3)
        for options in (dict(start=-.1, zero_at_battle=7000),
                        dict(start=.8, zero_at_battle=-1),
                        dict(start=.8, zero_at_battle=1.5),
                        dict(start=.8, zero_at_battle=7000, mode="unknown"),
                        dict(start=.8, zero_at_battle=7000, mode="linear"),
                        dict(start=.8, zero_at_battle=7000, mode="Step")):
            with self.assertRaises(ValueError):
                table.configure_epsilon(**options)
            self.assertEqual(table.epsilon, .3)
            self.assertIsNone(table.epsilon_schedule)

    def test_schedule_checkpoint_modes_and_legacy_strings(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'table.npz'
            for mode in EpsilonSchedule.Decay:
                with self.subTest(mode=mode):
                    table = LUTInterface()
                    table.episodes = 3500
                    table.configure_epsilon(.8, 7000, mode)
                    table.saveWeights(path)
                    with np.load(path, allow_pickle=False) as data:
                        self.assertEqual(data['epsilon_decay_mode'].item(), mode.value)
                    restored = LUTInterface()
                    restored.loadWeights(path)
                    self.assertIs(restored.epsilon_schedule.mode, mode)
                    restored.decay_epsilon()
                    self.assertEqual(restored.epsilon, table.epsilon)
                    # Checkpoints from before Decay stored lowercase strings.
                    np.savez(path, Q=table.Q, epsilon=table.epsilon,
                             episodes=3500, epsilon_start=.8,
                             epsilon_zero_at_battle=7000,
                             epsilon_decay_mode=mode.value.lower())
                    restored.loadWeights(path)
                    self.assertIs(restored.epsilon_schedule.mode, mode)
                    restored.decay_epsilon()
                    self.assertEqual(restored.epsilon, table.epsilon)

    def test_bot_converts_environment_decay_to_enum(self):
        with TemporaryDirectory() as directory:
            environment = {'CAPPEELER_QTABLE': str(Path(directory) / 'table.npz'),
                           'CAPPEELER_DIAGNOSTICS': '0'}
            for setting, expected in ((None, EpsilonSchedule.Decay.kStep),
                                      ('linear', EpsilonSchedule.Decay.kLinear),
                                      ('step', EpsilonSchedule.Decay.kStep)):
                with self.subTest(setting=setting):
                    options = dict(environment)
                    if setting is not None:
                        options['CAPPEELER_EPSILON_DECAY'] = setting
                    with patch('os.environ', options), \
                            patch.object(CapPeeler.__bases__[0], '__init__', return_value=None), \
                            patch.object(CapPeeler, 'set_event_priority'):
                        bot = CapPeeler(epsilon_zero_at_battle=7000,
                                        epsilon_decay=EpsilonSchedule.Decay.kStep)
                    self.assertIs(bot.qtable.epsilon_schedule.mode, expected)

    def test_bot_algorithm_and_reward_mode_enums(self):
        cases = (
            ({}, {}, LUTInterface.Algorithm.kQLearning, LUTInterface.RewardMode.kIntermediate),
            ({}, dict(algorithm=LUTInterface.Algorithm.kSarsa,
                      reward_mode=LUTInterface.RewardMode.kTerminal),
             LUTInterface.Algorithm.kSarsa, LUTInterface.RewardMode.kTerminal),
            ({'CAPPEELER_ALGORITHM': 'SARSA', 'CAPPEELER_REWARDS': 'TERMINAL'}, {},
             LUTInterface.Algorithm.kSarsa, LUTInterface.RewardMode.kTerminal),
            ({'CAPPEELER_ALGORITHM': 'qlearning', 'CAPPEELER_REWARDS': 'intermediate'},
             dict(algorithm=LUTInterface.Algorithm.kSarsa,
                  reward_mode=LUTInterface.RewardMode.kTerminal),
             LUTInterface.Algorithm.kQLearning, LUTInterface.RewardMode.kIntermediate),
        )
        with TemporaryDirectory() as directory:
            for overrides, arguments, algorithm, rewards in cases:
                with self.subTest(overrides=overrides, arguments=arguments):
                    environment = {'CAPPEELER_QTABLE': str(Path(directory) / 'table.npz'),
                                   'CAPPEELER_DIAGNOSTICS': '0', **overrides}
                    with patch('os.environ', environment), \
                            patch.object(CapPeeler.__bases__[0], '__init__', return_value=None), \
                            patch.object(CapPeeler, 'set_event_priority'):
                        bot = CapPeeler(**arguments)
                    self.assertIs(bot.algorithm, algorithm)
                    self.assertIs(bot.reward_mode, rewards)

    def test_bot_rejects_invalid_algorithm_and_reward_mode(self):
        cases = (({'CAPPEELER_ALGORITHM': 'unknown'}, {}),
                 ({'CAPPEELER_REWARDS': 'unknown'}, {}),
                 ({}, {'algorithm': 'sarsa'}),
                 ({}, {'reward_mode': 'terminal'}))
        for environment, arguments in cases:
            with self.subTest(environment=environment, arguments=arguments):
                with patch('os.environ', environment), \
                        patch.object(CapPeeler.__bases__[0], '__init__', return_value=None):
                    with self.assertRaises(ValueError):
                        CapPeeler(**arguments)

    def test_schedule_round_boundary_and_resume(self):
        bot = self.bot()
        bot.qtable.episodes = 6999
        bot.qtable.configure_epsilon(start=.8, zero_at_battle=7000, mode=EpsilonSchedule.Decay.kStep)
        self.assertEqual(bot.qtable.epsilon, .8)
        with TemporaryDirectory() as directory:
            bot.table_path = Path(directory) / 'table.npz'
            bot.metrics_path = Path(directory) / 'table.csv'
            bot.on_round_ended(Event(round_number=1, turn_number=1))
            bot.on_game_ended(Event())
            self.assertEqual(bot.qtable.epsilon, 0)
            restored = LUTInterface()
            restored.loadWeights(bot.table_path)
            self.assertEqual(restored.epsilon_schedule, bot.qtable.epsilon_schedule)
            self.assertEqual(restored.episodes, 7000)
            restored.episodes += 1
            restored.decay_epsilon()
            self.assertEqual(restored.epsilon, 0)

    def test_counts_track_updates_per_action_and_state(self):
        table = LUTInterface()
        state, nxt = State(100, 0, 20, 20), State(250, -90, 50, 50)
        table.choose_action(state)
        table.choose_action(state, explore=False)
        self.assertEqual(table.update_counts.sum(), 0)
        table.update(state, 0, 0, nxt)  # Zero TD error is still an update.
        table.update(state, 0, 10, nxt, next_action=1)  # SARSA.
        table.update(state, 4, 100, done=True)
        self.assertEqual(table.update_counts[state.to_tuple() + (0,)], 2)
        self.assertEqual(table.update_counts[state.to_tuple() + (4,)], 1)
        self.assertEqual(table.state_update_counts[state.to_tuple()], 3)
        self.assertEqual(table.state_update_counts[nxt.to_tuple()], 0)
        with self.assertRaises(ValueError):
            table.update(state, 0, 10)
        self.assertEqual(table.update_counts.sum(), 3)

    def test_counts_persist_and_resume(self):
        state = State(100, 0, 20, 20)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'table.npz'
            table = LUTInterface()
            table.update(state, 4, 100, done=True)
            table.saveWeights(path)
            restored = LUTInterface()
            restored.loadWeights(path)
            np.testing.assert_array_equal(table.update_counts, restored.update_counts)
            restored.update(state, 4, 100, done=True)
            self.assertEqual(restored.update_counts[state.to_tuple() + (4,)], 2)

    def test_legacy_table_loads_without_inventing_counts(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.npz'
            table = LUTInterface()
            values = np.ones_like(table.Q)
            np.savez(path, Q=values, epsilon=.2, episodes=100)
            table.update(State(100, 0, 20, 20), 0, 0, done=True)
            table.loadWeights(path)
            np.testing.assert_array_equal(table.Q, values)
            self.assertEqual(table.episodes, 100)
            self.assertEqual(table.update_counts.sum(), 0)

    def test_invalid_saved_counts_do_not_replace_training_state(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'table.npz'
            table = LUTInterface()
            table.update(State(100, 0, 20, 20), 0, 10, done=True)
            before = table.Q.copy()
            counts_before = table.update_counts.copy()
            for counts in (np.zeros((2, 2), dtype=int),
                           np.full(table.Q.shape, -1, dtype=int),
                           np.full(table.Q.shape, .5)):
                with self.subTest(shape=counts.shape, dtype=counts.dtype):
                    np.savez(path, Q=np.ones_like(table.Q), epsilon=.5,
                             update_counts=counts)
                    with self.assertRaises(ValueError):
                        table.loadWeights(path)
                    np.testing.assert_array_equal(table.Q, before)
                    np.testing.assert_array_equal(table.update_counts, counts_before)

    def test_sarsa_uses_selected_action_instead_of_maximum(self):
        table = LUTInterface(alpha=0.1, gamma=0.95)
        state, nxt = State(100, 0, 20, 20), State(250, -90, 50, 50)
        table.Q[nxt.to_tuple()] = [-4, 8, 1, 2, 0, 3]
        table.update(state, 2, 10, nxt, next_action=0)
        self.assertAlmostEqual(table.Q[state.to_tuple() + (2,)], .62)
        table.update(state, 2, -100, done=True)
        self.assertAlmostEqual(table.Q[state.to_tuple() + (2,)], -9.442)

    def test_sarsa_executes_action_used_for_bootstrap(self):
        bot = self.bot()
        bot.algorithm = LUTInterface.Algorithm.kSarsa
        bot._bot_internals = Event(radar_turn_rate=0)
        bot.qtable.choose_action = Mock(side_effect=[1, 4])
        bot.qtable.update = Mock()
        bot.on_tick(Event(turn_number=1))
        bot.on_tick(Event(turn_number=6))
        self.assertEqual(bot._action, 4)
        self.assertEqual(bot.qtable.update.call_args.kwargs['next_action'], 4)
        self.assertEqual(bot.qtable.choose_action.call_count, 2)

    def test_terminal_only_ignores_intermediate_events(self):
        bot = self.bot()
        bot.reward_mode = LUTInterface.RewardMode.kTerminal
        bot._state = State(100, 0, 20, 20)
        bot.on_bullet_hit(Event())
        bot.on_hit_by_bullet(Event())
        bot.on_hit_wall(Event())
        bot.on_hit_bot(Event())
        self.assertEqual(bot._reward, 0)
        bot.on_won_round(Event())
        self.assertEqual(bot._reward, 100)
        self.assertEqual(bot._total_reward, 100)

    def test_fixed_zero_exploration_survives_round_end(self):
        bot = self.bot()
        bot.fixed_epsilon = 0.0
        with TemporaryDirectory() as directory:
            bot.table_path = Path(directory) / 'table.npz'
            bot.metrics_path = Path(directory) / 'table.csv'
            bot.on_round_ended(Event(round_number=1, turn_number=1))
            bot.on_game_ended(Event())
            self.assertEqual(bot.qtable.epsilon, 0.0)

    def test_lut_implements_base_interface(self):
        self.assertIsInstance(LUTInterface(), BaseInterface)
        with self.assertRaises(TypeError):
            BaseInterface()
    def test_bellman_update_uses_maximum_next_action(self):
        table = LUTInterface(alpha=0.1, gamma=0.95)
        s, nxt = State(100, 0, 20, 20), State(250, -90, 50, 50)
        table.Q[s.to_tuple() + (2,)] = 2
        table.Q[nxt.to_tuple()] = [-4, 8, 1, 2, 0, 3]
        error = table.update(s, 2, 10, nxt)
        self.assertAlmostEqual(error, 15.6)
        self.assertAlmostEqual(table.Q[s.to_tuple() + (2,)], 3.56)
        table.update(s, 2, -100, done=True)
        self.assertAlmostEqual(table.Q[s.to_tuple() + (2,)], -6.796)

    def test_policy_and_decay(self):
        table = LUTInterface(epsilon=1, seed=42)
        state = State(100, 0, 20, 20)
        table.Q[state.to_tuple() + (3,)] = 10
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
        self.assertEqual(State(149, 0, 29, 69).to_tuple(), (0, 0, 0, 1))
        self.assertEqual(State(150, -90, 30, 70).to_tuple(), (1, 1, 1, 2))
        self.assertEqual(State(400, 90, 70, 100).to_tuple(), (2, 3, 2, 2))
        self.assertEqual(State(400, 180, 70, 100).to_tuple(), (2, 2, 2, 2))
        for bearing, expected in ((-135, 1), (-45, 0), (45, 3), (135, 2), (270, 1)):
            self.assertEqual(State(150.5, bearing, 30.5, 70.5).to_tuple(),
                             (1, expected, 1, 2))

    def test_state_preserves_raw_observation_snapshot(self):
        state = State(150.5, -90.5, 30.5, 70.5)
        self.assertIsInstance(state, State)
        self.assertEqual(state.distance, 150.5)
        self.assertEqual(state.bearing, -90.5)
        self.assertEqual(state.own_energy, 30.5)
        self.assertEqual(state.enemy_energy, 70.5)
        self.assertEqual(state.to_tuple(), (1, 1, 1, 2))
        self.assertEqual(state.to_tuple(), (1, 1, 1, 2))
        with self.assertRaises(FrozenInstanceError):
            state.distance = 0

    def test_cappeeler_observation_creates_state(self):
        bot = self.bot()
        bot.distance_to = Mock(return_value=250.5)
        bot.bearing_to = Mock(return_value=-90.5)
        with patch.object(CapPeeler, "energy", new_callable=PropertyMock, return_value=75):
            state = CapPeeler._observation(bot)
        self.assertEqual(state, State(250.5, -90.5, 75, 100))
        self.assertEqual(state.to_tuple(), (1, 1, 2, 2))

    def bot(self):
        # No network or game session: exercise actual learning callbacks.
        bot = object.__new__(CapPeeler)
        bot.qtable = LUTInterface(epsilon=0, seed=42)
        bot.training = True
        bot._reset_episode()
        bot._observation = Mock(return_value=State(100, 0, 20, 20))
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
            bot.on_round_started(Event(round_number=2))
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
        self.assertEqual(bot.qtable.update_counts.sum(), 0)
        self.assertEqual(bot.qtable.epsilon, 1)
        self.assertEqual(bot.qtable.episodes, 0)


if __name__ == '__main__':
    unittest.main()
