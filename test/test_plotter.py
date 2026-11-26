"""Check shared report statistics and schedule alignment after moving plotters."""
import csv
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test.plotter.common import load_run, greedy_blocks, rolling_mean
from LUTInterface import EpsilonSchedule


class ReportPlotterTests(unittest.TestCase):
    def test_rolling_windows_preserve_partial_and_full_statistics(self):
        np.testing.assert_allclose(rolling_mean([-10, 30, -20], 2), [-10, 10, 5])
        np.testing.assert_allclose(rolling_mean([-10, 30, -20], 2, True), [10, 5])
        self.assertEqual(len(rolling_mean([1], 2, True)), 0)
        with self.assertRaises(ValueError):
            rolling_mean([1], 0)

    def test_probe_selection_uses_epsilon_during_round_not_csv_next_value(self):
        schedule = EpsilonSchedule(.95, 7000, EpsilonSchedule.Decay.kLinearPerf)
        with TemporaryDirectory() as directory:
            folder = Path(directory) / 'probe'
            folder.mkdir()
            with (folder / 'cappeeler.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['episode', 'outcome', 'reward', 'epsilon'])
                for episode in range(1, 212):
                    greedy = 201 <= episode <= 210
                    writer.writerow([episode, 'win' if greedy else 'loss',
                                     100 if greedy else -100, schedule.value(episode)])
            np.savez(folder / 'cappeeler.npz', episodes=211, epsilon_start=.95,
                     epsilon_zero_at_battle=7000, epsilon_decay_mode='LinearPerf',
                     epsilon_perf_interval=200, epsilon_perf_rounds=10)
            run = load_run('probe', directory)
            np.testing.assert_array_equal(run['episodes'][run['probe']], np.arange(201, 211))
            self.assertEqual(run['recorded_epsilon'][199], 0)
            self.assertGreater(run['epsilon'][199], 0)
            self.assertEqual(run['epsilon'][209], 0)
            self.assertGreater(run['recorded_epsilon'][209], 0)
            np.testing.assert_allclose(greedy_blocks(run), [[205.5, 100, 100, 10]])


if __name__ == '__main__':
    unittest.main()
