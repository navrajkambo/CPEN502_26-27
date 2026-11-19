import csv
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from LUTInterface import LUTInterface
from bots.CapPeeler.CapPeeler import CapPeeler


class LearningPlotTests(unittest.TestCase):
    def test_rolling_performance_and_saved_image(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("episode", "outcome", "reward", "epsilon"))
                writer.writerows(((1, "loss", -10, 1), (2, "win", 30, .8),
                                  (3, "loss", -20, .6)))
            fig, axes = LUTInterface().plotLearningPerformance(path, window=2)
            try:
                np.testing.assert_allclose(axes[0].lines[1].get_ydata(), [-10, 10, 5])
                np.testing.assert_allclose(axes[1].lines[0].get_ydata(), [0, 50, 50])
                np.testing.assert_allclose(axes[2].lines[0].get_ydata(), [1, .8, .6])
                np.testing.assert_array_equal(axes[2].lines[0].get_xdata(), [1, 2, 3])
                self.assertGreater(path.with_suffix(".png").stat().st_size, 0)
            finally:
                plt.close(fig)

    def test_empty_history_and_invalid_window(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            path.write_text("episode,outcome,reward,epsilon\n")
            with self.assertRaisesRegex(ValueError, "No completed rounds"):
                LUTInterface().plotLearningPerformance(path)
            with self.assertRaisesRegex(ValueError, "positive integer"):
                LUTInterface().plotLearningPerformance(path, window=0)

    def test_game_end_finalizes_then_launches_one_plot(self):
        with TemporaryDirectory() as directory:
            bot = object.__new__(CapPeeler)
            bot.training = True
            bot.metrics_path = Path(directory) / "history.csv"
            bot.qtable = LUTInterface()
            bot.qtable.episodes = 3
            # Represents the final round writer; plotting must run after this.
            bot._finish_episode = Mock(side_effect=lambda: bot.metrics_path.touch())
            with patch("bots.CapPeeler.CapPeeler.subprocess.Popen") as launch:
                bot.on_game_ended(None)
                bot.on_game_ended(None)
                launch.assert_called_once()
                self.assertIn(str(bot.metrics_path.resolve()), launch.call_args.args[0])
                self.assertIn("--show", launch.call_args.args[0])
                self.assertTrue(launch.call_args.kwargs["close_fds"])
                self.assertEqual(launch.call_args.kwargs["stdin"], -3)
                self.assertEqual(launch.call_args.kwargs["stderr"], -2)

    def test_plot_launch_failure_does_not_escape_game_end(self):
        with TemporaryDirectory() as directory:
            bot = object.__new__(CapPeeler)
            bot.training = True
            bot.metrics_path = Path(directory) / "history.csv"
            bot.metrics_path.touch()
            bot.qtable = LUTInterface()
            bot._finish_episode = Mock()
            bot.diagnostics = Mock()
            with patch("bots.CapPeeler.CapPeeler.subprocess.Popen", side_effect=OSError("failure")):
                bot.on_game_ended(Mock(number_of_rounds=100))
            bot.diagnostics.record.assert_called_once()


if __name__ == "__main__":
    unittest.main()
