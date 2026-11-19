from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from BotDiagnostics import BotDiagnostics


class BotDiagnosticsTests(unittest.TestCase):
    def test_stall_records_once_until_progress_resumes(self):
        with TemporaryDirectory() as directory:
            diagnostics = BotDiagnostics(Path(directory) / "bot.log")
            try:
                with patch("BotDiagnostics.time.monotonic", return_value=100):
                    diagnostics.heartbeat("round 100 turn 50")
                with patch("BotDiagnostics.faulthandler.dump_traceback") as dump:
                    diagnostics._check_stall(110)
                    dump.assert_not_called()
                    diagnostics._check_stall(121)
                    diagnostics._check_stall(130)
                    dump.assert_called_once()
                    diagnostics.idle("game ended")
                    diagnostics._check_stall(200)
                    dump.assert_called_once()
                    with patch("BotDiagnostics.time.monotonic", return_value=201):
                        diagnostics.heartbeat("next game turn 1")
                    diagnostics._check_stall(222)
                    self.assertEqual(dump.call_count, 2)
                self.assertIn("round 100 turn 50", diagnostics.path.read_text())
            finally:
                diagnostics.close()
