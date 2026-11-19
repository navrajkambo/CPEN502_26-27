## CHATGPT (diagnostics for debugging freezing issue with robocode)
import faulthandler
from pathlib import Path
import threading
import time


class BotDiagnostics:
    def __init__(self, filename, timeout=20):
        self.path = Path(filename)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._last_tick = time.monotonic()
        self._phase = "waiting for game"
        self._watching = False
        self._reported_tick = None
        self._stop = threading.Event()
        self.record("Bot started")
        self._thread = threading.Thread(target=self._watch, daemon=True,
                                        name="CapPeeler diagnostics")
        self._thread.start()

    def heartbeat(self, phase):
        # No file I/O on the per-tick path.
        self._phase = phase
        self._last_tick = time.monotonic()
        self._watching = True

    def idle(self, phase):
        self._watching = False
        self._phase = phase
        self.record(phase)

    def record(self, message):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")

    def _check_stall(self, now):
        last_tick = self._last_tick
        if (not self._watching or now - last_tick < self.timeout
                or self._reported_tick == last_tick):
            return
        self._reported_tick = last_tick
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(f"No tick progress for {now - last_tick:.1f}s; "
                         f"last activity: {self._phase}\n")
            stream.flush()
            faulthandler.dump_traceback(file=stream, all_threads=True)

    def _watch(self):
        while not self._stop.wait(5):
            try:
                self._check_stall(time.monotonic())
            except OSError:
                # Diagnostics must never interfere with the battle.
                pass

    def close(self):
        self._stop.set()
