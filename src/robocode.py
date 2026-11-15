import subprocess
import platform
import os
import signal


class RoboCode:
    def __init__(self, path="robocode", version="1.4.0"):
        self.path = path
        self.version = version

        self.server = None
        self.gui = None

    def _start(self, jar, name):
        jar_path = os.path.abspath(os.path.join(self.path, jar))

        if not os.path.isfile(jar_path):
            raise FileNotFoundError(f"Cannot find {jar_path}")

        if platform.system() == "Windows":
            # Open a separate, visible console window.
            proc = subprocess.Popen(
                ["java", "-jar", jar_path],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            # Launch Java with its normal GUI window (if applicable).
            proc = subprocess.Popen(
                ["java", "-jar", jar_path],
                start_new_session=True
            )

        print(f"{name} started (PID: {proc.pid})")
        return proc

    def _stop(self, proc, name):
        if proc is None:
            return None

        # May already have been closed manually.
        if proc.poll() is not None:
            print(f"{name} already stopped.")
            return None

        if platform.system() == "Windows":
            # Terminate the process.
            proc.terminate()
        else:
            # Request graceful shutdown.
            os.kill(proc.pid, signal.SIGTERM)

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        print(f"{name} stopped.")
        return None

    def start_server(self):
        if self.server is None or self.server.poll() is not None:
            self.server = self._start(
                f"robocode-tankroyale-server-{self.version}.jar",
                "RoboCode Server"
            )

    def start_gui(self):
        if self.gui is None or self.gui.poll() is not None:
            self.gui = self._start(
                f"robocode-tankroyale-gui-{self.version}.jar",
                "RoboCode GUI"
            )

    def stop_server(self):
        self.server = self._stop(self.server, "RoboCode Server")

    def stop_gui(self):
        self.gui = self._stop(self.gui, "RoboCode GUI")

    def start(self):
        self.start_server()
        self.start_gui()

    def stop(self):
        self.stop_gui()
        self.stop_server()