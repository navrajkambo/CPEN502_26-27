"""Compatibility entry point; plotting lives in test/plotter/training.py."""
# CHATGPT
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from test.plotter.training import main

if __name__ == '__main__':
    main()
