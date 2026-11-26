"""Compatibility entry point; plotting lives in test/plotter/compare.py."""
## CHATGPT compare learning between SARSA and Q-Learning
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from test.plotter.compare import main

if __name__ == '__main__':
    main()
