"""Plotting implementations for the Part 2 report and saved battle histories."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
# Learning code uses absolute imports such as `from BaseInterface import ...`.
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
