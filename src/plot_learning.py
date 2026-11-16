"""View saved training results without starting a battle."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from LUTInterface import LUTInterface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "training" / "cappeeler.csv")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--show", action="store_true", help="Open the plot window")
    args = parser.parse_args()
    fig, _ = LUTInterface().plotLearningPerformance(
        args.filename, window=args.window, show=args.show)
    plt.close(fig)


if __name__ == "__main__":
    main()
