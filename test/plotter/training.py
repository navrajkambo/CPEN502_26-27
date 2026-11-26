"""Single-run plots, including the plot automatically opened by CapPeeler."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from . import ROOT
from .common import load_history, rolling_mean


def plot_learning_performance(filename, window=50, output_path=None, show=False):
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError('window must be a positive integer')
    path = Path(filename)
    run = load_history(path)
    episodes, rewards, wins = run['episodes'], run['rewards'], run['wins']
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    fig.suptitle('CapPeeler training performance (includes exploration)')
    axes[0].plot(episodes, rewards, color='lightsteelblue', alpha=.6, label='Round reward')
    axes[0].plot(episodes, rolling_mean(rewards, window), color='tab:blue',
                 label=f'Average over last {window} rounds')
    axes[0].set_ylabel('Reward')
    axes[0].legend(loc='best')
    axes[1].plot(episodes, 100 * rolling_mean(wins, window), color='tab:red')
    axes[1].set_ylabel(f'Win rate (%)\nLast {window} rounds')
    axes[1].set_ylim(0, 100)
    axes[2].plot(episodes, run['recorded_epsilon'], color='tab:green')
    axes[2].set_ylabel('Epsilon after round')
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xlabel('Completed training rounds (battles / episodes)')
    for axis in axes:
        axis.grid(True, alpha=.25)
    destination = Path(output_path) if output_path else path.with_suffix('.png')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    if show:
        plt.show()
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename', nargs='?', type=Path, default=ROOT / 'training/cappeeler.csv')
    parser.add_argument('--window', type=int, default=50)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    fig, _ = plot_learning_performance(args.filename, args.window, args.output, args.show)
    plt.close(fig)


if __name__ == '__main__':
    main()
