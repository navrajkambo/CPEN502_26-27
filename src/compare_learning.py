"""Compare measured battle histories, e.g. --run Q-learning=q.csv --run SARSA=s.csv."""
## CHATGPT compare learning between SARSA and Q-Learning
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rolling_mean(values, window):
    totals = np.r_[0., np.cumsum(values)]
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - window)
    return (totals[ends] - totals[starts]) / (ends - starts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, metavar='LABEL=CSV')
    parser.add_argument('--window', type=int, default=50)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epsilon-summary', type=Path,
                        help='Also plot final-window win rate vs fixed epsilon')
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window must be positive')
    histories = []
    for specification in args.run:
        label, filename = specification.split('=', 1)
        with Path(filename).open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f'No completed rounds: {filename}')
        episodes = np.array([int(row['episode']) for row in rows])
        if episodes[0] != 1 or np.any(np.diff(episodes) != 1):
            raise ValueError(f'Use a continuous history starting at episode 1: {filename}')
        histories.append((label, episodes,
                          np.array([row['outcome'] == 'win' for row in rows], dtype=float),
                          np.array([float(row['reward']) for row in rows]),
                          np.array([float(row['epsilon']) for row in rows])))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    fig.suptitle('Measured training performance (includes exploration)')
    for label, episodes, wins, rewards, epsilon in histories:
        axes[0].plot(episodes, 100 * rolling_mean(wins, args.window), label=label)
        axes[1].plot(episodes, rolling_mean(rewards, args.window), label=label)
    axes[0].set_ylabel(f'Win rate (%) / last {args.window} rounds')
    axes[0].set_ylim(0, 100)
    axes[1].set_ylabel(f'Mean reward / last {args.window} rounds')
    axes[1].set_xlabel('Completed training rounds')
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    if args.epsilon_summary:
        lengths = {len(item[1]) for item in histories}
        if len(lengths) != 1 or next(iter(lengths)) < args.window:
            raise ValueError('Epsilon summary requires equal training budgets >= window')
        points = []
        for label, episodes, wins, rewards, epsilon in histories:
            if not np.allclose(epsilon, epsilon[0]):
                raise ValueError(f'Epsilon must be fixed throughout each run: {label}')
            points.append((epsilon[0], 100 * wins[-args.window:].mean()))
        points.sort()
        fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        axis.plot(*zip(*points), marker='o')
        axis.set(xlabel='Fixed exploration probability epsilon',
                 ylabel=f'Win rate (%) / final {args.window} training rounds', ylim=(0, 100))
        axis.grid(alpha=.25)
        args.epsilon_summary.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.epsilon_summary, dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    main()
