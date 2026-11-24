"""Reproduce assignment comparison figures from the five completed battle runs."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'doc' / 'Part 2' / 'figures'
WINDOW = 200
CUTOFF = 7000
RUNS = {'Q-learning': 'qlearning', 'SARSA': 'sarsa',
        'epsilon = 0.25': '0.25 epsilon', 'epsilon = 0.50': '0.5 epsilon',
        'epsilon = 0.75': '0.75 epsilon'}


def load(folder):
    path = ROOT / 'training' / folder / 'cappeeler.csv'
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    episodes = np.array([int(row['episode']) for row in rows])
    if not np.array_equal(episodes, np.arange(1, 10001)):
        raise ValueError(f'Expected 10,000 consecutive episodes: {path}')
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as data:
        start = float(data['epsilon_start'])
        cutoff = int(data['epsilon_zero_at_battle'])
        mode = str(data['epsilon_decay_mode'].item())
        if cutoff != CUTOFF or int(data['episodes']) != len(rows):
            raise ValueError(f'CSV/checkpoint metadata mismatch: {path}')
    recorded = np.array([float(row['epsilon']) for row in rows])
    expected = (np.where(episodes < cutoff, start, 0.) if mode == 'Step'
                else np.maximum(0., start * (1 - episodes / cutoff)))
    np.testing.assert_allclose(recorded, expected, atol=1e-12)
    used = (np.where(episodes <= cutoff, start, 0.) if mode == 'Step'
            else np.maximum(0., start * (1 - (episodes - 1) / cutoff)))
    return dict(episodes=episodes, wins=np.array([row['outcome'] == 'win' for row in rows]),
                rewards=np.array([float(row['reward']) for row in rows]),
                turns=np.array([int(row['turns']) for row in rows]),
                epsilon=used, start=start, mode=mode, source=path.relative_to(ROOT).as_posix())


def mean_curve(values):
    # Full windows only, so the first plotted point summarizes 200 episodes.
    totals = np.r_[0., np.cumsum(values)]
    return (totals[WINDOW:] - totals[:-WINDOW]) / WINDOW


def curves(histories, labels, filename, title):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={'height_ratios': [2, 2, 1]},
                             constrained_layout=True)
    fig.suptitle(title, fontsize=15)
    for label in labels:
        run = histories[label]
        x = run['episodes'][WINDOW - 1:]
        line, = axes[0].plot(x, 100 * mean_curve(run['wins']), label=label)
        color = line.get_color()
        axes[1].plot(x, mean_curve(run['rewards']), color=color)
        axes[2].plot(run['episodes'], run['epsilon'], color=color, drawstyle='steps-post')
    for axis in axes:
        axis.axvspan(CUTOFF + .5, 10000, color='#e6f1e7', alpha=.65)
        axis.axvline(CUTOFF + .5, color='#555555', linestyle='--', linewidth=1)
        axis.grid(alpha=.2)
        axis.set_xlim(1, 10000)
    axes[0].set(ylabel=f'Win rate (%)\nLast {WINDOW} rounds', ylim=(0, 103))
    axes[0].legend(loc='upper left')
    axes[0].text(8500, 5, 'Greedy actions; learning continues', ha='center', fontsize=9)
    axes[1].set_ylabel(f'Mean reward\nLast {WINDOW} rounds')
    axes[2].set(ylabel='Epsilon used', ylim=(-.02, 1.02), xlabel='Completed training rounds')
    fig.savefig(OUTPUT / filename, dpi=160)
    plt.close(fig)


def metrics(run, selection):
    return {'rounds': len(run['wins'][selection]),
            'win_rate_percent': float(100 * np.mean(run['wins'][selection])),
            'mean_reward': float(np.mean(run['rewards'][selection])),
            'mean_turns': float(np.mean(run['turns'][selection]))}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    histories = {label: load(folder) for label, folder in RUNS.items()}
    policy = ['Q-learning', 'SARSA']
    exploration = list(RUNS)[2:]
    curves(histories, policy, 'policy_comparison.png',
           'Q-learning vs SARSA: linear epsilon 0.95 to 0 at 7,000 rounds')
    curves(histories, exploration, 'epsilon_comparison.png',
           'Exploration comparison: kStep epsilon, cutoff at 7,000 rounds')
    phases = {'first_500': slice(0, 500), 'exploration_1_7000': slice(0, CUTOFF),
              'greedy_7001_10000': slice(CUTOFF, None), 'final_500': slice(-500, None),
              'preceding_500': slice(-1000, -500)}
    summary = {label: {'source': run['source'], 'epsilon_start': run['start'],
                       'epsilon_schedule': run['mode'],
                       **{name: metrics(run, selection) for name, selection in phases.items()}}
               for label, run in histories.items()}
    (OUTPUT / 'measured_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    x = [histories[label]['start'] for label in exploration]
    for phase, label, marker, offset in [
            ('exploration_1_7000', 'Rounds 1-7,000 (exploration enabled)', 'o', (0, 10)),
            ('greedy_7001_10000', 'Rounds 7,001-10,000 (epsilon = 0)', 's', (-18, -20)),
            ('final_500', 'Final 500 rounds (epsilon = 0)', '^', (18, 12))]:
        y = [summary[name][phase]['win_rate_percent'] for name in exploration]
        axis.plot(x, y, marker=marker, label=label)
        for a, b in zip(x, y):
            axis.annotate(f'{b:.1f}%', (a, b), xytext=offset, textcoords='offset points', ha='center')
    axis.set(title='Measured win rate vs initial kStep epsilon',
             xlabel='Epsilon during rounds 1-7,000', ylabel='Win rate (%)',
             ylim=(0, 108), xticks=x)
    axis.grid(alpha=.2)
    axis.legend(loc='center right')
    fig.savefig(OUTPUT / 'epsilon_summary.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
