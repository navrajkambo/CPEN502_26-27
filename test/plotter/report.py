"""Reproduce assignment comparison figures from the completed battle runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


from . import ROOT
from .common import load_run, rolling_mean
from .training import plot_learning_performance
from .initialization import plot_zero_initializations
from .greedy import plot_periodic_greedy, plot_greedy_windows_only
OUTPUT = ROOT / 'doc' / 'Part 2' / 'figures'
WINDOW = 200
CUTOFF = 7000
RUNS = {'Q-learning': 'qlearning', 'SARSA': 'sarsa',
        'epsilon = 0.25': '0.25 epsilon', 'epsilon = 0.50': '0.5 epsilon',
        'epsilon = 0.75': '0.75 epsilon', 'epsilon = 0': '0 epsilon (random init)'}


def load(folder):
    run = load_run(folder)
    if len(run['episodes']) != 10000 or run['cutoff'] != CUTOFF:
        raise ValueError(f'Expected 10,000 rounds and a {CUTOFF}-round cutoff: {folder}')
    return run


def mean_curve(values):
    return rolling_mean(values, WINDOW, full_windows=True)


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
    axes[1].set_ylabel(f'Mean reward\nLast {WINDOW} rounds')
    axes[2].set(ylabel='Epsilon used', ylim=(-.02, 1.02), xlabel='Completed training rounds')
    fig.savefig(OUTPUT / filename, dpi=160)
    plt.close(fig)


def metrics(run, selection):
    return {'rounds': len(run['wins'][selection]),
            'win_rate_percent': float(100 * np.mean(run['wins'][selection])),
            'mean_reward': float(np.mean(run['rewards'][selection])),
            'mean_turns': float(np.mean(run['turns'][selection]))}


def reward_mode_comparison(intermediate, terminal):
    """Plot matched epsilon schedules; reward values are intentionally omitted."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]},
                             constrained_layout=True)
    for label, run in [('Intermediate rewards (Q-learning)', intermediate),
                       ('Terminal-only rewards', terminal)]:
        x = run['episodes'][WINDOW - 1:]
        axes[0].plot(x, 100 * mean_curve(run['wins']), label=label)
        axes[1].plot(run['episodes'], run['epsilon'], drawstyle='steps-post', label=label)
    axes[0].axvline(CUTOFF + .5, color='#555555', linestyle='--', linewidth=1)
    axes[0].axvspan(CUTOFF + .5, 10000, color='#e6f1e7', alpha=.65)
    axes[0].set(ylabel=f'Win rate (%)\nLast {WINDOW} rounds', ylim=(0, 103),
                title='Intermediate vs terminal-only rewards')
    axes[0].legend(loc='upper left')
    axes[0].grid(alpha=.2)
    axes[1].set(ylabel='Epsilon used', ylim=(-.02, 1.02), xlabel='Completed training rounds')
    axes[1].grid(alpha=.2)
    axes[1].set_xlim(1, 10000)
    fig.savefig(OUTPUT / 'reward_mode_comparison.png', dpi=160)
    plt.close(fig)


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT,
                        help='Destination for all ten figures and measured_summary.json')
    args = parser.parse_args()
    OUTPUT = args.output
    OUTPUT.mkdir(parents=True, exist_ok=True)
    histories = {label: load(folder) for label, folder in RUNS.items()}
    terminal = load('terminal')
    policy = ['Q-learning', 'SARSA']
    exploration = list(RUNS)[2:]
    curves(histories, policy, 'policy_comparison.png',
           'Q-learning vs SARSA: linear epsilon 0.95 to 0 at 7,000 rounds')
    curves(histories, exploration, 'epsilon_comparison.png',
           'Exploration comparison: kStep epsilon, cutoff at 7,000 rounds')
    reward_mode_comparison(histories['Q-learning'], terminal)
    phases = {'first_500': slice(0, 500), 'exploration_1_7000': slice(0, CUTOFF),
              'greedy_7001_10000': slice(CUTOFF, None), 'final_500': slice(-500, None),
              'preceding_500': slice(-1000, -500)}
    summary = {label: {'source': run['source'], 'epsilon_start': run['start'],
                       'epsilon_schedule': run['mode'],
                       **{name: metrics(run, selection) for name, selection in phases.items()}}
               for label, run in histories.items()}
    summary['Terminal-only Q-learning'] = {
        'source': terminal['source'], 'epsilon_start': terminal['start'],
        'epsilon_schedule': terminal['mode'],
        **{name: metrics(terminal, selection) for name, selection in phases.items()}}
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
    zero = load_run('0 epsilon')
    random = histories['epsilon = 0']
    performance = load_run('qlearning (online performance measurment)')
    plot_zero_initializations(zero, random, OUTPUT / 'epsilon_zero_initialization_comparison.png')
    plot_periodic_greedy(histories['Q-learning'], performance,
                         OUTPUT / 'qlearning_periodic_greedy_performance.png')
    plot_greedy_windows_only(performance, OUTPUT / 'qlearning_greedy_windows_only.png')
    for folder, filename in [
            ('qlearning', 'qlearning_training.png'),
            ('0 epsilon', 'zero_epsilon_zero_init.png'),
            ('0 epsilon (random init)', 'zero_epsilon_random_init.png')]:
        fig, _ = plot_learning_performance(ROOT / 'training' / folder / 'cappeeler.csv',
                                           output_path=OUTPUT / filename)
        plt.close(fig)
    print(f'Saved ten report figures and measured_summary.json to {OUTPUT.resolve()}')


if __name__ == '__main__':
    main()
