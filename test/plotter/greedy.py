"""Plots for periodic greedy measurements during kLinearPerf training."""
from pathlib import Path
import matplotlib.pyplot as plt
from .common import greedy_blocks, rolling_mean


def plot_periodic_greedy(original, performance, output):
    blocks = greedy_blocks(performance)
    if not len(blocks):
        raise ValueError('No periodic greedy windows to plot')
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout='constrained')
    for run, label, color in [
            (original, 'Original linear: all training rounds', '#777777'),
            (performance, 'LinearPerf: all rounds (includes greedy windows)', '#0072B2')]:
        x = run['episodes'][199:]
        axes[0].plot(x, 100 * rolling_mean(run['wins'], 200, True), lw=1.4,
                     label=label + '; 200-round mean', color=color)
        axes[1].plot(x, rolling_mean(run['rewards'], 200, True), lw=1.4, color=color)
    axes[0].plot(blocks[:, 0], blocks[:, 1], '-o', ms=4, lw=.9, color='#D55E00',
                 label='LinearPerf greedy windows only; 10 rounds per point')
    axes[1].plot(blocks[:, 0], blocks[:, 2], '-o', ms=4, lw=.9, color='#D55E00')
    cutoff = performance['cutoff']
    end = max(original['episodes'][-1], performance['episodes'][-1])
    for axis in axes:
        axis.axvline(cutoff + .5, ls='--', color='#333333', lw=1)
        axis.axvspan(cutoff + .5, end, color='#e6f1e7', alpha=.7)
        axis.grid(alpha=.2)
        axis.set_xlim(1, end)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel('Win rate (%)')
    axes[0].legend(fontsize=8, loc='lower right')
    axes[1].set_ylabel('Mean round reward')
    axes[1].set_xlabel('Completed rounds (learning remains enabled in every phase)')
    fig.suptitle('Q-learning: periodic greedy performance versus training performance')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_greedy_windows_only(performance, output):
    blocks = greedy_blocks(performance)
    if not len(blocks):
        raise ValueError('No periodic greedy windows to plot')
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout='constrained')
    axes[0].plot(blocks[:, 0], blocks[:, 1], '-o', color='#D55E00', lw=1.2, ms=4)
    axes[0].set(ylabel='Greedy-window win rate (%)', ylim=(-3, 103))
    axes[1].plot(blocks[:, 0], blocks[:, 2], '-o', color='#D55E00', lw=1.2, ms=4)
    axes[1].axhline(0, lw=.7, color='#777777')
    axes[1].set(ylabel='Greedy-window mean reward',
                xlabel='Completed training rounds (window midpoint)')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.set_xlim(0, performance['cutoff'])
    n = performance['schedule'].perf_rounds
    fig.suptitle('Q-learning: greedy performance windows only\n'
                 f'Each point summarizes {n} rounds with epsilon = 0; learning remains enabled',
                 fontsize=13)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
