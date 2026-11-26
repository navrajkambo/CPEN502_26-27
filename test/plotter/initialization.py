"""Compare the zero-epsilon runs with zero and random Q-value initialization."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .common import rolling_mean


def plot_zero_initializations(zero, random, output):
    if np.any(zero['epsilon']) or np.any(random['epsilon']):
        raise ValueError('Initialization comparison requires epsilon = 0 throughout')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), sharex='col')
    for run, label, color in [(zero, 'Zero initialization', '#0072B2'),
                               (random, 'Random initialization', '#D55E00')]:
        for col, stop in [(0, len(run['episodes'])), (1, 1000)]:
            x = run['episodes'][:stop]
            axes[0, col].plot(x, 100 * rolling_mean(run['wins'])[:stop],
                              label=label, color=color, lw=1.45)
            axes[1, col].plot(x, rolling_mean(run['rewards'])[:stop],
                              label=label, color=color, lw=1.45)
    axes[0, 0].set_title(f"All {len(zero['episodes']):,} rounds")
    axes[0, 1].set_title('First 1,000 rounds')
    for col in [0, 1]:
        axes[0, col].set_ylim(0, 102)
        axes[0, col].set_ylabel('Win rate (%) over last 50 rounds')
        axes[1, col].set_ylabel('Mean reward over last 50 rounds')
        axes[1, col].set_xlabel('Completed training rounds')
        axes[0, col].legend(loc='lower right', framealpha=.95)
        for row in [0, 1]:
            axes[row, col].grid(alpha=.2)
            axes[row, col].set_xlim(1, len(zero['episodes']) if col == 0 else 1000)
    fig.suptitle('Zero epsilon: zero versus random Q-value initialization', fontsize=15)
    fig.text(.5, .012, 'Both runs: epsilon = 0 throughout; learning remains enabled. '
             'Trailing 50-round means use available rounds at the start.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .955])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return output
