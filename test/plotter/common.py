"""Read battle histories and compute the statistics shared by report plots."""
import csv
from pathlib import Path
import numpy as np
from . import ROOT


def load_history(filename):
    path = Path(filename)
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'No completed rounds to plot: {path}')
    return dict(episodes=np.array([int(r['episode']) for r in rows]),
                wins=np.array([r['outcome'] == 'win' for r in rows], dtype=float),
                rewards=np.array([float(r['reward']) for r in rows]),
                recorded_epsilon=np.array([float(r['epsilon']) for r in rows]),
                turns=np.array([int(r.get('turns', 0)) for r in rows]))


def rolling_mean(values, window=50, full_windows=False):
    """Trailing means; optionally omit the initial incomplete windows."""
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError('window must be a positive integer')
    values = np.asarray(values, dtype=float)
    totals = np.r_[0., np.cumsum(values)]
    ends = np.arange(window if full_windows else 1, len(values) + 1)
    starts = np.maximum(0, ends - window)
    return (totals[ends] - totals[starts]) / (ends - starts)


def load_run(folder, training_root=None):
    """Use checkpoint metadata to recover epsilon used, not epsilon after round."""
    from LUTInterface import EpsilonSchedule

    path = Path(training_root or ROOT / 'training') / folder / 'cappeeler.csv'
    run = load_history(path)
    episodes = run['episodes']
    if not np.array_equal(episodes, np.arange(1, len(episodes) + 1)):
        raise ValueError(f'Expected consecutive episodes starting at 1: {path}')
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as data:
        modes = {mode.value.casefold(): mode for mode in EpsilonSchedule.Decay}
        mode = modes[str(data['epsilon_decay_mode'].item()).casefold()]
        schedule = EpsilonSchedule(
            float(data['epsilon_start']), int(data['epsilon_zero_at_battle']), mode,
            int(data['epsilon_perf_interval']) if 'epsilon_perf_interval' in data else 200,
            int(data['epsilon_perf_rounds']) if 'epsilon_perf_rounds' in data else 10)
        if int(data['episodes']) != len(episodes):
            raise ValueError(f'CSV/checkpoint episode mismatch: {path}')
    np.testing.assert_allclose(run['recorded_epsilon'],
                               [schedule.value(int(e)) for e in episodes], atol=1e-12)
    run.update(epsilon=np.array([schedule.value(int(e) - 1) for e in episodes]),
               probe=np.array([schedule.is_performance_round(int(e) - 1) for e in episodes]),
               start=schedule.start, mode=schedule.mode.value, cutoff=schedule.zero_at_battle,
               schedule=schedule, source=path.as_posix())
    try:
        run['source'] = path.relative_to(ROOT).as_posix()
    except ValueError:
        pass
    return run


def greedy_blocks(run):
    """One sample per periodic greedy window; exclude the final permanent zero phase."""
    schedule = run['schedule']
    selected = run['probe']
    groups = (run['episodes'] - 1) // schedule.perf_interval
    samples = []
    for group in np.unique(groups[selected]):
        mask = selected & (groups == group)
        samples.append((run['episodes'][mask].mean(), 100 * run['wins'][mask].mean(),
                        run['rewards'][mask].mean(), int(mask.sum())))
    return np.asarray(samples, dtype=float).reshape(-1, 4)
