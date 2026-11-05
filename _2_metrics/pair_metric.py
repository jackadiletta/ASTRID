"""Activity–gap concentration on the same complete windows as Jack's metric.

An activity is a continuous run above 4.44 m/s lasting at least one second.
A gap is time between consecutive retained activities, not recovery time.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from jack_metric import HIGH_SPEED, MAX_STEP, SECOND, WINDOW, preprocess, window_summary

MIN_ACTIVITY = SECOND
PAIR_COLUMNS = ['block_id', 'elapsed_start_s', 'end_s', 'next_start_s',
                'duration_s', 'following_gap_s']


def score_pairs(pairs) -> float | None:
    """Duration-weighted mean of D / (D + G); no usable pair means missing."""
    if len(pairs) == 0:
        return None
    duration = np.asarray(pairs['duration_s'], dtype=float)
    gap = np.asarray(pairs['following_gap_s'], dtype=float)
    if not (np.isfinite(duration).all() and np.isfinite(gap).all()
            and (duration > 0).all() and (gap >= 0).all()):
        raise ValueError('Each pair needs a positive duration and a nonnegative finite gap')
    return float(np.sum(duration * duration / (duration + gap)) / duration.sum())


def _eligible_intervals(gps, complete):
    """Split sample intervals at window edges, retaining complete windows only."""
    time = gps.elapsed_ns.to_numpy()
    speed, valid = gps.speed.to_numpy(), gps.valid.to_numpy()
    for i in range(len(time) - 1):
        start, stop = time[i], time[i + 1]
        if not (valid[i] and valid[i + 1]) or stop - start > MAX_STEP:
            continue
        while start < stop:
            window = start // WINDOW
            end = min(stop, (window + 1) * WINDOW)
            if complete[window]:
                yield int(start), int(end), bool(speed[i] > HIGH_SPEED)
            start = end


def activity_gap_pairs(table):
    """Extract adjacent retained activities within continuous eligible blocks.

    Complete adjacent windows stay connected. Missing or excluded time breaks
    the block. Activities touching a block boundary are conservatively censored
    and excluded from pairs. Short fast runs can remain inside a following gap.
    No interval is extended beyond its last observed timestamp.
    """
    gps = preprocess(table)
    windows = window_summary(table)
    activities = []
    run_start, run_left_censored = None, False
    previous_end, block = None, -1

    def finish_run(end, right_censored):
        nonlocal run_start
        if run_start is not None and end - run_start >= MIN_ACTIVITY:
            activities.append({'block_id': block, 'start_ns': run_start, 'end_ns': end,
                               'censored': run_left_censored or right_censored})
        run_start = None

    for start, end, high in _eligible_intervals(gps, windows.complete.to_numpy()):
        new_block = previous_end is None or start != previous_end
        if new_block:
            if previous_end is not None:
                finish_run(previous_end, right_censored=True)
            block += 1
        if high and run_start is None:
            run_start, run_left_censored = start, new_block
        elif not high:
            finish_run(start, right_censored=False)
        previous_end = end
    if previous_end is not None:
        finish_run(previous_end, right_censored=True)

    rows = []
    for current, following in zip(activities, activities[1:]):
        if current['block_id'] != following['block_id']:
            continue
        if current['censored'] or following['censored']:
            continue
        start, end, next_start = current['start_ns'], current['end_ns'], following['start_ns']
        rows.append([current['block_id'], start / SECOND, end / SECOND, next_start / SECOND,
                     (end - start) / SECOND, (next_start - end) / SECOND])
    pairs = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    pairs.attrs.update(retained_activities=len(activities),
                       censored_retained_activities=sum(a['censored'] for a in activities),
                       complete_windows=int(windows.complete.sum()),
                       total_windows=len(windows),
                       eligible_seconds=float(windows.loc[windows.complete, 'valid_seconds'].sum()))
    return pairs


def generate_pair_metric(parquet_path: str | Path) -> float | None:
    table = pq.ParquetFile(parquet_path).read(columns=['time', 'speed', 'lat', 'lon'])
    return score_pairs(activity_gap_pairs(table))
