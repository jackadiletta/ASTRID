"""Small regression checks: python test_pair_metric.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from pair_metric import activity_gap_pairs, score_pairs


def trace(bouts, seconds=120, missing_time=None):
    """Artificial 10 Hz signal for extraction checks, not a physical athlete model."""
    ticks = np.arange(seconds * 10 + 1)
    speed = np.zeros(len(ticks))
    for start, end in bouts:
        speed[(ticks >= round(start * 10)) & (ticks < round(end * 10))] = 5.0
    keep = ticks != round(missing_time * 10) if missing_time is not None else np.ones(len(ticks), bool)
    return pa.table({'time': pd.to_timedelta(ticks[keep] * 100, unit='ms').astype(str),
                     'speed': speed[keep], 'lat': np.full(keep.sum(), 60.0),
                     'lon': np.full(keep.sum(), 10.0)})


def check():
    results = []

    # Same duration and gap distributions, different pairing.
    a = pd.DataFrame({'duration_s': [2, 8], 'following_gap_s': [2, 8]})
    b = pd.DataFrame({'duration_s': [2, 8], 'following_gap_s': [8, 2]})
    assert np.isclose(score_pairs(a), .5)
    assert np.isclose(score_pairs(b), .68)
    assert np.isclose(score_pairs(b * 2), .68)
    results.append('Known scores 0.50 and 0.68, and scale invariance: PASS')

    # The excluded 0.5-second fast bout remains inside the measured gap.
    pairs = activity_gap_pairs(trace([(1, 2), (3, 3.5), (5, 6)]))
    assert len(pairs) == 1 and pairs.iloc[0].duration_s == 1
    assert pairs.iloc[0].following_gap_s == 3
    results.append('Short excluded fast activity stays inside following gap: PASS')

    # A high-speed run continues through a complete 30-second window boundary.
    pairs = activity_gap_pairs(trace([(29.5, 31.5), (40, 42)]))
    assert len(pairs) == 1 and pairs.iloc[0].duration_s == 2
    assert pairs.iloc[0].following_gap_s == 8.5
    results.append('Adjacent complete windows do not split activity: PASS')

    # A missing GPS step invalidates its window and cuts the eligible block.
    pairs = activity_gap_pairs(trace([(10, 12), (70, 72)], missing_time=40))
    assert len(pairs) == 0 and pairs.attrs['complete_windows'] == 3
    results.append('Missing time prevents a cross-block pair: PASS')

    # A final boundary-clipped bout has no invented end or following gap.
    pairs = activity_gap_pairs(trace([(10, 12), (119, 121)]))
    assert len(pairs) == 0 and pairs.attrs['censored_retained_activities'] == 1
    assert score_pairs(pairs) is None
    results.append('Boundary-censored activity stays excluded: PASS')

    # A normal final activity can close a preceding pair but has no pair of its own.
    pairs = activity_gap_pairs(trace([(10, 12), (20, 22)]))
    assert len(pairs) == 1 and pairs.iloc[0].next_start_s == 20
    empty = activity_gap_pairs(trace([(1, 3), (5, 7)], seconds=20))
    assert len(empty) == 0 and score_pairs(empty) is None
    results.append('No terminal following gap, no complete windows gives missing: PASS')

    print('\n'.join(results))
    output = Path(__file__).with_name('pair_test_results.json')
    output.write_text(json.dumps({'status': 'PASS', 'checks': results}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    check()
