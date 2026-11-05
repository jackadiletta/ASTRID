"""Compare unchanged Jack scoring with activity-gap concentration.

Run without arguments to reuse the saved real tables and recreate examples.
To re-extract the same real case: python compare_metrics.py --parquet FILE
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from jack_metric import COMPONENTS, component_slopes, score_windows, window_summary
from pair_metric import activity_gap_pairs, score_pairs

HERE = Path(__file__).resolve().parent


def toy_recording(bouts):
    """Two minutes at 10 Hz. Given bouts use 5 m/s, all other times use zero.

    A straight path integrates the specified speed. These abrupt, idealized
    signals test the formulas, not physiological plausibility.
    """
    ticks = np.arange(1201)
    speed = np.zeros(len(ticks))
    for start, end in bouts:
        speed[(ticks >= round(start * 10)) & (ticks < round(end * 10))] = 5.0
    east_m = np.r_[0.0, np.cumsum(speed[:-1] * .1)]
    longitude = 10 + np.rad2deg(east_m / (6_371_000 * np.cos(np.deg2rad(60))))
    return pa.table({'time': pd.to_timedelta(ticks * 100, unit='ms').astype(str),
                     'speed': speed, 'lat': np.full(len(ticks), 60.0), 'lon': longitude})


def result(name, windows, pairs):
    return {'case': name, 'jack_metric_per_min': score_windows(windows),
            'activity_gap_concentration': score_pairs(pairs),
            'pair_count': len(pairs), 'paired_activity_seconds': float(pairs.duration_s.sum()),
            'following_gap_seconds': float(pairs.following_gap_s.sum()),
            'complete_windows': int(windows.complete.sum()),
            'candidate_windows': len(windows), 'component_slopes': component_slopes(windows),
            'pairs': pairs.to_dict(orient='records')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', type=Path)
    args = parser.parse_args()
    real_windows = pd.read_csv(HERE / 'real_30s_windows.csv')
    if args.parquet:
        raw = pq.ParquetFile(args.parquet).read(columns=['time', 'speed', 'lat', 'lon'])
        fresh_windows = window_summary(raw)
        pd.testing.assert_frame_equal(real_windows, fresh_windows, check_dtype=False,
                                      check_exact=False, atol=1e-10, rtol=0)
        real_pairs = activity_gap_pairs(raw)
        real_pairs.to_csv(HERE / 'real_activity_gap_pairs.csv', index=False)
        evidence = {'real_parquet_sha256': hashlib.sha256(args.parquet.read_bytes()).hexdigest(),
                    'source_bytes': args.parquet.stat().st_size, 'raw_rows': raw.num_rows,
                    'coverage': real_pairs.attrs,
                    'support_rule': 'Common complete 30-second windows from unchanged Jack preprocessing.'}
        (HERE / 'extraction_record.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    else:
        real_pairs = pd.read_csv(HERE / 'real_activity_gap_pairs.csv')

    cases = [result('real_TeamA_2020-08-03', real_windows, real_pairs)]
    examples = [('matched_pairs', [(2, 4), (6, 14), (22, 24)], .50),
                ('swapped_gaps', [(2, 4), (12, 20), (22, 24)], .68)]
    synthetic_tables, pair_tables, curve_tables = [], [], []
    for name, bouts, expected in examples:
        raw = toy_recording(bouts)
        windows = window_summary(raw)
        pairs = activity_gap_pairs(raw)
        np.testing.assert_allclose(score_pairs(pairs), expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(score_windows(windows), -1.8, atol=1e-12, rtol=0)
        assert len(pairs) == 2
        cases.append(result(name, windows, pairs))
        synthetic_tables.append(raw.to_pandas().assign(case=name))
        pair_tables.append(pairs.assign(case=name))
        curve_tables.append(windows.assign(case=name))
    for name in COMPONENTS:
        np.testing.assert_allclose(curve_tables[0][name], curve_tables[1][name], atol=1e-12, rtol=0)
    assert sorted(pair_tables[0].duration_s) == sorted(pair_tables[1].duration_s)
    assert sorted(pair_tables[0].following_gap_s) == sorted(pair_tables[1].following_gap_s)
    np.testing.assert_allclose(cases[0]['jack_metric_per_min'], -7.427626360835874e-5, atol=1e-12, rtol=0)

    pd.concat(synthetic_tables, ignore_index=True).to_csv(HERE / 'synthetic_raw_signals.csv', index=False)
    pd.concat(pair_tables, ignore_index=True).to_csv(HERE / 'synthetic_activity_gap_pairs.csv', index=False)
    pd.concat(curve_tables, ignore_index=True).to_csv(HERE / 'synthetic_30s_windows.csv', index=False)
    data = {'real': cases[0], 'synthetic': cases[1:],
            'formula': 'sum(D * D / (D + G)) / sum(D)', 'candidate_unit': 'dimensionless, 0 to 1',
            'status': 'Proposed feature. No predictive validation.',
            'support': 'Both scores use the same complete windows and source-speed/coordinate validity.',
            'checks': {'real_jack_matches_previous': 'PASS', 'synthetic_scores': 'PASS',
                       'same_synthetic_jack_components': 'PASS', 'same_duration_and_gap_distributions': 'PASS'}}
    (HERE / 'comparison.json').write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    summary = pd.DataFrame([{k: v for k, v in row.items() if k not in ['pairs', 'component_slopes']} for row in cases])
    summary.to_csv(HERE / 'comparison.csv', index=False)
    print(summary.to_string(index=False))
    print('All comparison checks passed.')


if __name__ == '__main__':
    main()
