"""Jack's space-time intensity metric for one SoccerMon GPS recording.

Use generate_metric(path) for the score, or window_summary(table) to inspect
the three measurements that go into it. Time is measured from device start.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SECOND = 1_000_000_000  # integer nanoseconds keep window boundaries exact
WINDOW = 30 * SECOND
MAX_STEP = 150_000_000  # 0.15 seconds
HIGH_SPEED = 4.44  # m/s
CELL_SIZE = 5.0  # metres
COMPONENTS = ['max_acceleration', 'longest_high_speed_s', 'occupied_cells']


def preprocess(table):
    """Collapse repeated GPS clocks and flag samples usable by all three measures."""
    raw = table.select(['time', 'speed', 'lat', 'lon']).to_pandas()
    if raw.time.isna().any():
        raise ValueError('Missing GPS time')
    first = raw.time.ne(raw.time.shift()).fillna(True)
    gps = raw.loc[first].reset_index(drop=True).copy()
    values = raw[['speed', 'lat', 'lon']]
    consistent = values.groupby(first.cumsum()).nunique(dropna=False).eq(1).all(axis=1)
    clock = pd.to_timedelta(gps.time, errors='raise')
    if clock.isna().any() or (clock.diff().dropna() <= pd.Timedelta(0)).any():
        raise ValueError('GPS time must increase after duplicate clocks are collapsed')
    ns = clock.to_numpy(dtype='timedelta64[ns]').astype('int64')
    gps['elapsed_ns'] = ns - ns[0] if len(ns) else ns
    gps['valid'] = (consistent.to_numpy() & gps.speed.between(0, 12)
                    & gps.lat.between(-90, 90) & gps.lon.between(-180, 180)
                    & ((gps.lat != 0) | (gps.lon != 0)))

    # Use the median usable position as the origin of a local east/north grid.
    good = gps.loc[gps.valid]
    lat0, lon0 = good.lat.median(), good.lon.median()
    east_m = 6_371_000 * np.cos(np.deg2rad(lat0)) * np.deg2rad(good.lon - lon0)
    north_m = 6_371_000 * np.deg2rad(good.lat - lat0)
    gps['cell_x'] = np.floor(east_m / CELL_SIZE)
    gps['cell_y'] = np.floor(north_m / CELL_SIZE)
    return gps


def window_summary(table):
    """Measure observed 30-second windows using the intervals between GPS samples."""
    gps = preprocess(table)
    time = gps.elapsed_ns.to_numpy()
    speed, valid = gps.speed.to_numpy(), gps.valid.to_numpy()
    cell_x, cell_y = gps.cell_x.to_numpy(), gps.cell_y.to_numpy()
    count = int((time[-1] + WINDOW - 1) // WINDOW) if len(time) else 0
    observed = np.zeros(count, dtype='int64')
    max_acceleration = np.zeros(count)
    longest_run = np.zeros(count, dtype='int64')
    cells = [set() for _ in range(count)]
    high_start, high_end, high_window = 0, -1, -1

    for i in range(len(time) - 1):
        start, stop = time[i], time[i + 1]
        if not (valid[i] and valid[i + 1]) or stop - start > MAX_STEP:
            continue
        acceleration = (speed[i + 1] - speed[i]) / ((stop - start) / SECOND)
        # An observed interval can straddle a window boundary. Credit each side once.
        while start < stop:
            window = start // WINDOW
            end = min(stop, (window + 1) * WINDOW)
            observed[window] += end - start
            max_acceleration[window] = max(max_acceleration[window], acceleration, 0.0)
            cells[window].add((cell_x[i], cell_y[i]))
            if speed[i] > HIGH_SPEED:
                if start != high_end or window != high_window:
                    high_start = start
                longest_run[window] = max(longest_run[window], end - high_start)
                high_end, high_window = end, window
            start = end

    windows = pd.DataFrame({'window_id': np.arange(count),
                            'midpoint_min': (np.arange(count) + 0.5) * 30 / 60,
                            'valid_seconds': observed / SECOND,
                            'complete': observed == WINDOW,
                            'max_acceleration': max_acceleration,
                            'longest_high_speed_s': longest_run / SECOND,
                            'occupied_cells': [len(group) for group in cells]})
    windows.loc[observed == 0, COMPONENTS] = np.nan
    return windows


def component_slopes(windows):
    """Fit one straight-line trend per component on the same complete windows."""
    complete = windows.loc[windows.complete]
    if len(complete) < 3:
        return None
    slopes = {}
    for name in COMPONENTS:
        values = complete[name]
        span = values.max() - values.min()
        if span == 0:
            slopes[name] = 0.0
        else:
            normalized = (values - values.min()) / span
            slope, _ = np.polyfit(complete.midpoint_min, normalized, deg=1)
            slopes[name] = float(slope)
    return slopes


def score_windows(windows) -> float | None:
    slopes = component_slopes(windows)
    return sum(slopes.values()) if slopes is not None else None


def generate_metric(parquet_path: str | Path) -> float | None:
    table = pq.ParquetFile(parquet_path).read(columns=['time', 'speed', 'lat', 'lon'])
    return score_windows(window_summary(table))
