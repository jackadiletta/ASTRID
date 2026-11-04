// Match test_graphic_2.ipynb: normalize first, then slice [499:-500].
export function normalize(values) {
  let min = Infinity, max = -Infinity;
  for (const value of values) if (Number.isFinite(value)) { min = Math.min(min, value); max = Math.max(max, value); }
  return Array.from(values, value => Number.isFinite(value) ? (max === min ? 0 : (value - min) / (max - min)) : null);
}
export function trajectory(columns, window = 1000) {
  const { lat, lon, accl_x, accl_y, accl_z } = columns;
  const n = lat.length;
  if (n < window) throw new Error(`This session has ${n} samples; the notebook requires at least ${window}.`);
  const acceleration = Array.from({ length: n }, (_, i) => [accl_x[i], accl_y[i], accl_z[i]].every(v => v != null && Number.isFinite(Number(v))) ? Math.hypot(accl_x[i], accl_y[i], accl_z[i]) : NaN);
  // Track invalid values separately so one missing value does not poison every later window.
  let sum = 0, invalid = 0;
  const smoothed = [];
  for (let i = 0; i < n; i++) {
    if (Number.isFinite(acceleration[i])) sum += acceleration[i]; else invalid++;
    if (i >= window) { const old = acceleration[i - window]; if (Number.isFinite(old)) sum -= old; else invalid--; }
    if (i >= window - 1) smoothed.push(invalid ? NaN : sum / window);
  }
  const start = window / 2 - 1, end = n - window / 2;
  // The supplied files have a pandas RangeIndex, so notebook time is row position.
  const index = columns.index || Array.from({ length: n }, (_, i) => i);
  return {
    x: normalize(lat).slice(start, end), y: normalize(lon).slice(start, end),
    z: normalize(index).slice(start, end), color: normalize(smoothed),
    hover: Array.from({ length: end - start }, (_, j) => [String(index[j + start]), columns.time ? columns.time[j + start] : '']),
    rows: n,
  };
}
