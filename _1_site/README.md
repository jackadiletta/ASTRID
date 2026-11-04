# ASTRID session viewer

From the repository root:

```sh
conda activate ASTRID_ENV
python3 -m pip install -r _1_site/requirements.txt
python3 _0_generate_manifest.py
python3 _1_site/app.py
```

Open http://127.0.0.1:8000. Select team → player → year → month → day. Each choice enables the next dropdown and clears later choices. Selecting a day loads that session and the closest prior and next available game sessions for the same player.

Flask reads `_0_data/manifest.csv` and the original Parquet paths it names. The columns `year`, `month`, `day`, `team`, `player`, `file_path`, and `Game_Day` are required. `Game_Day` must be exactly `True` or `False`. The generator extracts the complete player UUID from each filename. Relative data paths are resolved against the repository root. Run the generator after adding sessions.

Plain JavaScript renders a local Plotly 3D trajectory using local east/north coordinates in meters, with normalized pandas index as time. Spatial coordinates are no longer normalized separately. Longitude offsets are scaled by the cosine of the session median latitude; both axes use a spherical Earth radius of 6,371,000 m. This local equirectangular projection preserves field-scale proportions. The reference point is the session median latitude/longitude. The forward-filled latitude/longitude values are available on hover. Horizontal axes use equal meter scaling; the time axis has an independent display height so it remains visible. Acceleration is `sqrt(accl_x² + accl_y² + accl_z²)` with a 1,000-sample rolling mean normalized to [0,1]. Coordinates use the notebook slice `[499:-500]`; hover indexes are aligned to that slice. All points are retained. The rolling mean uses cumulative sums for linear runtime. Plotly is served from the installed Python package: no CDN, Node, or frontend build.

There are no substitute data sources or normalization fallbacks. Missing files, schema errors, short sessions, non-finite data, and constant normalization inputs produce visible errors and server tracebacks. Selection errors return 400/404. The manifest is cached until its modification time changes; Parquet data is read fresh per selection. The old browser-only worker modules are unused by this Flask app.

Run checks with `python3 -m unittest discover -s _1_site -p 'test_*.py'`. Requires a browser with WebGL. This is a local development server.

From any working directory, run the repository's `./launch_site.sh` to stop an existing instance of this checkout and start the site. Run `./stop_site.sh` from another terminal to stop it. Both scripts use ASTRID_ENV. The Linux process controller recognizes the original launcher and direct `python3 _1_site/app.py` processes too. It sends SIGTERM, waits up to five seconds, then sends SIGKILL if necessary. Unrelated servers using port 8000 are left alone; a port conflict remains a visible startup error.

The arc title labels the selected recording as Game day or Non-game session using the manifest flag. The right column contains separate prior-game and next-game panes. They find the closest `Game_Day=True` recording for the same team and player strictly before/after the selected date, each within 14 calendar days (day 14 included). It searches across month/year boundaries. A game on day X itself is excluded. If no matching recording exists, the corresponding pane stays empty and says “No prior game within 2 weeks.” or “No game within 2 weeks.” This is based on available recordings and report-derived manifest labels, not a complete external fixture schedule.

Both game heatmaps are static Matplotlib PNGs. They use 50 × 50 bins over the actual local east/north extent in meters, retaining the notebook's average jointly normalized smoothed acceleration per occupied cell, empty cells blank, transposed grid, lower origin, equal spatial aspect, `Blues`, and a fixed 0–1 color scale. This is acceleration intensity, not occupancy or distance traveled. No points are downsampled.

Game statistics come directly from `soccer_mon/subjective/game-performance/game-performance.csv`, matched by team and game date. The selected player's report is highlighted when available; teammate reports remain explicitly labeled. A valid CSV without matching reports shows “No game statistics available.” Missing/invalid source files produce visible errors. The manifest's `Game_Stats` is formatted DataFrame text and is not used as structured ratings. No manifest generator or data files are modified by the website.

No field rotation or boundary inference is applied. On load, zero and missing latitude/longitude values are forward-filled independently within each session before projection, reference-point calculation, and heatmap binning. No rows are removed; acceleration and timestamps are unchanged. Leading zero/missing values raise a visible error because no preceding valid value exists. Infinite or out-of-range coordinates still raise errors. The source Parquet files are never modified. The local projection is intended for field-scale data; large geographic jumps should be inspected in the source data.

The website loads both games together through `/api/games`. The combined minimum and maximum of the two games' 1,000-sample smoothed acceleration signals define one shared 0–1 normalization, applied before calculating the cell averages. Identical smoothed acceleration values therefore have identical colors across both games. When only one game is available, that game's range defines the scale; when neither is available, no color bar is shown. The arc keeps its session-specific acceleration normalization.

The two game heatmaps stack vertically in a column with the same total height as the arc pane. They use one vertical color bar to the right, spanning from the top heatmap area to the bottom heatmap area. The frontend legend uses the same 256-color Matplotlib Blues palette and the existing paired normalization. Game ratings are always visible beside each heatmap, with bounded scrolling for multiple reports. Image aspect ratios remain preserved. Sections use light dividing lines instead of nested cards; the redundant page heading is removed. The arc's own color bar is shortened as well. The old individual prior/next endpoints remain available, but the paired endpoint supplies the website's common scale.
