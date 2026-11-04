"""Local Flask API and JavaScript trajectory viewer."""
import base64
import csv
from datetime import date
from io import BytesIO
from threading import Lock
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, abort, jsonify, request, send_from_directory
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from plotly.offline import get_plotlyjs
from werkzeug.exceptions import HTTPException

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
FIELDS = ("team", "player", "year", "month", "day")
COLUMNS = ["lat", "lon", "accl_x", "accl_y", "accl_z"]
WINDOW = 1000
GRID_SIZE = 50
EARTH_RADIUS_M = 6371000.0
REPORTS = ROOT / "soccer_mon/subjective/game-performance/game-performance.csv"
PLOT_LOCK = Lock()
app = Flask(__name__, static_folder=None)

@lru_cache(maxsize=1)
def read_manifest(stamp):
    with (ROOT / "_0_data/manifest.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set((*FIELDS, "file_path", "Game_Day")) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}. Run _0_generate_manifest.py.")
        rows = list(reader)
    if not rows or any(not row[key] for row in rows for key in (*FIELDS, "file_path")):
        raise ValueError("Manifest is empty or contains empty required values")
    for row in rows:
        if row["Game_Day"] not in ("True", "False"):
            raise ValueError(f"Invalid Game_Day value: {row['Game_Day']!r}")
        row["is_game_day"] = row["Game_Day"] == "True"
        session_date(row)  # Reject invalid calendar dates before selection.
    keys = [tuple(row[key] for key in FIELDS) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Manifest contains duplicate session selections")
    return rows

def manifest():
    return read_manifest((ROOT / "_0_data/manifest.csv").stat().st_mtime_ns)

def normalize(values, name):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    span = np.ptp(values)
    if span == 0:
        raise ValueError(f"{name} is constant; notebook normalization divides by zero")
    return (values - values.min()) / span

def spatial_coordinates(latitude, longitude):
    """Local east/north meters, preserving distance instead of stretching each axis."""
    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)
    if not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        raise ValueError("GPS coordinates contain non-finite values")
    if (np.abs(latitude) > 90).any() or (np.abs(longitude) > 180).any():
        raise ValueError("GPS coordinates are outside valid latitude/longitude ranges")
    origin_lat = float(np.median(latitude))
    origin_lon = float(np.median(longitude))
    east = EARTH_RADIUS_M * np.cos(np.radians(origin_lat)) * np.radians(longitude - origin_lon)
    north = EARTH_RADIUS_M * np.radians(latitude - origin_lat)
    return east, north, {"latitude": origin_lat, "longitude": origin_lon}


def forward_fill_gps(frame):
    """Replace zero/missing GPS coordinates with the preceding valid value."""
    coordinates = frame[["lat", "lon"]].replace(0, np.nan).ffill()
    if coordinates.isna().any().any():
        raise ValueError("GPS coordinates begin with zero or missing values; no previous value exists to forward-fill")
    return coordinates


def trajectory(frame):
    if len(frame) < WINDOW:
        raise ValueError(f"Session has {len(frame)} samples; at least {WINDOW} are required")
    coordinates = forward_fill_gps(frame)
    east, north, origin = spatial_coordinates(coordinates["lat"].to_numpy(), coordinates["lon"].to_numpy())
    axes = [east, north, normalize(frame.index.to_numpy(dtype=float), "index")]
    acceleration = np.sqrt(np.square(frame[["accl_x", "accl_y", "accl_z"]].to_numpy(dtype=float)).sum(axis=1))
    if not np.isfinite(acceleration).all():
        raise ValueError("Acceleration contains non-finite values")
    if np.ptp(acceleration) == 0:
        raise ValueError("Acceleration is constant; normalization divides by zero")
    cumulative = np.concatenate(([0.0], np.cumsum(acceleration)))
    smoothed = (cumulative[WINDOW:] - cumulative[:-WINDOW]) / WINDOW
    start, stop = WINDOW // 2 - 1, len(frame) - WINDOW // 2
    if np.ptp(east[start:stop]) == 0 or np.ptp(north[start:stop]) == 0:
        raise ValueError("GPS spatial coordinates are constant on a plotted axis")
    return dict(zip(("x", "y", "z"), [axis[start:stop].tolist() for axis in axes]),
                color=normalize(smoothed, "smoothed acceleration").tolist(),
                hover=frame.index[start:stop].astype(str).tolist(), rows=len(frame),
                latitude=coordinates["lat"].iloc[start:stop].tolist(),
                longitude=coordinates["lon"].iloc[start:stop].tolist(), origin=origin,
                acceleration_min=float(smoothed.min()), acceleration_max=float(smoothed.max()))

@app.get("/")
def index():
    return send_from_directory(SITE, "index.html")

@app.get("/assets/viewer.js")
def viewer_js():
    return send_from_directory(SITE, "viewer.js")

@app.get("/assets/plotly.js")
def plotly_js():
    response = Response(get_plotlyjs(), mimetype="application/javascript")
    response.cache_control.max_age = 86400
    return response

@app.get("/api/manifest")
def sessions():
    return jsonify([{key: row[key] for key in FIELDS} for row in manifest()])

def session_date(row):
    return date(int(row["year"]), int(row["month"]), int(row["day"]))


def selected_session():
    if set(request.args) != set(FIELDS) or any(len(request.args.getlist(key)) != 1 for key in FIELDS):
        abort(400, description="Provide exactly team, player, year, month, and day")
    for row in manifest():
        if all(row[key] == request.args[key] for key in FIELDS):
            return row
    abort(404, description="Session not found in manifest")


def load_trajectory(row):
    path = Path(row["file_path"])
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT):
        raise ValueError("Manifest path is outside the project")
    result = trajectory(pd.read_parquet(path, columns=COLUMNS, engine="pyarrow"))
    result["file"] = str(path.relative_to(ROOT))
    return result


def adjacent_game_session(selected, rows, direction):
    """Closest available game for this player, up to 14 days before/after X."""
    selected_date = session_date(selected)
    candidates = [row for row in rows
                  if row["team"] == selected["team"] and row["player"] == selected["player"]
                  and row["is_game_day"]
                  and 0 < direction * (session_date(row) - selected_date).days <= 14]
    if not candidates:
        return None
    return min(candidates, key=lambda row: abs((session_date(row) - selected_date).days))


def next_game_session(selected, rows):
    return adjacent_game_session(selected, rows, direction=1)


def prior_game_session(selected, rows):
    return adjacent_game_session(selected, rows, direction=-1)


def spatial_bins(track):
    edges = []
    for axis in ("x", "y"):
        values = np.asarray(track[axis], dtype=float)
        if not np.isfinite(values).all() or np.ptp(values) == 0:
            raise ValueError(f"Heatmap {axis} coordinates must have finite, nonzero spatial extent")
        edges.append(np.linspace(values.min(), values.max(), GRID_SIZE + 1))
    return edges


def acceleration_heatmap(track):
    """Mean acceleration in 50 x 50 bins over the actual spatial extent."""
    x_edges, y_edges = spatial_bins(track)
    sums, _, _ = np.histogram2d(track["x"], track["y"], bins=[x_edges, y_edges], weights=track["color"])
    counts, _, _ = np.histogram2d(track["x"], track["y"], bins=[x_edges, y_edges])
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


def heatmap_png(track, show_colorbar=True):
    # Independent figures and a lock keep concurrent Flask requests safe.
    with PLOT_LOCK:
        figure = Figure(figsize=(6, 5) if show_colorbar else (6, 3))
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        x_edges, y_edges = spatial_bins(track)
        im = ax.imshow(acceleration_heatmap(track).T, origin="lower",
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                       aspect="equal", cmap="Blues", vmin=0, vmax=1)
        ax.set_xlabel("East from reference (m)")
        ax.set_ylabel("North from reference (m)")
        if show_colorbar:
            colorbar = figure.colorbar(im, ax=ax, shrink=0.6)
            colorbar.set_label("Normalized Acceleration")
        figure.tight_layout()
        output = BytesIO()
        figure.savefig(output, format="png")
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def shared_game_scale(tracks):
    """Rescale both games using their combined smoothed acceleration range."""
    if not tracks:
        return None
    low = min(track["acceleration_min"] for track in tracks.values())
    high = max(track["acceleration_max"] for track in tracks.values())
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("Shared acceleration range must be finite and nonconstant")
    for track in tracks.values():
        raw = (np.asarray(track["color"]) * (track["acceleration_max"] - track["acceleration_min"])
               + track["acceleration_min"])
        track["color"] = ((raw - low) / (high - low)).tolist()
    return {"min": float(low), "max": float(high)}


def shared_colorbar_png():
    """One compact horizontal color bar for the two stacked game heatmaps."""
    with PLOT_LOCK:
        figure = Figure(figsize=(6, 0.65))
        FigureCanvasAgg(figure)
        ax = figure.add_axes([0.08, 0.38, 0.84, 0.22])
        colorbar = figure.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="Blues"),
                                  cax=ax, orientation="horizontal", ticks=np.linspace(0, 1, 6))
        ax.set_title("Normalized acceleration · shared scale", fontsize=9, pad=4)
        colorbar.ax.tick_params(labelsize=8, pad=2, length=2)
        output = BytesIO()
        figure.savefig(output, format="png")
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


@lru_cache(maxsize=1)
def read_game_reports(stamp):
    reports = pd.read_csv(REPORTS, dtype={"player_name": str, "timestamp": str})
    required = ["player_name", "timestamp", "team_performance", "offensive_performance", "defensive_performance"]
    if reports[required].isna().any().any():
        raise ValueError("Game reports contain missing required values")
    reports["date"] = pd.to_datetime(reports["timestamp"], format="%d.%m.%Y", errors="raise").dt.date
    reports["team"] = reports["player_name"].str.split("-").str[0]
    for column in required[2:]:
        scores = pd.to_numeric(reports[column], errors="raise")
        if not (scores.between(1, 10) & (scores % 1 == 0)).all():
            raise ValueError(f"Invalid game rating in {column}")
        reports[column] = scores.astype(int)
    return reports


def game_statistics(row):
    """Read actual ratings, rather than the manifest's formatted DataFrame text."""
    reports = read_game_reports(REPORTS.stat().st_mtime_ns)
    matches = reports.loc[(reports["date"] == session_date(row)) & (reports["team"] == "Team" + row["team"])]
    statistics = matches[["player_name", "team_performance", "offensive_performance", "defensive_performance"]].to_dict("records")
    for report in statistics:
        report["selected_player"] = report["player_name"] == row["player"]
    return statistics


@app.get("/api/session")
def session():
    row = selected_session()
    result = load_trajectory(row)
    result["is_game_day"] = row["is_game_day"]
    result["date"] = session_date(row).isoformat()
    return Response(json.dumps(result, allow_nan=False, separators=(",", ":")), mimetype="application/json")


def game_response(direction):
    selected = selected_session()
    row = adjacent_game_session(selected, manifest(), direction)
    if row is None:
        message = "No prior game within 2 weeks." if direction == -1 else "No game within 2 weeks."
        return jsonify(game=None, message=message)
    track = load_trajectory(row)
    days = abs((session_date(row) - session_date(selected)).days)
    return jsonify(game={
        "date": session_date(row).isoformat(),
        "days_before" if direction == -1 else "days_after": days,
        "file": track["file"], "rows": track["rows"], "origin": track["origin"],
        "image": heatmap_png(track), "statistics": game_statistics(row),
    })


@app.get("/api/games")
def games():
    selected = selected_session()
    rows = manifest()
    matches = {"prior": prior_game_session(selected, rows), "next": next_game_session(selected, rows)}
    tracks = {direction: load_trajectory(row) for direction, row in matches.items() if row is not None}
    scale = shared_game_scale(tracks)
    if scale is not None:
        scale["image"] = shared_colorbar_png()
    result = {"scale": scale}
    for direction, row in matches.items():
        if row is None:
            message = "No prior game within 2 weeks." if direction == "prior" else "No game within 2 weeks."
            result[direction] = {"game": None, "message": message}
            continue
        track = tracks[direction]
        days = abs((session_date(row) - session_date(selected)).days)
        result[direction] = {"game": {
            "date": session_date(row).isoformat(),
            "days_before" if direction == "prior" else "days_after": days,
            "file": track["file"], "rows": track["rows"], "origin": track["origin"],
            "image": heatmap_png(track, show_colorbar=False), "statistics": game_statistics(row),
        }}
    return jsonify(result)


@app.get("/api/next-game")
def next_game():
    return game_response(direction=1)


@app.get("/api/prior-game")
def prior_game():
    return game_response(direction=-1)

@app.errorhandler(Exception)
def error(exc):
    if isinstance(exc, HTTPException):
        return jsonify(error=exc.description), exc.code
    app.logger.exception("Request failed")
    return jsonify(error=f"{type(exc).__name__}: {exc}"), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, threaded=True)
