import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from scipy.stats import linregress

def calculate_accel_speed_space_dist(path_to_file):
    '''
    
    Given a path to a parquet file - reads it and eventually collectes 
    these data per 30 seconds: mean, min, max, 10%, 25%, 75%, 90% for entire team for each metric. 

    the algorithm looks like:

    1) read parquet
    2) remove all slices of time with -1 or NaN values, backfill
    3) for each 30 second increment in a session, calculate metrics: avg speed, avg accel, spatial coverage, avg HR
    4) save in a ragged/sparse data structure with dims [30-sec inc, metric, value]

    '''

    player_data = pd.read_parquet(path_to_file)
    if len(player_data)==0:
        return None, None
    player_name = player_data["player_name"].iloc[0]
    assert player_data["player_name"].nunique() == 1, "Expected exactly one player per parquet file"
    # clean data
    measurement_columns = ["speed", "heart_rate", "accl_x", "accl_y", "accl_z"]
    player_data[["speed", "heart_rate"]] = player_data[["speed", "heart_rate"]].replace(-1, np.nan)
    missing_columns = player_data[measurement_columns].isna().all()
    assert not missing_columns.any(), f"Entire columns are missing in {path_to_file}: {missing_columns[missing_columns].index.tolist()}"
    player_data[measurement_columns] = player_data[measurement_columns].ffill().bfill()

    outliers = (
        player_data["speed"].lt(0)
        | player_data["speed"].gt(12)
        | player_data["heart_rate"].lt(0)
        | player_data["heart_rate"].gt(230)
    )
    if sum(outliers) > 0:
        print(player_data.loc[outliers])
        print(path_to_file)

    # check lat long
    player_data.drop(index=player_data.index[player_data["lat"].eq(0) | player_data["lon"].eq(0)], inplace=True)
    latitude = player_data["lat"]
    longitude = player_data["lon"]
    invalid_coordinates = (
        latitude.isna()
        | longitude.isna()
        | latitude.isin([0, -1])
        | longitude.isin([0, -1])
        | ~latitude.between(20, 72)
        | ~longitude.between(1.1, 40)
    )

    if invalid_coordinates.any():
        print("Invalid coords\n", player_data.loc[invalid_coordinates, ["time", "lat", "lon"]])
        player_data = player_data.loc[~invalid_coordinates].copy()
        # raise ValueError(f"Invalid GPS coordinates in {path_to_file}")

    player_data["elapsed_time"] = pd.to_timedelta(player_data["time"])
    player_data["elapsed_time"] -= player_data["elapsed_time"].iloc[0] # subtract min time from all times

    # tile spatial coverage
    tile_size_m = 2.0
    earth_radius_m = 6_371_000
    latitude_origin = player_data["lat"].median()
    longitude_origin = player_data["lon"].median()

    player_data["east_m"] = earth_radius_m * np.cos(np.deg2rad(latitude_origin)) * np.deg2rad(player_data["lon"] - longitude_origin)
    player_data["north_m"] = earth_radius_m * np.deg2rad(player_data["lat"] - latitude_origin)

    player_data["tile_x"] = np.floor(player_data["east_m"] / tile_size_m)
    player_data["tile_y"] = np.floor(player_data["north_m"] / tile_size_m)

    records = {
        "chunk_time": [],
        "average_speed": [],
        "average_acceleration_magnitude": [],
        "average_heart_rate": [],
        "occupied_tiles": [],
    }
    # chunk time calc
    session_date = pd.Timestamp(Path(path_to_file).name[:10])
    player_data["timestamp"] = pd.Timestamp(session_date).normalize() + pd.to_timedelta(player_data["time"])
    player_data["chunk_time"] = player_data["timestamp"].dt.floor("30s")

    for chunk_time, chunk in player_data.groupby(player_data["chunk_time"], sort=False):
        span_seconds = (chunk["timestamp"].iloc[-1] - chunk["timestamp"].iloc[0]).total_seconds()
        if span_seconds < 15:
            continue
        gps = chunk.drop_duplicates(subset="elapsed_time").sort_values("elapsed_time")

        average_speed = gps["speed"].mean()
        acceleration_magnitude = np.sqrt(chunk["accl_x"]**2 + chunk["accl_y"]**2 + chunk["accl_z"]**2)
        average_acceleration_magnitude = acceleration_magnitude.mean()
        average_heart_rate = gps.loc[gps["heart_rate"] > 0, "heart_rate"].mean()

        occupied_tiles = len(chunk[["tile_x", "tile_y"]].dropna().drop_duplicates())

        records["chunk_time"].append(chunk_time)
        records["average_speed"].append(average_speed)
        records["average_acceleration_magnitude"].append(average_acceleration_magnitude)
        records["average_heart_rate"].append(average_heart_rate)
        records["occupied_tiles"].append(occupied_tiles)

    return pd.DataFrame(records), player_name

def calc_global_dists(all_sessions):
    '''

    5) condense the player dimension and replace with distribution to make a ragged / space data structure with
    dims [30-sec inc, metric, distribution, value]

    '''
    all_sessions["team"] = all_sessions["player"].str.split("-", n=1).str[0]

    metric_columns = ["average_speed", "average_acceleration_magnitude", "average_heart_rate", "occupied_tiles"]

    records = {
        "team": [],
        "chunk_time": [],
        "metric": [],
        "distribution": [],
        "value": [],
    }

    groups = all_sessions.groupby(["team", "chunk_time"])

    for (team, chunk_time), chunk in tqdm(groups, total=groups.ngroups, desc="Calculating distributions"):
        for metric in metric_columns:
            values = chunk[metric].dropna()

            distributions = {
                "min": values.min(),
                "max": values.max(),
                "mean": values.mean(),
                "p10": values.quantile(0.10),
                "p25": values.quantile(0.25),
                "median": values.median(),
                "p75": values.quantile(0.75),
                "p90": values.quantile(0.90),
            }

            for distribution, value in distributions.items():
                records["team"].append(team)
                records["chunk_time"].append(chunk_time)
                records["metric"].append(metric)
                records["distribution"].append(distribution)
                records["value"].append(value)
    return pd.DataFrame(records)
                

def calculate_accel_speed_space_trends(input_data):
    path_to_file, global_dists = input_data
    '''
    Therefore, each parquet file needs to collect the trend for the speed
    acceleration, and spatial coverage for discretized 30 second segments,
    as well as average speed during session, average acceleration.
    '''
    player_data = pd.read_parquet(path_to_file)
    if len(player_data)==0:
        return None, None
    player_name = player_data["player_name"].iloc[0]
    assert player_data["player_name"].nunique() == 1, "Expected exactly one player per parquet file"
    # clean data
    measurement_columns = ["speed", "heart_rate", "accl_x", "accl_y", "accl_z"]
    player_data[["speed", "heart_rate"]] = player_data[["speed", "heart_rate"]].replace(-1, np.nan)
    missing_columns = player_data[measurement_columns].isna().all()
    assert not missing_columns.any(), f"Entire columns are missing in {path_to_file}: {missing_columns[missing_columns].index.tolist()}"
    player_data[measurement_columns] = player_data[measurement_columns].ffill().bfill()

    outliers = (
        player_data["speed"].lt(0)
        | player_data["speed"].gt(12)
        | player_data["heart_rate"].lt(0)
        | player_data["heart_rate"].gt(230)
    )
    if sum(outliers) > 0:
        print(player_data.loc[outliers])
        print(path_to_file)

    # check lat long
    player_data.drop(index=player_data.index[player_data["lat"].eq(0) | player_data["lon"].eq(0)], inplace=True)
    latitude = player_data["lat"]
    longitude = player_data["lon"]
    invalid_coordinates = (
        latitude.isna()
        | longitude.isna()
        | latitude.isin([0, -1])
        | longitude.isin([0, -1])
        | ~latitude.between(4, 72)
        | ~longitude.between(4, 32)
    )

    if invalid_coordinates.any():
        print("Invalid coords\n", player_data.loc[invalid_coordinates, ["time", "lat", "lon"]])
        player_data = player_data.loc[~invalid_coordinates].copy()

    player_data["elapsed_time"] = pd.to_timedelta(player_data["time"])
    player_data["elapsed_time"] -= player_data["elapsed_time"].iloc[0] # subtract min time from all times

    # tile spatial coverage
    tile_size_m = 2.0
    earth_radius_m = 6_371_000
    latitude_origin = player_data["lat"].median()
    longitude_origin = player_data["lon"].median()

    player_data["east_m"] = earth_radius_m * np.cos(np.deg2rad(latitude_origin)) * np.deg2rad(player_data["lon"] - longitude_origin)
    player_data["north_m"] = earth_radius_m * np.deg2rad(player_data["lat"] - latitude_origin)

    player_data["tile_x"] = np.floor(player_data["east_m"] / tile_size_m)
    player_data["tile_y"] = np.floor(player_data["north_m"] / tile_size_m)

    records = {
        "chunk_time": [],
        "mmn_normalized_average_speed": [],
        "mmn_normalized_average_acceleration_magnitude": [],
        "mmn_normalized_average_heart_rate": [],
        "mmn_normalized_occupied_tiles": [],
        "IQR_normalized_average_speed": [],
        "IQR_normalized_average_acceleration_magnitude": [],
        "IQR_normalized_average_heart_rate": [],
        "IQR_normalized_occupied_tiles": [],
    }
    # chunk time calc
    session_date = pd.Timestamp(Path(path_to_file).name[:10])
    player_data["timestamp"] = pd.Timestamp(session_date).normalize() + pd.to_timedelta(player_data["time"])
    player_data["chunk_time"] = player_data["timestamp"].dt.floor("30s")

    team = player_name.split("-", 1)[0]
    distribution_lookup = global_dists.set_index(["team", "chunk_time", "metric", "distribution"])["value"]
    assert distribution_lookup.index.is_unique, "Duplicate distribution entries"

    for chunk_time, chunk in player_data.groupby(player_data["chunk_time"], sort=False):
        span_seconds = (chunk["timestamp"].iloc[-1] - chunk["timestamp"].iloc[0]).total_seconds()
        if span_seconds < 15:
            continue
        gps = chunk.drop_duplicates(subset="elapsed_time").sort_values("elapsed_time")

        average_speed = gps["speed"].mean()
        acceleration_magnitude = np.sqrt(chunk["accl_x"]**2 + chunk["accl_y"]**2 + chunk["accl_z"]**2)
        average_acceleration_magnitude = acceleration_magnitude.mean()
        average_heart_rate = gps.loc[gps["heart_rate"] > 0, "heart_rate"].mean()

        occupied_tiles = len(chunk[["tile_x", "tile_y"]].dropna().drop_duplicates())

        records["chunk_time"].append(chunk_time)
        metric_values = {
            "average_speed": average_speed,
            "average_acceleration_magnitude": average_acceleration_magnitude,
            "average_heart_rate": average_heart_rate,
            "occupied_tiles": occupied_tiles,
        }
        for metric, value in metric_values.items():
            minimum = distribution_lookup.loc[(team, chunk_time, metric, "min")]
            maximum = distribution_lookup.loc[(team, chunk_time, metric, "max")]
            median = distribution_lookup.loc[(team, chunk_time, metric, "median")]
            p25 = distribution_lookup.loc[(team, chunk_time, metric, "p25")]
            p75 = distribution_lookup.loc[(team, chunk_time, metric, "p75")]

            minmax_range = maximum - minimum
            iqr = p75 - p25

            minmax_normalized = (value - minimum) / minmax_range if minmax_range > 0 else np.nan
            iqr_normalized = (value - median) / iqr if iqr > 0 else np.nan

            records[f"mmn_normalized_{metric}"].append(minmax_normalized)
            records[f"IQR_normalized_{metric}"].append(iqr_normalized)

    session_data = pd.DataFrame(records)
    elapsed_minutes = (session_data["chunk_time"] - session_data["chunk_time"].min()).dt.total_seconds() / 60

    normalized_metrics = [column for column in session_data.columns if column.startswith(("mmn_normalized_", "IQR_normalized_"))]
    trends = {"metric": [], "slope": [], "r_squared": []}

    for metric in normalized_metrics:
        values = session_data[metric]
        valid = np.isfinite(elapsed_minutes) & np.isfinite(values)

        if valid.sum() < 3 or elapsed_minutes.loc[valid].nunique() < 2:
            trends["metric"].append(metric)
            trends["slope"].append(np.nan)
            trends["r_squared"].append(np.nan)
            continue

        regression = linregress(elapsed_minutes.loc[valid], values.loc[valid])

        trends["metric"].append(metric)
        trends["slope"].append(regression.slope)
        trends["r_squared"].append(regression.rvalue**2 if values.loc[valid].nunique() > 1 else np.nan)

    session_trends = pd.DataFrame(trends)

    return session_trends, player_name

def build_day_trends(recordings):
    metrics = (
        "average_speed",
        "average_acceleration_magnitude",
        "average_heart_rate",
        "occupied_tiles",
    )
    columns = [
        f"mmn_normalized_{metric}_{stat}"
        for metric in metrics
        for stat in ("slope", "r_squared")
    ]

    rows = []
    for _, team_day in recordings.groupby(["team", "date"], sort=False):
        windows_by_file = {}
        for path in team_day["file_path"]:
            windows, player = calculate_accel_speed_space_dist(path)
            if windows is not None and not windows.empty:
                windows_by_file[path] = windows.assign(player=player)

        if not windows_by_file:
            continue

        distributions = calc_global_dists(
            pd.concat(windows_by_file.values(), ignore_index=True)
        )
        for path in windows_by_file:
            trends, _ = calculate_accel_speed_space_trends((path, distributions))
            if trends is None or trends.empty:
                continue

            row = {"file_path": path}
            for metric, slope, r_squared in trends.itertuples(index=False, name=None):
                if metric.startswith("mmn_normalized_"):
                    row[f"{metric}_slope"] = slope
                    row[f"{metric}_r_squared"] = r_squared
            rows.append(row)

    return pd.DataFrame(rows, columns=["file_path", *columns])

if __name__ == "__main__":
    output_path_session_data = Path("./_0_data/Session_chunk_data.pkl")
    manifest = pd.read_csv("/home/jack/desktop/ASTRID/_0_data/manifest.csv")

    sessions = []
    if output_path_session_data.exists():
        all_sessions = pd.read_pickle(output_path_session_data)
        processed_files = all_sessions["file_path"].unique()
        manifest = manifest.loc[~manifest["file_path"].isin(processed_files)]
        sessions = [all_sessions]
    
    manifest = manifest.sort_values("file_path")

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(calculate_accel_speed_space_dist, file_path): file_path for file_path in manifest["file_path"].drop_duplicates()}

        for future in tqdm(as_completed(futures), total=len(futures)):
            file_path = futures[future]
            df, player_name = future.result()  # Worker exceptions are raised here.

            if df is None or df.empty:
                continue

            df["player"] = player_name
            df["file_path"] = file_path
            sessions.append(df)

            all_sessions = pd.concat(sessions, ignore_index=True)
            temporary_path = output_path_session_data.with_suffix(".tmp")
            all_sessions.to_pickle(temporary_path)
            temporary_path.replace(output_path_session_data)

    distribution_path = Path("./_0_data/distributions.pkl")
    if len(manifest) > 0 or not distribution_path.exists():
        distribution_data = calc_global_dists(all_sessions)
        distribution_data.to_pickle(distribution_path)
    else:
        distribution_data=pd.read_pickle(distribution_path)

    output_path_session_trends = Path("./_0_data/Session_trend_data.pkl")
    manifest = pd.read_csv("/home/jack/desktop/ASTRID/_0_data/manifest.csv")
    session_trends = []
    if output_path_session_trends.exists():
        all_session_trends = pd.read_pickle(output_path_session_trends)
        processed_files = all_session_trends["file_path"].unique()
        manifest = manifest.loc[~manifest["file_path"].isin(processed_files)]
        session_trends = [all_session_trends]

    manifest = manifest.sort_values("file_path")

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(calculate_accel_speed_space_trends, (file_path, distribution_data.copy())): file_path for file_path in manifest["file_path"].drop_duplicates()}

        for future in tqdm(as_completed(futures), total=len(futures)):
            file_path = futures[future]
            df, player_name = future.result()  # Worker exceptions are raised here.

            if df is None or df.empty:
                continue

            df["player"] = player_name
            df["file_path"] = file_path
            session_trends.append(df)

            all_session_trends = pd.concat(session_trends, ignore_index=True)
            temporary_path = output_path_session_trends.with_suffix(".tmp")
            all_session_trends.to_pickle(temporary_path)
            temporary_path.replace(output_path_session_trends)
    