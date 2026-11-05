import os
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from pprint import pprint

soccer_mon_path = "/home/jack/desktop/ASTRID/soccer_mon"
game_perf       = "/home/jack/desktop/ASTRID/soccer_mon/subjective/game-performance/game-performance.csv"

readiness_data = "/home/jack/desktop/ASTRID/soccer_mon/subjective/wellness/readiness.csv"
fatigue_data   = "/home/jack/desktop/ASTRID/soccer_mon/subjective/wellness/fatigue.csv"
mood_data      = "/home/jack/desktop/ASTRID/soccer_mon/subjective/wellness/mood.csv"
sleep_duration = "/home/jack/desktop/ASTRID/soccer_mon/subjective/wellness/sleep_duration.csv"
sleep_quality  = "/home/jack/desktop/ASTRID/soccer_mon/subjective/wellness/sleep_quality.csv"
soreness       = "/home/jack/desktop/ASTRID/soccer_mon/subjective/wellness/soreness.csv"
stress         = "/home/jack/desktop/ASTRID/soccer_mon/subjective/wellness/stress.csv"

load_path = "/home/jack/desktop/ASTRID/soccer_mon/subjective/training-load"
load_files= ["acwr.csv", "atl.csv", "ctl28.csv", "ctl42.csv", "daily_load.csv", "weekly_load.csv", "strain.csv", "monotony.csv"]

session_path = "/home/jack/desktop/ASTRID/soccer_mon/subjective/training-load/session.json"

save_path      = "./_0_data/"



manifest = {
    "file_path" : [],
    "team" : [],
    "year" : [],
    "month" : [],
    "day" : [],
    "player" : [],
}

suffix = ".parquet"
suf_len = len(suffix)

def process_filename(fname):
    title_values = fname.split("-")
    year  = title_values[0]
    month = title_values[1]
    day   = title_values[2]
    team  = title_values[3][-1] # should be a character
    player = "-".join(title_values[3:])
    return year, month, day, team, player

def process_ts(date_dots):
    day, month, year = date_dots.split(".")
    return pd.Series([month, day, year])

def is_game_day(game_perf_df, player_row):
    year_filter  = game_perf_df["year"]         == player_row["year"]
    month_filter = game_perf_df["month"]        == player_row["month"]
    day_filter   = game_perf_df["day"]          == player_row["day"]
    team_filter  = game_perf_df["player_name"].str.split("-").str[0] == "Team" + player_row["team"]
    potential_game = game_perf_df.loc[year_filter & month_filter & day_filter & team_filter]
    if len(potential_game) > 0:
        # Keep all matching reports; several teammates may report the same game.
        return pd.Series([True, potential_game])
    else:
        return pd.Series([False, []])

def wellness_entry(wellness_df, player_row):
    entry_row = wellness_df.iloc[:, 0] == ".".join([player_row["day"], player_row["month"], player_row["year"]])
    if player_row["player"] in wellness_df.columns:
        wellness = wellness_df.loc[entry_row, player_row["player"]]
        assert len(wellness) == 1
        return wellness.iloc[0]
    else:
        return -1

for root, dirs, files in os.walk(soccer_mon_path):
    for file in files:
        if file[-suf_len:] == suffix:
            year, month, day, team, player = process_filename(fname=file[:-suf_len])
            full_path = os.path.join(root, file)
            manifest["file_path"].append(full_path)
            manifest["team"].append(team)
            manifest["year"].append(year)
            manifest["month"].append(month)
            manifest["day"].append(day)
            manifest["player"].append(player)

manifest = pd.DataFrame(manifest)

game_perf_df = pd.read_csv(game_perf)
game_perf_df[["month", "day", "year"]] = game_perf_df["timestamp"].apply(lambda X: process_ts(X))

manifest[["Game_Day", "Game_Stats"]] = manifest.apply(lambda X: is_game_day(game_perf_df, X), axis=1)

readiness_df = pd.read_csv(readiness_data)
fatigue_df = pd.read_csv(fatigue_data)
mood_df = pd.read_csv(mood_data)
sleep_duration_df = pd.read_csv(sleep_duration)
sleep_quality_df = pd.read_csv(sleep_quality)
soreness_df = pd.read_csv(soreness)
stress_df = pd.read_csv(stress)
   

for df_name, wellness_df in zip(
    ["readiness", "fatigue",  "mood",  "sleep_duration",  "sleep_quality",  "soreness",  "stress"],
    [readiness_df, fatigue_df, mood_df, sleep_duration_df, sleep_quality_df, soreness_df, stress_df]
    ):
    manifest[df_name] = manifest.apply(lambda X: wellness_entry(wellness_df=wellness_df, player_row=X), axis=1)

for csv_file in load_files:
    full_path = os.path.join(load_path,csv_file)
    col = csv_file.split(".")[0]
    df = pd.read_csv(full_path)
    manifest[col] = manifest.apply(lambda X: wellness_entry(wellness_df=df, player_row=X), axis=1)

with open(session_path, "r") as f:
    session = json.load(f)

manifest["srpe"]     = None
manifest["rpe"]      = None
manifest["duration"] = None
manifest["session_count"] = None

players = list(session.keys())
for player in players:
    player_filter= manifest["player"]== player
    organized_sessions = defaultdict(list)
    for session_entry in session[player]:
        organized_sessions[session_entry["date"]].append(session_entry)
    for session_date, session_entry_list in organized_sessions.items():
        day, month, year = session_date.split(".")
        session_entries  = len(session_entry_list)
        duration         = sum([session_entry["duration"] for session_entry in session_entry_list])
        srpe             = sum([session_entry["srpe"] for session_entry in session_entry_list])
        rpe              = sum([session_entry["rpe"] for session_entry in session_entry_list]) / session_entries
        year_filter  = manifest["year"]  == year
        month_filter = manifest["month"] == month
        day_filter   = manifest["day"]   == day
        assert len(manifest.loc[player_filter & year_filter & month_filter & day_filter]) <= 1
        manifest.loc[player_filter & year_filter & month_filter & day_filter, "srpe"]          = srpe
        manifest.loc[player_filter & year_filter & month_filter & day_filter, "rpe"]           = rpe
        manifest.loc[player_filter & year_filter & month_filter & day_filter, "duration"]      = duration
        manifest.loc[player_filter & year_filter & month_filter & day_filter, "session_count"] = session_entries

# Calculate features directly from each player's chronological manifest rows.
manifest = manifest.sort_values(["player", "year", "month", "day"], key=lambda values: values if values.name == "player" else pd.to_numeric(values)).reset_index(drop=True)

measurement_columns = ["readiness", "fatigue", "mood", "sleep_duration", "sleep_quality", "soreness", "stress", *[csv_file.removesuffix(".csv") for csv_file in load_files], "srpe", "rpe", "duration", "session_count"]
manifest[measurement_columns] = manifest[measurement_columns].replace(-1, np.nan)

# Windows count manifest rows and include the current row; standard deviations use ddof=1.
# Undefined ratios remain missing; source values are not filled.
manifest["daily_load_roll3_mean"] = manifest.groupby("player")["daily_load"].transform(lambda values: values.rolling(3, min_periods=2).mean())
manifest["daily_load_roll3_sum"] = manifest.groupby("player")["daily_load"].transform(lambda values: values.rolling(3, min_periods=2).sum())
manifest["daily_load_roll3_std"] = manifest.groupby("player")["daily_load"].transform(lambda values: values.rolling(3, min_periods=2).std())
manifest["daily_load_roll3_cv"] = manifest["daily_load_roll3_std"] / manifest["daily_load_roll3_mean"].replace(0, np.nan)
manifest["daily_load_roll7_mean"] = manifest.groupby("player")["daily_load"].transform(lambda values: values.rolling(7, min_periods=3).mean())
manifest["daily_load_roll7_sum"] = manifest.groupby("player")["daily_load"].transform(lambda values: values.rolling(7, min_periods=3).sum())
manifest["daily_load_roll7_std"] = manifest.groupby("player")["daily_load"].transform(lambda values: values.rolling(7, min_periods=3).std())
manifest["daily_load_roll7_cv"] = manifest["daily_load_roll7_std"] / manifest["daily_load_roll7_mean"].replace(0, np.nan)
manifest["daily_load_roll14_mean"] = manifest.groupby("player")["daily_load"].transform(lambda values: values.rolling(14, min_periods=7).mean())
manifest["daily_load_roll14_sum"] = manifest.groupby("player")["daily_load"].transform(lambda values: values.rolling(14, min_periods=7).sum())
manifest["daily_load_roll14_std"] = manifest.groupby("player")["daily_load"].transform(lambda values: values.rolling(14, min_periods=7).std())
manifest["daily_load_roll14_cv"] = manifest["daily_load_roll14_std"] / manifest["daily_load_roll14_mean"].replace(0, np.nan)
manifest["daily_load_absolute_change"] = manifest.groupby("player")["daily_load"].diff().abs()
manifest["daily_load_roll3_mean_absolute_change"] = manifest.groupby("player")["daily_load_absolute_change"].transform(lambda values: values.rolling(3, min_periods=2).mean())
manifest["load_7_28_ratio"] = manifest["daily_load_roll7_mean"] / manifest.groupby("player")["daily_load"].transform(lambda values: values.rolling(28, min_periods=14).mean()).replace(0, np.nan)
manifest["fatigue_roll3_mean"] = manifest.groupby("player")["fatigue"].transform(lambda values: values.rolling(3, min_periods=2).mean())
manifest["fatigue_roll3_std"] = manifest.groupby("player")["fatigue"].transform(lambda values: values.rolling(3, min_periods=2).std())
manifest["fatigue_roll7_mean"] = manifest.groupby("player")["fatigue"].transform(lambda values: values.rolling(7, min_periods=3).mean())
manifest["fatigue_roll7_std"] = manifest.groupby("player")["fatigue"].transform(lambda values: values.rolling(7, min_periods=3).std())
manifest["mood_roll3_mean"] = manifest.groupby("player")["mood"].transform(lambda values: values.rolling(3, min_periods=2).mean())
manifest["mood_roll3_std"] = manifest.groupby("player")["mood"].transform(lambda values: values.rolling(3, min_periods=2).std())
manifest["mood_roll7_mean"] = manifest.groupby("player")["mood"].transform(lambda values: values.rolling(7, min_periods=3).mean())
manifest["mood_roll7_std"] = manifest.groupby("player")["mood"].transform(lambda values: values.rolling(7, min_periods=3).std())
manifest["sleep_duration_roll3_mean"] = manifest.groupby("player")["sleep_duration"].transform(lambda values: values.rolling(3, min_periods=2).mean())
manifest["sleep_duration_roll3_std"] = manifest.groupby("player")["sleep_duration"].transform(lambda values: values.rolling(3, min_periods=2).std())
manifest["sleep_duration_roll7_mean"] = manifest.groupby("player")["sleep_duration"].transform(lambda values: values.rolling(7, min_periods=3).mean())
manifest["sleep_duration_roll7_std"] = manifest.groupby("player")["sleep_duration"].transform(lambda values: values.rolling(7, min_periods=3).std())
manifest["sleep_quality_roll3_mean"] = manifest.groupby("player")["sleep_quality"].transform(lambda values: values.rolling(3, min_periods=2).mean())
manifest["sleep_quality_roll3_std"] = manifest.groupby("player")["sleep_quality"].transform(lambda values: values.rolling(3, min_periods=2).std())
manifest["sleep_quality_roll7_mean"] = manifest.groupby("player")["sleep_quality"].transform(lambda values: values.rolling(7, min_periods=3).mean())
manifest["sleep_quality_roll7_std"] = manifest.groupby("player")["sleep_quality"].transform(lambda values: values.rolling(7, min_periods=3).std())
manifest["soreness_roll3_mean"] = manifest.groupby("player")["soreness"].transform(lambda values: values.rolling(3, min_periods=2).mean())
manifest["soreness_roll3_std"] = manifest.groupby("player")["soreness"].transform(lambda values: values.rolling(3, min_periods=2).std())
manifest["soreness_roll7_mean"] = manifest.groupby("player")["soreness"].transform(lambda values: values.rolling(7, min_periods=3).mean())
manifest["soreness_roll7_std"] = manifest.groupby("player")["soreness"].transform(lambda values: values.rolling(7, min_periods=3).std())
manifest["stress_roll3_mean"] = manifest.groupby("player")["stress"].transform(lambda values: values.rolling(3, min_periods=2).mean())
manifest["stress_roll3_std"] = manifest.groupby("player")["stress"].transform(lambda values: values.rolling(3, min_periods=2).std())
manifest["stress_roll7_mean"] = manifest.groupby("player")["stress"].transform(lambda values: values.rolling(7, min_periods=3).mean())
manifest["stress_roll7_std"] = manifest.groupby("player")["stress"].transform(lambda values: values.rolling(7, min_periods=3).std())

manifest.dropna(subset=["readiness"], inplace=True)

print(manifest)

manifest.to_csv(save_path+"manifest.csv", index=False)