import os
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm

soccer_mon_path = "/home/jack/desktop/ASTRID/soccer_mon"
game_perf       = "/home/jack/desktop/ASTRID/soccer_mon/subjective/game-performance/game-performance.csv"

readiness_data = "/home/jack/desktop/ASTRID/soccer_mon/subjective/wellness/readiness.csv"
fatigue_data   = "/home/jack/desktop/ASTRID/soccer_mon/subjective/wellness/fatigue.csv"
mood_data      = "/home/jack/desktop/ASTRID/soccer_mon/subjective/wellness/mood.csv"
sleep_duration = "/home/jack/desktop/ASTRID/soccer_mon/subjective/wellness/sleep_duration.csv"
sleep_quality  = "/home/jack/desktop/ASTRID/soccer_mon/subjective/wellness/sleep_quality.csv"
soreness       = "/home/jack/desktop/ASTRID/soccer_mon/subjective/wellness/soreness.csv"
stress         = "/home/jack/desktop/ASTRID/soccer_mon/subjective/wellness/stress.csv"

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

print(manifest)

manifest.to_csv(save_path+"manifest.csv", index=False)


