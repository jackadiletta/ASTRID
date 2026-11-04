import base64
from datetime import date, timedelta
from io import BytesIO
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from app import (app, manifest, trajectory, COLUMNS, FIELDS, next_game_session,
                 acceleration_heatmap, heatmap_png, game_statistics, session_date,
                 prior_game_session, spatial_coordinates, spatial_bins, EARTH_RADIUS_M, forward_fill_gps, shared_game_scale)
from PIL import Image

class ViewerTests(unittest.TestCase):
    def test_trajectory_coordinates_and_notebook_acceleration(self):
        frame = pd.read_parquet(manifest()[0]["file_path"], columns=COLUMNS)
        actual = trajectory(frame)
        coordinates = frame[["lat", "lon"]].replace(0, np.nan).ffill()
        norm = lambda values: (values - values.min()) / (values.max() - values.min())
        expected_x = EARTH_RADIUS_M * np.cos(np.deg2rad(coordinates.lat.median())) * np.deg2rad(coordinates.lon.to_numpy() - coordinates.lon.median())
        expected_y = EARTH_RADIUS_M * np.deg2rad(coordinates.lat.to_numpy() - coordinates.lat.median())
        np.testing.assert_allclose(actual["x"], expected_x[499:-500])
        np.testing.assert_allclose(actual["y"], expected_y[499:-500])
        np.testing.assert_allclose(actual["z"], norm(frame.index.to_numpy(dtype=float))[499:-500])
        np.testing.assert_array_equal(actual["latitude"], coordinates.lat.to_numpy()[499:-500])
        np.testing.assert_array_equal(actual["longitude"], coordinates.lon.to_numpy()[499:-500])
        acceleration = np.sqrt((frame[["accl_x", "accl_y", "accl_z"]].to_numpy() ** 2).sum(axis=1))
        expected = norm(sliding_window_view(acceleration, 1000).mean(axis=1))
        np.testing.assert_allclose(actual["color"], expected, rtol=1e-7, atol=1e-8)
        self.assertEqual(actual["hover"][0], str(frame.index[499]))

    def test_endpoints(self):
        with app.test_client() as client:
            with client.get("/") as response:
                self.assertEqual(response.status_code, 200)
            with client.get("/assets/viewer.js") as response:
                self.assertEqual(response.status_code, 200)
            self.assertEqual(client.get("/assets/plotly.js").status_code, 200)
            rows = client.get("/api/manifest").get_json()
            self.assertGreater(len(rows), 0)
            response = client.get("/api/session", query_string=rows[0])
            self.assertEqual(response.status_code, 200, response.get_json())
            self.assertEqual(len(response.json["x"]), response.json["rows"] - 999)
            self.assertEqual(response.json["is_game_day"], manifest()[0]["is_game_day"])
            self.assertEqual(client.get("/api/session").status_code, 400)
            self.assertEqual(client.get("/api/session", query_string={**rows[0], "player":"missing"}).status_code, 404)
            with self.assertLogs(app.logger, level="ERROR"), patch("app.pd.read_parquet", side_effect=FileNotFoundError("explicit missing file")):
                response = client.get("/api/session", query_string=rows[0])
                self.assertEqual(response.status_code, 500)
                self.assertIn("explicit missing file", response.json["error"])

    def test_invalid_data_fails(self):
        with self.assertRaisesRegex(ValueError, "at least"):
            trajectory(pd.DataFrame(np.ones((10, 5)), columns=COLUMNS))
        with self.assertRaisesRegex(ValueError, "constant"):
            trajectory(pd.DataFrame(np.ones((1100, 5)), columns=COLUMNS))
        frame = pd.DataFrame(np.arange(5500).reshape(1100,5).astype(float), columns=COLUMNS)
        frame["lat"] = np.linspace(63.44, 63.45, len(frame))
        frame["lon"] = np.linspace(10.45, 10.46, len(frame))
        frame.loc[2,"lat"] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            trajectory(frame)


    def test_next_game_boundaries(self):
        def row(day, player="selected", team="A", game=True):
            return dict(year=str(day.year), month=f"{day.month:02}", day=f"{day.day:02}",
                        team=team, player=player, is_game_day=game)
        day = date(2020, 12, 25)
        selected = row(day)
        at_limit = row(day + timedelta(days=14))
        earlier = row(day + timedelta(days=2))
        unrelated = [selected, row(day - timedelta(days=1)), row(day + timedelta(days=1), player="other"),
                     row(day + timedelta(days=1), team="B"), row(day + timedelta(days=1), game=False),
                     row(day + timedelta(days=15))]
        self.assertIsNone(next_game_session(selected, unrelated))
        self.assertEqual(next_game_session(selected, unrelated + [at_limit]), at_limit)
        self.assertEqual(next_game_session(selected, unrelated + [at_limit, earlier]), earlier)
        leap_day = row(date(2020, 2, 29))
        self.assertEqual(next_game_session(row(date(2020, 2, 28)), [leap_day]), leap_day)

    def test_heatmap_uses_spatial_extent_and_notebook_acceleration(self):
        row = next(row for row in manifest() if row["is_game_day"])
        frame = pd.read_parquet(row["file_path"], columns=COLUMNS)
        coordinates = frame[["lat", "lon"]].replace(0, np.nan).ffill()
        norm = lambda values: (values - values.min()) / (values.max() - values.min())
        x = (EARTH_RADIUS_M * np.cos(np.deg2rad(coordinates.lat.median())) * np.deg2rad(coordinates.lon.to_numpy() - coordinates.lon.median()))[499:-500]
        y = (EARTH_RADIUS_M * np.deg2rad(coordinates.lat.to_numpy() - coordinates.lat.median()))[499:-500]
        acceleration = np.sqrt((frame[["accl_x", "accl_y", "accl_z"]].to_numpy() ** 2).sum(axis=1))
        colors = norm(sliding_window_view(acceleration, 1000).mean(axis=1))
        bins = [np.linspace(x.min(), x.max(), 51), np.linspace(y.min(), y.max(), 51)]
        sums, _, _ = np.histogram2d(x, y, bins=bins, weights=colors)
        counts, _, _ = np.histogram2d(x, y, bins=bins)
        expected = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        actual_track = trajectory(frame)
        np.testing.assert_allclose(acceleration_heatmap(actual_track), expected, rtol=1e-7, atol=1e-8)
        encoded = heatmap_png(actual_track).split(",", 1)[1]
        with Image.open(BytesIO(base64.b64decode(encoded))) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (600, 500))

    def test_next_game_endpoint_with_real_data(self):
        rows = manifest()
        selected = next(row for row in rows if next_game_session(row, rows) is not None)
        expected = next_game_session(selected, rows)
        with app.test_client() as client:
            response = client.get("/api/next-game", query_string={key:selected[key] for key in FIELDS})
            self.assertEqual(response.status_code, 200, response.get_json())
            game = response.json["game"]
            self.assertEqual(game["date"], session_date(expected).isoformat())
            self.assertTrue(1 <= game["days_after"] <= 14)
            self.assertTrue(game["image"].startswith("data:image/png;base64,"))
            self.assertEqual(game["statistics"], game_statistics(expected))
            self.assertTrue(game["statistics"])
            self.assertTrue(all(report["player_name"].startswith("Team"+selected["team"]+"-") for report in game["statistics"]))
            missing = max(rows, key=session_date)
            with patch("app.load_trajectory") as loader:
                response = client.get("/api/next-game", query_string={key:missing[key] for key in FIELDS})
                self.assertEqual(response.status_code, 200)
                self.assertIsNone(response.json["game"])
                self.assertEqual(response.json["message"], "No game within 2 weeks.")
                loader.assert_not_called()
            self.assertEqual(client.get("/api/next-game").status_code, 400)

    def test_no_statistics_and_game_errors(self):
        rows = manifest()
        selected = next(row for row in rows if next_game_session(row, rows) is not None)
        params = {key:selected[key] for key in FIELDS}
        with app.test_client() as client:
            with patch("app.game_statistics", return_value=[]):
                response = client.get("/api/next-game", query_string=params)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json["game"]["statistics"], [])
            with self.assertLogs(app.logger, level="ERROR"), patch("app.load_trajectory", side_effect=ValueError("bad game data")):
                response = client.get("/api/next-game", query_string=params)
                self.assertEqual(response.status_code, 500)
                self.assertIn("bad game data", response.json["error"])


    def test_rectangular_geometry(self):
        # Construct a real 105 x 68 meter rectangle near the dataset's latitude.
        lat0, lon0 = 63.44, 10.45
        latitude = lat0 + np.rad2deg(np.array([-34, -34, 34, 34]) / EARTH_RADIUS_M)
        longitude = lon0 + np.rad2deg(np.array([-52.5, 52.5, 52.5, -52.5]) / (EARTH_RADIUS_M * np.cos(np.deg2rad(lat0))))
        east, north, origin = spatial_coordinates(latitude, longitude)
        self.assertAlmostEqual(np.ptp(east), 105, places=6)
        self.assertAlmostEqual(np.ptp(north), 68, places=6)
        self.assertAlmostEqual(np.ptp(east) / np.ptp(north), 105 / 68, places=6)
        bins = spatial_bins({"x":east, "y":north})
        self.assertAlmostEqual(bins[0][-1]-bins[0][0], 105, places=6)
        self.assertAlmostEqual(bins[1][-1]-bins[1][0], 68, places=6)
        self.assertEqual(origin, {"latitude":lat0, "longitude":lon0})
        # Zero GPS readings remain in the coordinates rather than being filtered.
        east, north, _ = spatial_coordinates([0, lat0, lat0], [0, lon0, lon0])
        self.assertGreater(abs(north[0]), 1_000_000)
        with self.assertRaisesRegex(ValueError, "outside valid"):
            spatial_coordinates([91], [10])
        with self.assertRaisesRegex(ValueError, "nonzero spatial extent"):
            spatial_bins({"x":[1,1], "y":[0,1]})

    def test_prior_game_boundaries(self):
        def row(day, player="selected", team="A", game=True):
            return dict(year=str(day.year), month=f"{day.month:02}", day=f"{day.day:02}",
                        team=team, player=player, is_game_day=game)
        day = date(2021, 1, 5)
        selected = row(day)
        at_limit = row(day - timedelta(days=14))
        nearer = row(day - timedelta(days=2))
        unrelated = [selected, row(day + timedelta(days=1)), row(day - timedelta(days=1), player="other"),
                     row(day - timedelta(days=1), team="B"), row(day - timedelta(days=1), game=False),
                     row(day - timedelta(days=15))]
        self.assertIsNone(prior_game_session(selected, unrelated))
        self.assertEqual(prior_game_session(selected, unrelated + [at_limit]), at_limit)
        self.assertEqual(prior_game_session(selected, unrelated + [at_limit, nearer]), nearer)
        leap_day = row(date(2020, 2, 29))
        self.assertEqual(prior_game_session(row(date(2020, 3, 1)), [leap_day]), leap_day)

    def test_prior_game_endpoint_with_real_data(self):
        rows = manifest()
        selected = next(row for row in rows if prior_game_session(row, rows) is not None)
        expected = prior_game_session(selected, rows)
        with app.test_client() as client:
            response = client.get("/api/prior-game", query_string={key:selected[key] for key in FIELDS})
            self.assertEqual(response.status_code, 200, response.get_json())
            game = response.json["game"]
            self.assertEqual(game["date"], session_date(expected).isoformat())
            self.assertTrue(1 <= game["days_before"] <= 14)
            self.assertTrue(game["image"].startswith("data:image/png;base64,"))
            self.assertEqual(game["statistics"], game_statistics(expected))
            missing = min(rows, key=session_date)
            with patch("app.load_trajectory") as loader:
                response = client.get("/api/prior-game", query_string={key:missing[key] for key in FIELDS})
                self.assertEqual(response.status_code, 200)
                self.assertIsNone(response.json["game"])
                self.assertEqual(response.json["message"], "No prior game within 2 weeks.")
                loader.assert_not_called()
            self.assertEqual(client.get("/api/prior-game").status_code, 400)
            with self.assertLogs(app.logger, level="ERROR"), patch("app.load_trajectory", side_effect=ValueError("bad prior game data")):
                response = client.get("/api/prior-game", query_string={key:selected[key] for key in FIELDS})
                self.assertEqual(response.status_code, 500)
                self.assertIn("bad prior game data", response.json["error"])


    def test_forward_fill_zero_and_missing_gps(self):
        frame = pd.DataFrame({"lat":[63.4, 0, np.nan, 63.5, 0],
                              "lon":[10.4, 10.5, 0, np.nan, 10.6]}, index=[1,2,3,4,5])
        original = frame.copy()
        result = forward_fill_gps(frame)
        np.testing.assert_array_equal(result.lat, [63.4, 63.4, 63.4, 63.5, 63.5])
        np.testing.assert_array_equal(result.lon, [10.4, 10.5, 10.5, 10.5, 10.6])
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(result.index.tolist(), frame.index.tolist())
        for value in (0, np.nan):
            with self.assertRaisesRegex(ValueError, "no previous value"):
                forward_fill_gps(pd.DataFrame({"lat":[value,63.4], "lon":[10.4,10.5]}))
        with self.assertRaisesRegex(ValueError, "no previous value"):
            forward_fill_gps(pd.DataFrame({"lat":[0,0], "lon":[10.4,10.5]}))

    def test_august_17_longitude_spike_is_forward_filled(self):
        row = next(row for row in manifest() if row["player"] == "TeamA-5a3be598-2f51-426f-95a5-b63d1d7c1b43"
                   and session_date(row) == date(2020,8,17))
        frame = pd.read_parquet(row["file_path"], columns=COLUMNS)
        zeros = np.flatnonzero(frame.lon.eq(0))
        self.assertEqual(len(zeros),10)
        track = trajectory(frame)
        self.assertLess(max(track["x"])-min(track["x"]), 200)
        for index in zeros:
            self.assertEqual(track["longitude"][index-499], frame.lon.iloc[zeros[0]-1])
        self.assertTrue(frame.lon.iloc[zeros].eq(0).all())


    def test_shared_game_acceleration_scale(self):
        tracks = {
            "prior": {"acceleration_min":1.0, "acceleration_max":3.0, "color":[0.0,0.5,1.0]},
            "next": {"acceleration_min":2.0, "acceleration_max":6.0, "color":[0.0,0.5,1.0]},
        }
        scale = shared_game_scale(tracks)
        self.assertEqual(scale, {"min":1.0,"max":6.0})
        np.testing.assert_allclose(tracks["prior"]["color"], [0,0.2,0.4])
        np.testing.assert_allclose(tracks["next"]["color"], [0.2,0.6,1])
        # Acceleration 2 has the same normalized color in both sessions.
        self.assertEqual(tracks["prior"]["color"][1], tracks["next"]["color"][0])
        only = {"prior":{"acceleration_min":2.0,"acceleration_max":6.0,"color":[0.0,0.5,1.0]}}
        self.assertEqual(shared_game_scale(only), {"min":2.0,"max":6.0})
        np.testing.assert_array_equal(only["prior"]["color"], [0,0.5,1])
        self.assertIsNone(shared_game_scale({}))
        with self.assertRaisesRegex(ValueError, "nonconstant"):
            shared_game_scale({"prior":{"acceleration_min":2,"acceleration_max":2,"color":[0]}})

    def test_paired_games_endpoint_with_shared_scale(self):
        selected = next(row for row in manifest() if row["player"] == "TeamA-5a3be598-2f51-426f-95a5-b63d1d7c1b43"
                        and session_date(row) == date(2020,8,17))
        with app.test_client() as client:
            response = client.get("/api/games",query_string={key:selected[key] for key in FIELDS})
            self.assertEqual(response.status_code,200,response.json)
            self.assertEqual(response.json["prior"]["game"]["date"], "2020-08-16")
            self.assertEqual(response.json["next"]["game"]["date"], "2020-08-22")
            scale = response.json["scale"]
            self.assertLess(scale["min"],scale["max"])
            for direction in ("prior","next"):
                game = response.json[direction]["game"]
                self.assertTrue(game["statistics"])
                with Image.open(BytesIO(base64.b64decode(game["image"].split(",",1)[1]))) as image:
                    self.assertEqual(image.size,(600,300))
            with Image.open(BytesIO(base64.b64decode(scale["image"].split(",",1)[1]))) as image:
                self.assertEqual(image.size,(600,65))
            self.assertEqual(client.get("/api/games").status_code,400)
            with patch("app.prior_game_session",return_value=None), patch("app.next_game_session",return_value=None), patch("app.load_trajectory") as loader:
                response = client.get("/api/games",query_string={key:selected[key] for key in FIELDS})
                self.assertEqual(response.status_code,200)
                self.assertIsNone(response.json["scale"])
                self.assertIsNone(response.json["prior"]["game"])
                self.assertIsNone(response.json["next"]["game"])
                loader.assert_not_called()
            with patch("app.prior_game_session",return_value=None):
                response = client.get("/api/games",query_string={key:selected[key] for key in FIELDS})
                self.assertEqual(response.status_code,200,response.json)
                self.assertIsNone(response.json["prior"]["game"])
                self.assertIsNotNone(response.json["next"]["game"])
                self.assertIsNotNone(response.json["scale"])
            with self.assertLogs(app.logger,level="ERROR"), patch("app.load_trajectory",side_effect=ValueError("bad game data")):
                response = client.get("/api/games",query_string={key:selected[key] for key in FIELDS})
                self.assertEqual(response.status_code,500)
                self.assertIn("bad game data",response.json["error"])

if __name__ == "__main__":
    unittest.main()
