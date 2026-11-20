import numpy as np
import pandas as pd

from scipy.special import logit
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

def classifier_models():
    return {
        "LOG": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=1.0,
                solver="lbfgs",
                fit_intercept=True,
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            )),
        ]),
        "RF": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=250,
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=10,
                max_features="sqrt",
                bootstrap=True,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        # Your extension: these parameters are not from Long et al.
        "XGB": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=250,
                max_depth=3,
                learning_rate=0.05,
                tree_method="hist",
                random_state=42,
                n_jobs=8,
            )),
        ]),
    }

def regressor_model():
    # Your extension: continuous-outcome regression.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            tree_method="hist",
            random_state=42,
            n_jobs=8,
        )),
    ])


def fit_classifier(estimator, train_x, train_y, train_players, test_x, test_y):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probabilities = np.full(len(train_y), np.nan)

    for fit_indices, validation_indices in splitter.split(train_x, train_y, groups=train_players):
        fold_x = train_x.iloc[fit_indices]
        fold_y = train_y.iloc[fit_indices]

        assert fold_y.nunique() == 2, "Training fold lacks both classes"
        assert not fold_x.isna().all().any(), "Training fold contains an entirely missing feature"

        fold_model = clone(estimator)
        fold_model.fit(fold_x, fold_y)
        oof_probabilities[validation_indices] = fold_model.predict_proba(train_x.iloc[validation_indices])[:, 1]

    assert np.isfinite(oof_probabilities).all(), "Invalid OOF predictions"

    # Balanced accuracy = (sensitivity + specificity) / 2.
    false_positive_rate, true_positive_rate, thresholds = roc_curve(train_y, oof_probabilities, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    balanced_scores = (true_positive_rate[finite] + 1 - false_positive_rate[finite]) / 2
    threshold = float(thresholds[finite][np.argmax(balanced_scores)])

    # Platt calibration: unweighted logistic regression on probability logits.
    oof_logits = logit(np.clip(oof_probabilities, 1e-15, 1 - 1e-15)).reshape(-1, 1)
    calibrator = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000)
    calibrator.fit(oof_logits, train_y)

    model = clone(estimator)
    model.fit(train_x, train_y)

    probabilities = model.predict_proba(test_x)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    test_logits = logit(np.clip(probabilities, 1e-15, 1 - 1e-15)).reshape(-1, 1)
    calibrated_probabilities = calibrator.predict_proba(test_logits)[:, 1]

    tn, fp, fn, tp = confusion_matrix(test_y, predictions, labels=[0, 1]).ravel()

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(test_y, predictions),
        "f1": f1_score(test_y, predictions),
        "auc": roc_auc_score(test_y, probabilities),
        "pr_auc": average_precision_score(test_y, probabilities),
        "balanced_accuracy": balanced_accuracy_score(test_y, predictions),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "predicted_positive_rate": predictions.mean(),
        "brier": brier_score_loss(test_y, probabilities),
        "calibrated_brier": brier_score_loss(test_y, calibrated_probabilities),
    }

    fitted = {"model": model, "calibrator": calibrator, "threshold": threshold}
    return fitted, metrics


def fit_regressor(estimator, train_x, train_y, train_players, test_x, test_y):
    splitter = GroupKFold(n_splits=5)
    oof_predictions = np.full(len(train_y), np.nan)

    for fit_indices, validation_indices in splitter.split(train_x, train_y, groups=train_players):
        fold_x = train_x.iloc[fit_indices]
        fold_y = train_y.iloc[fit_indices]

        assert not fold_x.isna().all().any(), "Training fold contains an entirely missing feature"

        fold_model = clone(estimator)
        fold_model.fit(fold_x, fold_y)
        oof_predictions[validation_indices] = fold_model.predict(train_x.iloc[validation_indices])

    assert np.isfinite(oof_predictions).all(), "Invalid OOF predictions"

    oof_mse = mean_squared_error(train_y, oof_predictions)

    model = clone(estimator)
    model.fit(train_x, train_y)

    predictions = model.predict(test_x)
    assert np.isfinite(predictions).all(), "Invalid test predictions"

    metrics = {
        "oof_mse": oof_mse,
        "mse": mean_squared_error(test_y, predictions),
    }

    fitted = {"model": model}
    return fitted, metrics


def run_experiment(data, feature_sets, classification_target, regression_target):
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")

    assert not data.duplicated(["player", "date"]).any(), "Duplicate player-day rows"

    results = []
    fitted_models = {}

    tasks = {
        "classification": {
            "target": classification_target,
            "estimators": classifier_models(),
            "fit_function": fit_classifier,
        },
        "regression": {
            "target": regression_target,
            "estimators": {"XGB": regressor_model()},
            "fit_function": fit_regressor,
        },
    }

    for task_name, task in tasks.items():
        target = task["target"]
        eligible = data.dropna(subset=["player", "date", target])

        train = eligible.loc[eligible["date"].dt.year.eq(2020)].reset_index(drop=True)
        test = eligible.loc[eligible["date"].dt.year.eq(2021)].reset_index(drop=True)

        assert not train.empty, f"No training rows for {task_name}"
        assert not test.empty, f"No test rows for {task_name}"
        assert train["player"].nunique() >= 5, "Five-fold validation requires at least five athletes"
        assert np.isfinite(train[target]).all(), f"Invalid training target: {target}"
        assert np.isfinite(test[target]).all(), f"Invalid test target: {target}"

        if task_name == "classification":
            assert set(train[target].unique()) == {0, 1}, "Training requires both binary classes"
            assert set(test[target].unique()) == {0, 1}, "Testing requires both binary classes"

        for feature_name, features in feature_sets.items():
            assert len(features) > 0, f"No features in {feature_name}"
            assert len(features) == len(set(features)), f"Duplicate features in {feature_name}"
            assert classification_target not in features, "Classification target included as a feature"
            assert regression_target not in features, "Regression target included as a feature"

            train_x = train[features]
            test_x = test[features]
            train_y = train[target]
            test_y = test[target]
            train_players = train["player"]

            assert not train_x.isna().all().any(), f"Entirely missing training feature in {feature_name}"
            assert not np.isinf(train_x.to_numpy(dtype=float)).any(), "Infinite training predictors"
            assert not np.isinf(test_x.to_numpy(dtype=float)).any(), "Infinite test predictors"

            for model_name, estimator in task["estimators"].items():
                print(f"{task_name}: {feature_name}, {model_name}")

                fitted, metrics = task["fit_function"](
                    estimator,
                    train_x,
                    train_y,
                    train_players,
                    test_x,
                    test_y,
                )

                fitted["features"] = list(features)
                fitted_models[(task_name, feature_name, model_name)] = fitted

                results.append({
                    "task": task_name,
                    "feature_set": feature_name,
                    "model": model_name,
                    "n_features": len(features),
                    "n_train": len(train),
                    "n_test": len(test),
                    **metrics,
                })

    return pd.DataFrame(results), fitted_models


feature_sets = {
    "Long55": [
        "acwr", "atl", "ctl28", "ctl42", "daily_load",
        "daily_load_absolute_change", "daily_load_roll14_cv", "daily_load_roll14_mean", "daily_load_roll14_std", "daily_load_roll14_sum",
        "daily_load_roll3_cv", "daily_load_roll3_mean", "daily_load_roll3_mean_absolute_change", "daily_load_roll3_std", "daily_load_roll3_sum",
        "daily_load_roll7_cv", "daily_load_roll7_mean", "daily_load_roll7_std", "daily_load_roll7_sum", "fatigue",
        "fatigue_roll3_mean", "fatigue_roll3_std", "fatigue_roll7_mean", "fatigue_roll7_std", "load_7_28_ratio",
        "monotony", "mood", "mood_roll3_mean", "mood_roll3_std", "mood_roll7_mean",
        "mood_roll7_std", "readiness", "session_srpe", "sleep_duration", "sleep_duration_roll3_mean",
        "sleep_duration_roll3_std", "sleep_duration_roll7_mean", "sleep_duration_roll7_std", "sleep_quality", "sleep_quality_roll3_mean",
        "sleep_quality_roll3_std", "sleep_quality_roll7_mean", "sleep_quality_roll7_std", "soreness", "soreness_roll3_mean",
        "soreness_roll3_std", "soreness_roll7_mean", "soreness_roll7_std", "strain", "stress",
        "stress_roll3_mean", "stress_roll3_std", "stress_roll7_mean", "stress_roll7_std", "weekly_load",
    ],
}


def regression_models():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    # These continuous-outcome models extend Long et al.'s classification study.
    return {
        "LIN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "RF": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=250, max_depth=5, min_samples_split=2, min_samples_leaf=10, max_features="sqrt", bootstrap=True, random_state=42, n_jobs=-1)),
        ]),
        "XGB": regressor_model(),
    }


def load_experiment_data(root):
    manifest = pd.read_csv(root / "_0_data" / "manifest.csv").rename(columns={"srpe": "session_srpe"})
    manifest["date"] = pd.to_datetime(manifest[["year", "month", "day"]], errors="raise")
    trends = pd.read_pickle(root / "_0_data" / "Session_trend_data.pkl")
    expected_metrics = [prefix + metric for prefix in ["mmn_normalized_", "IQR_normalized_"] for metric in ["average_speed", "average_acceleration_magnitude", "average_heart_rate", "occupied_tiles"]]
    assert len(feature_sets["Long55"]) == 55
    assert set(trends["metric"]) == set(expected_metrics), "Unexpected or missing trend metrics"
    assert not manifest.duplicated(["player", "date"]).any(), "Duplicate manifest player-day rows"
    assert not trends.duplicated(["file_path", "metric"]).any(), "Duplicate trend file/metric rows"
    assert trends.groupby("file_path")["player"].nunique().eq(1).all(), "Multiple players assigned to a trend file"
    assert trends.groupby("file_path")["metric"].nunique().eq(8).all(), "Each file must have eight trend rows"
    # trend_players = trends[["file_path", "player"]].drop_duplicates()
    # player_check = manifest[["file_path", "player"]].merge(trend_players, on="file_path", how="left", validate="one_to_one", suffixes=("_manifest", "_trend"))
    # assert player_check["player_manifest"].eq(player_check["player_trend"]).all(), "Manifest and trend player assignments disagree"
    # Player, team, dates, and labels come exclusively from the manifest.
    trend_table = trends.pivot(index="file_path", columns="metric", values=["slope", "r_squared"])
    trend_table.columns = [f"{metric}_{statistic}" for statistic, metric in trend_table.columns]
    trend_features = [f"{metric}_{statistic}" for metric in expected_metrics for statistic in ["slope", "r_squared"]]
    assert manifest["file_path"].is_unique, "Duplicate manifest file paths"
    dataset = manifest.merge(trend_table.reset_index(), on="file_path", how="left", validate="one_to_one", indicator=True)
    assert dataset["_merge"].eq("both").all(), "Manifest files lack matching trends; finish trend generation before training"
    dataset = dataset.drop(columns="_merge").sort_values(["player", "date"]).reset_index(drop=True)
    dataset[trend_features] = dataset[trend_features].apply(pd.to_numeric, errors="raise")
    dataset = dataset.rename(columns={col: f"{col}_mean_1d" for col in trend_features})
    # Compute rolling averages separately for each player.
    session_trend_cols = [f"{col}_mean_1d" for col in trend_features]
    by_player = dataset.set_index("date").groupby("player")[session_trend_cols]
    rolling_features = []
    for days, min_obs in ((3, 2), (7, 3), (14, 7)):
        averages = by_player.rolling(f"{days}D", min_periods=min_obs, closed="right").mean()
        new_cols = {col: col.replace("_mean_1d", f"_mean_{days}d") for col in session_trend_cols}
        averages = averages.rename(columns=new_cols)
        dataset = dataset.join(averages, on=["player", "date"], validate="one_to_one")
        rolling_features.extend(new_cols.values())
    session_features = [f"{col}_mean_1d" for col in trend_features]
    designs = {
        "Long55": list(feature_sets["Long55"]),
        "Long55 + Trends": list(feature_sets["Long55"]) + session_features,
        "Trends": session_features,
        "Long55 + Rolling Trends": list(feature_sets["Long55"]) + session_features + rolling_features,
        "Rolling Trends": session_features + rolling_features,
    }
    for name, features in designs.items():
        assert len(features) == len(set(features)), f"Duplicate features in {name}"
        dataset[features] = dataset[features].apply(pd.to_numeric, errors="raise")
        assert not np.isinf(dataset[features].to_numpy(dtype=float)).any(), f"Infinite predictors in {name}"
    for target in ["readiness_decline", "next_day_readiness"]:
        dataset[target] = pd.to_numeric(dataset[target], errors="raise")
        assert not np.isinf(dataset[target].to_numpy(dtype=float, na_value=np.nan)).any(), f"Infinite target: {target}"
    return dataset, designs


def run_saved_experiments(dataset, designs, output_directory):
    import joblib

    tasks = {
        "classifier": ("readiness_decline", classifier_models(), fit_classifier),
        "regressor": ("next_day_readiness", regression_models(), fit_regressor),
    }
    splits = {}
    # Validate all splits/features before fitting or creating artifacts.
    for mode, (target, estimators, fit_function) in tasks.items():
        eligible = dataset.dropna(subset=["player", "date", target])
        train = eligible.loc[eligible["date"].dt.year.eq(2020)].reset_index(drop=True)
        test = eligible.loc[eligible["date"].dt.year.eq(2021)].reset_index(drop=True)
        assert not train.empty, f"No eligible 2020 training rows for {mode}"
        assert not test.empty, f"No eligible 2021 test rows for {mode}"
        assert train["player"].nunique() >= 5, f"Fewer than five training athletes for {mode}"
        if mode == "classifier":
            assert set(train[target].unique()) == {0, 1}, "Training requires both binary classes"
            assert set(test[target].unique()) == {0, 1}, "Testing requires both binary classes"
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        else:
            splitter = GroupKFold(n_splits=5)
        for name, features in designs.items():
            assert not train[features].isna().all().any(), f"Entirely missing training feature in {name}"
            for fit_indices, validation_indices in splitter.split(train[features], train[target], groups=train["player"]):
                fold = train.iloc[fit_indices]
                assert not fold[features].isna().all().any(), f"Entirely missing fold features in {mode}/{name}: {fold[features].columns[fold[features].isna().all()].tolist()}"
                if mode == "classifier":
                    assert fold[target].nunique() == 2, "Training fold lacks both classes"
        splits[mode] = (train, test)

    output_directory.mkdir(parents=True, exist_ok=True)
    columns = ["Model", "Feature Set", "Mode", "AUC", "F1", "Accuracy", "MSE"]
    results = []
    predictions = []
    results_path = output_directory / "experiment_results.csv"
    for mode, (target, estimators, fit_function) in tasks.items():
        train, test = splits[mode]
        for name, features in designs.items():
            for model_name, estimator in estimators.items():
                print(f"Running {mode} / {name} / {model_name}: {len(features)} features, {len(train)} train rows, {len(test)} test rows", flush=True)
                fitted, metrics = fit_function(estimator, train[features], train[target], train["player"], test[features], test[target])
                model = fitted["model"]
                if mode == "classifier":
                    predicted = model.predict_proba(test[features])[:, 1]
                else:
                    predicted = model.predict(test[features])
                predictions.append(pd.DataFrame({
                    "player": test["player"].to_numpy(),
                    "date": test["date"].to_numpy(),
                    "model": model_name,
                    "feature_set": name,
                    "actual": test[target].to_numpy(),
                    "predicted": predicted,
                    "mode": mode,
                }))
                # Include preprocessing, feature order, and classifier threshold/calibration.
                fitted.update({"features": list(features), "target": target, "mode": mode, "feature_set": name, "train_year": 2020, "test_year": 2021, "metrics": metrics})
                design_slug = name.replace(" + ", "_plus_")
                model_path = output_directory / f"{mode}_{design_slug}_{model_name}.joblib"
                temporary_model_path = model_path.with_suffix(".tmp")
                joblib.dump(fitted, temporary_model_path)
                temporary_model_path.replace(model_path)
                results.append({
                    "Model": model_name,
                    "Feature Set": name,
                    "Mode": mode,
                    "AUC": metrics["auc"] if mode == "classifier" else np.nan,
                    "F1": metrics["f1"] if mode == "classifier" else np.nan,
                    "Accuracy": metrics["accuracy"] if mode == "classifier" else np.nan,
                    "MSE": metrics["mse"] if mode == "regressor" else np.nan,
                })
                result_table = pd.DataFrame(results, columns=columns)
                temporary_results_path = results_path.with_suffix(".tmp")
                result_table.to_csv(temporary_results_path, index=False)
                temporary_results_path.replace(results_path)
                print(result_table.tail(1).to_string(index=False), flush=True)
    expected_count = len(designs) * sum(len(estimators) for _, estimators, _ in tasks.values())
    assert len(results) == expected_count, f"Expected {expected_count} experiments, got {len(results)}"
    print("\nAll experiment results:", flush=True)
    print(result_table.to_string(index=False), flush=True)
    print(f"Models and results saved in {output_directory}", flush=True)
    prediction_table = pd.concat(predictions, ignore_index=True)
    for mode, filename in [("classifier", "classifier_predictions.csv"), ("regressor", "regressor_predictions.csv")]:
        prediction_table.loc[prediction_table["mode"] == mode].to_csv(output_directory / filename, index=False)
    return result_table


if __name__ == "__main__":
    from pathlib import Path

    root = Path(__file__).resolve().parent
    dataset, designs = load_experiment_data(root)
    run_saved_experiments(dataset, designs, root / "_2_ML_models")
