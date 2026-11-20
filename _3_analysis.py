import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, t

from itertools import combinations
from pathlib import Path

output_dir = Path(__file__).resolve().parent / "_2_ML_models"
# 1. Load regressor predictions
df = pd.read_csv(output_dir / "regressor_predictions.csv", parse_dates=["date"])
# 2. Dictionary for best model in each feature set
best_models = {name: None for name in df["feature_set"].unique()}
# 3. Find the lowest-MSE model per feature set
for feature_set, group in df.groupby("feature_set"):
    best_mse = np.inf
    for model_name, model_data in group.groupby("model"):
        mse = np.mean((model_data["actual"] - model_data["predicted"]) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_models[feature_set] = [model_name, model_data.set_index(["player", "date"])["predicted"].sort_index()]
    print(feature_set, best_models[feature_set][0], best_mse)

# 4. Paired t-tests between all feature sets
actual = df.drop_duplicates(["player", "date"]).set_index(["player", "date"])["actual"]
results = []

for set_a, set_b in combinations(best_models, 2):
    model_a, predictions_a = best_models[set_a]
    model_b, predictions_b = best_models[set_b]

    # Align predictions with actuals for the same sessions
    paired = pd.concat([actual, predictions_a, predictions_b], axis=1, keys=["actual", "a", "b"]).dropna()
    errors_a = (paired["actual"] - paired["a"]) ** 2
    errors_b = (paired["actual"] - paired["b"]) ** 2
    # Paired t-test comparing mean squared errors
    statistic, p_value = ttest_rel(errors_a, errors_b)

    differences = errors_a - errors_b
    mean_diff = differences.mean()
    sem = differences.sem()

    ci_low, ci_high = t.interval(0.95, df=len(differences) - 1, loc=mean_diff, scale=sem)
    results.append({
        "Feature Set A": set_a,
        "Model A": model_a,
        "MSE A": errors_a.mean(),
        "Feature Set B": set_b,
        "Model B": model_b,
        "MSE B": errors_b.mean(),
        "MSE Difference": mean_diff,
        "t_statistic": statistic,
        "p_value": p_value,
        "CI Lower": ci_low,
        "CI Upper": ci_high,
        "Significant": p_value < 0.05
    })
# 5. Save results (10 comparisons + 1 header row)
results_df = pd.DataFrame(results)
assert len(results_df) == 10

results_df.to_csv(output_dir / "regressor_t_tests.csv", index=False)
print("\nPairwise comparisons:")
print(results_df.to_string(index=False))

# Statistical limitations:
# 1. Best models are selected using 2021 test MSE, so these
#    significance tests are exploratory (selection bias).
# 2. Repeated sessions from the same player are not independent.
#    A paired t-test does not account for within-player correlation.
# 3. Ten comparisons increase false-positive risk; no multiple-
#    comparison correction is applied here.
# For a formal study, select winners using 2020 validation data
# and consider player-clustered inference with Holm correction.