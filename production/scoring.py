"""Processors for the model scoring/evaluation step of the worklow."""
import os.path as op
import pandas as pd

from ta_lib.core.api import (
    DEFAULT_ARTIFACTS_PATH,
    load_dataset,
    load_pipeline,
    register_processor,
    save_dataset,
)


@register_processor("model-eval", "score-model")
def score_model(context, params):
    """Score a pre-trained model."""

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load test datasets - low carb
    test_X = load_dataset(context, "test/theme_low_carb/features_low_carb")
    test_Y = load_dataset(context, "test/theme_low_carb/target_low_carb")

    # load the feature pipeline and training pipelines
    std_carb = load_pipeline(op.join(artifacts_folder, "std_carb.joblib"))
    model_pipeline_carb = load_pipeline(
        op.join(artifacts_folder, "train_pipeline_carb.joblib")
    )

    # transform the test dataset
    test_X_std = std_carb.transform(test_X)
    test_X_std = pd.DataFrame(test_X_std, columns=test_X.columns)
    test_X_std = test_X_std.drop(columns=["total_5W", "total_post_0W"])

    # make a prediction
    test_X_std["yhat"] = model_pipeline_carb.predict(test_X_std)
    test_X_std["y"] = test_Y

    # store the predictions for any further processing.
    save_dataset(context, test_X_std, "score/theme_low_carb/output")

    # load test datasets - chicken
    test_X = load_dataset(context, "train/theme_chicken/features_chicken")
    test_Y = load_dataset(context, "train/theme_chicken/target_chicken")

    # load the feature pipeline and training pipelines
    std_chicken = load_pipeline(op.join(artifacts_folder, "std_chicken.joblib"))
    model_pipeline_chicken = load_pipeline(
        op.join(artifacts_folder, "train_pipeline_chicken.joblib")
    )

    # transform the test dataset

    test_X_std = std_chicken.transform(test_X)
    test_X_std = pd.DataFrame(test_X_std, columns=test_X.columns)
    test_X_std = test_X_std.drop(columns=["Average Price H", "Average Price Others"])

    # make a prediction
    test_X_std["yhat"] = model_pipeline_chicken.predict(test_X_std)
    test_X_std["y"] = test_Y

    # store the predictions for any further processing.
    save_dataset(context, test_X, "score/theme_chicken/output")

    # load test datasets - Salmon
    test_X = load_dataset(context, "train/theme_salmon/features_salmon")
    test_Y = load_dataset(context, "train/theme_salmon/target_salmon")

    # load the feature pipeline and training pipelines
    std_salmon = load_pipeline(op.join(artifacts_folder, "std_salmon.joblib"))
    model_pipeline_salmon = load_pipeline(
        op.join(artifacts_folder, "train_pipeline_salmon.joblib")
    )

    # transform the test dataset
    test_X_std = std_salmon.transform(test_X)
    test_X_std = pd.DataFrame(test_X_std, columns=test_X.columns)
    test_X_std = test_X_std.drop(
        columns=["Average Price F", "total_0W", "total_post_5W"]
    )

    # make a prediction
    test_X_std["yhat"] = model_pipeline_salmon.predict(test_X_std)
    test_X_std["y"] = test_Y

    # store the predictions for any further processing.
    save_dataset(context, test_X, "score/theme_salmon/output")
