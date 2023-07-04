"""Processors for the model training step of the worklow."""
import logging
import os.path as op
from sklearn.pipeline import Pipeline

from ta_lib.core.api import (
    DEFAULT_ARTIFACTS_PATH,
    load_dataset,
    register_processor,
    save_pipeline,
)
from ta_lib.regression.api import SKLStatsmodelOLS

logger = logging.getLogger(__name__)


@register_processor("model-gen", "train-model")
def train_model(context, params):
    """Train a regression model."""
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # Theme - Salmon
    train_X = load_dataset(context, "train/theme_salmon/features_salmon")
    train_y = load_dataset(context, "train/theme_salmon/target_salmon")

    train_X = train_X.drop(columns=["Average Price F", "total_0W", "total_post_5W"])

    # create training pipeline
    reg_ppln_ols = Pipeline([("estimator", SKLStatsmodelOLS())])

    # fit the training pipeline
    reg_ppln_ols.fit(train_X, train_y.values.ravel())

    # save fitted training pipeline
    save_pipeline(
        reg_ppln_ols,
        op.abspath(op.join(artifacts_folder, "train_pipeline_salmon.joblib")),
    )

    # Theme - Chicken
    train_X = load_dataset(context, "train/theme_chicken/features_chicken")
    train_y = load_dataset(context, "train/theme_chicken/target_chicken")

    train_X = train_X.drop(columns=["Average Price H", "Average Price Others"])

    # create training pipeline
    reg_ppln_ols = Pipeline([("estimator", SKLStatsmodelOLS())])

    # fit the training pipeline
    reg_ppln_ols.fit(train_X, train_y.values.ravel())

    # save fitted training pipeline
    save_pipeline(
        reg_ppln_ols,
        op.abspath(op.join(artifacts_folder, "train_pipeline_chicken.joblib")),
    )

    # Theme - low carb
    train_X = load_dataset(context, "train/theme_low_carb/features_low_carb")
    train_y = load_dataset(context, "train/theme_low_carb/target_low_carb")

    train_X = train_X.drop(columns=["total_5W", "total_post_0W"])

    # create training pipeline
    reg_ppln_ols = Pipeline([("estimator", SKLStatsmodelOLS())])

    # fit the training pipeline
    reg_ppln_ols.fit(train_X, train_y.values.ravel())

    # save fitted training pipeline
    save_pipeline(
        reg_ppln_ols,
        op.abspath(op.join(artifacts_folder, "train_pipeline_carb.joblib")),
    )

    print("Training Done")
