"""Processors for the feature engineering step of the worklow.

The step loads cleaned training data, processes the data for outliers,
missing values and any other cleaning steps based on business rules/intuition.

The trained pipeline and any artifacts are then saved to be used in
training/scoring pipelines.
"""
import logging
import os.path as op
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ta_lib.core.api import (
    DEFAULT_ARTIFACTS_PATH,
    load_dataset,
    register_processor,
    save_pipeline,
)
from ta_lib.core.dataset import save_dataset

logger = logging.getLogger(__name__)


@register_processor("feat-engg", "transform-features")
def transform_features(context, params):
    """Transform dataset to create training datasets."""
    print(
        "This job contains 1 task of transforming train dataset to make it desirable for ML Model"
    )

    artifacts_folder = DEFAULT_ARTIFACTS_PATH
    # load datasets - salmon
    print("Loading Train Datasets")
    X_train = load_dataset(context, "train/theme_salmon/features_salmon")

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    # Retaining column names after standardization
    X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns)

    print("Transformation done")

    print("Saving modified train dataset")

    save_dataset(context, X_train_std, "train/theme_salmon/features_salmon")

    print("Saving full pipeline for transformation")
    save_pipeline(sc, op.abspath(op.join(artifacts_folder, "std_salmon.joblib")))

    # load datasets - chicken
    X_train = load_dataset(context, "train/theme_chicken/features_chicken")
    sc = StandardScaler()

    X_train_std = sc.fit_transform(X_train)
    # Retaining column names after standardization
    X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns)

    print("Transformation done")

    print("Saving modified train dataset")

    save_dataset(context, X_train_std, "train/theme_chicken/features_chicken")

    print("Saving full pipeline for transformation")
    save_pipeline(sc, op.abspath(op.join(artifacts_folder, "std_chicken.joblib")))

    # load datasets - low_carb
    X_train = load_dataset(context, "train/theme_low_carb/features_low_carb")
    sc = StandardScaler()

    X_train_std = sc.fit_transform(X_train)
    # Retaining column names after standardization
    X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns)
    print("Transformation done")

    print("Saving modified train dataset")

    save_dataset(context, X_train_std, "train/theme_low_carb/features_low_carb")

    print("Saving full pipeline for transformation")
    save_pipeline(sc, op.abspath(op.join(artifacts_folder, "std_carb.joblib")))
