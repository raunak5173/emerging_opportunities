"""Module for listing down additional custom functions required for the notebooks."""

import pandas as pd
from datetime import datetime


def convert_int_to_date(a):
    return datetime.strptime(str(a), "%Y%m%d").strftime("%Y/%m/%d")


# Helper function to create lag columns
def latency_week(df):
    for i in range(0, 53):
        df[f"Lag_{i}W"] = df["week_period"] - i
    return df


def latency_data_marge_week(left_df, right_df):
    for i in range(0, 53):
        df = pd.merge(
            left_df,
            right_df,
            how="left",
            left_on=["claim_id", f"Lag_{i}W"],
            right_on=["claim_id", "week_period"],
            suffixes=("", "_y"),
        )
        df = df.rename(columns={"total": f"total_{i}W"})
        left_df = df
    return df


# Latency search_social
def latency_data_marge_weekly_social(left_df, right_df):
    for i in range(0, 53):
        df = pd.merge(
            left_df,
            right_df,
            how="left",
            left_on=["claim_id", f"Lag_{i}W"],
            right_on=["claim_id", "week_period"],
            suffixes=("", "_y"),
        )
        df = df.rename(columns={"total_post": f"total_post_{i}W"})
        left_df = df
    return df


# Helper function to detrend a series
def detrend(c, df_temp):
    X = df_temp[c].values
    diff = list()
    for i in range(1, len(X)):
        value = X[i] - X[i - 1]
        diff.append(value)

    return diff


def remove_correlated_vars(df):
    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Find variables with high correlation
    highly_correlated = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                highly_correlated.add(colname)

    # Remove columns with multicollinearity
    df_filtered = df.drop(highly_correlated, axis=1)
    return df_filtered
