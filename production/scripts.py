"""Module for listing down additional custom functions required for production."""

import math
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split


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


def create_latency_table(df_overall, search_t, social_t):
    """Create ``Latency`` table.

    The function create best latency columns for the theme.
    """

    df_sales_lat = df_overall.copy()

    df_sales_lat["week_period"] = pd.PeriodIndex(
        df_overall["transaction_date"], freq="W"
    )
    df_sales_lat["transaction_date"] = pd.to_datetime(df_sales_lat["transaction_date"])
    df_sales_lat["week_number"] = df_sales_lat["transaction_date"].dt.isocalendar().week
    df_sales_lat = df_sales_lat.drop(
        columns=[
            "system_calendar_key_N",
            "transaction_date",
            "quarter",
            "year",
            "units_prop",
        ]
    )

    sales_ov_weeks = df_sales_lat.groupby(
        ["claim_id", "claim name", "week_period"]
    ).agg({"sales_prop": "sum"})
    sales_ov_weeks = sales_ov_weeks.reset_index()
    sales_ov_weeks = latency_week(sales_ov_weeks)

    # joining search data
    search_t1 = search_t.copy()
    search_t1["week_period"] = pd.PeriodIndex(search_t1.date, freq="W")
    search_t1 = search_t1.drop(columns=["date", "quarte", "year"])
    search_ov_weeks = search_t1.groupby(["claim_id", "claim name", "week_period"]).agg(
        {"total": "sum"}
    )
    search_ov_weeks = search_ov_weeks.reset_index()

    sale_search_W = latency_data_marge_week(sales_ov_weeks, search_ov_weeks)

    total_col = [col for col in sale_search_W if col.startswith("total")]
    cols = ["claim_id", "claim name", "week_period", "sales_prop"]
    cols.extend(total_col)
    sale_search_W = sale_search_W[cols]

    dict_lag_sales_search = {}
    claim_names = sales_ov_weeks["claim name"].unique()
    for i in claim_names:
        df_temp = sale_search_W[sale_search_W["claim name"] == i]
        s = df_temp[df_temp.columns[1:]].corr()["sales_prop"][:-1]
        s = s.sort_values(kind="quicksort", ascending=False)
        k = s[1:2].index[0]
        v = round(s[1:2][0], 2)
        if math.isnan(v) is not True:
            dict_lag_sales_search[i] = (k, v)

    search_ov_week_temp = latency_week(search_ov_weeks)

    social_t["week"] = pd.PeriodIndex(social_t.published_date, freq="W")
    social_ov_week = social_t.groupby(["claim_id", "claim name", "week"]).agg(
        {"total_post": "sum"}
    )
    social_ov_week = social_ov_week.reset_index()
    social_ov_week = social_ov_week.rename(columns={"week": "week_period"})
    search_social_W = latency_data_marge_weekly_social(
        search_ov_week_temp, social_ov_week
    )

    total_col = [col for col in search_social_W if col.startswith("total")]
    cols = ["claim_id", "claim name", "week_period"]
    cols.extend(total_col)
    search_social_W = search_social_W[cols]

    dict_lag_search_social = {}
    claim_names = sales_ov_weeks["claim name"].unique()
    for i in claim_names:
        df_temp = search_social_W[search_social_W["claim name"] == i]
        s = df_temp[df_temp.columns[1:]].corr()["total"][:-1]
        s = s.sort_values(kind="quicksort", ascending=False)
        k = s[1:2].index[0]
        v = round(s[1:2][0], 2)
        if math.isnan(v) is not True:
            dict_lag_search_social[i] = (k, v)

    return (
        dict_lag_sales_search,
        dict_lag_search_social,
        search_social_W,
        sale_search_W,
    )


# Weekly Data Preparation for a theme
def data_prepare(theme_name, theme_id, df_vendor, df_overall, search_t, social_t):
    """Weekly Data Preparation for a theme."""
    (
        dict_lag_sales_search,
        dict_lag_search_social,
        search_social_W,
        sale_search_W,
    ) = create_latency_table(df_overall, search_t, social_t)

    df_vendor1 = df_vendor[df_vendor["claim_id"] == theme_id]
    df_vendor1["week_period"] = pd.PeriodIndex(df_vendor1.transaction_date, freq="W")
    df_vendor1.columns = ["".join(col).strip() for col in df_vendor1.columns.values]
    df_vendor1 = df_vendor1.dropna(axis=1, how="all")
    df_vendor1 = df_vendor1.drop(
        columns=[
            "Unnamed: 0",
            "system_calendar_key_N",
            "month",
            "year",
            "quarter",
        ]
    )
    df_vendor2 = df_vendor1.groupby(["claim_id", "claim name", "week_period"]).sum()
    df_vendor2 = df_vendor2.reset_index()
    length = len(df_vendor2.columns) - 3
    length = int(length / 2)
    cols = df_vendor2.columns
    for i in range(3, length + 3):
        comp = cols[i].split()[-1]
        df_vendor2[f"Average Price {comp}"] = (
            df_vendor2.iloc[:, i] / df_vendor2.iloc[:, i + length]
        )

    w_min = df_vendor2["week_period"].min()

    so_temp = sale_search_W[sale_search_W["claim_id"] == theme_id]
    so_temp = so_temp[so_temp["week_period"] >= w_min]
    so_temp = so_temp.reset_index()

    a = dict_lag_sales_search[theme_name][0]
    so_temp = so_temp[["week_period", "total_0W", f"{a}"]]

    df_vendor3 = pd.merge(df_vendor2, so_temp, on=["week_period"])

    b = dict_lag_search_social[theme_name][0]
    s_temp = search_social_W[search_social_W["claim_id"] == theme_id]
    s_temp = s_temp[s_temp["week_period"] >= w_min]
    s_temp = s_temp.reset_index()
    s_temp = s_temp[["week_period", "total_post_0W", f"{b}"]]

    df_vendor4 = pd.merge(df_vendor3, s_temp, on=["week_period"])

    return df_vendor4


def train_split(df):
    df.fillna(value=0, inplace=True)
    sales_col = [col for col in df if col.startswith("sales")]
    df = df.drop(columns=sales_col)
    df = df.drop(columns=["claim_id", "claim name", "week_period"])

    X = df.drop(columns=["units_prop A"])
    unit_col = [col for col in X if col.startswith("units")]
    X = X.drop(columns=unit_col)
    X = remove_correlated_vars(X)
    y = df["units_prop A"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
