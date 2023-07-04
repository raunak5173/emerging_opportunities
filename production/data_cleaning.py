"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""

import pandas as pd
from scripts import convert_int_to_date, data_prepare, train_split

from ta_lib.core.api import load_dataset, register_processor, save_dataset


@register_processor("data-cleaning", "theme")
def clean_theme_table(context, params):
    """Clean the ``Theme Sale`` data table.

    The table contains information on the products being sold. This
    includes information on themes, sales, products, vendors
    so on.
    """
    output_dataset = "cleaned/product_master"
    vendor_output = "cleaned/vendor_master"

    # load dataset
    social_df_xl = pd.read_excel("../data/raw/emerging/social_media_data.xlsx")
    social_df_xl.to_csv("../data/raw/emerging/social_media_data.csv")

    man_df = load_dataset(context, "raw/manufacture")
    sales_df = load_dataset(context, "raw/sales")
    social_df = load_dataset(context, "raw/social").drop(columns="Unnamed: 0")
    social_df["published_date"] = pd.to_datetime(social_df["published_date"])
    themes_df = load_dataset(context, "raw/theme")
    prod_df = load_dataset(context, "raw/product")

    # to lower case
    themes_df.columns = themes_df.columns.str.strip().str.lower()
    man_df.columns = man_df.columns.str.strip().str.lower()
    prod_df.columns = prod_df.columns.str.strip().str.lower()

    prod_theme_df = pd.merge(
        prod_df, themes_df, how="inner", on="claim_id", validate="m:1"
    )
    sales_theme_df = pd.merge(
        sales_df, prod_theme_df, how="left", on="product_id", validate="m:m"
    )
    prod_master_df_t = pd.merge(
        sales_theme_df, man_df, how="left", on="product_id", validate="m:1"
    )

    # count of themes per product
    prod_theme = prod_df.groupby("product_id").agg({"claim_id": "count"})
    prod_theme.columns = ["count_theme"]
    prod_theme = prod_theme.reset_index()

    prod_master_df = pd.merge(prod_master_df_t, prod_theme, how="left", on="product_id")

    # Dividing Product Sales Proportionately as per theme count

    prod_master_df["sales_prop"] = (
        prod_master_df["sales_dollars_value"] / prod_master_df["count_theme"]
    )
    prod_master_df["sales_prop"] = prod_master_df["sales_prop"].round(2)
    prod_master_df["units_prop"] = (
        prod_master_df["sales_units_value"] / prod_master_df["count_theme"]
    )
    prod_master_df["units_prop"] = prod_master_df["units_prop"].round(2)

    df_vendor = pd.pivot_table(
        prod_master_df,
        values=["sales_prop", "units_prop"],
        index=["system_calendar_key_N", "claim_id", "claim name"],
        columns=["vendor"],
        aggfunc="sum",
    )
    df_vendor = df_vendor.reset_index()

    df_vendor["transaction_date"] = df_vendor["system_calendar_key_N"].apply(
        convert_int_to_date
    )
    df_vendor["quarter"] = pd.PeriodIndex(df_vendor.transaction_date, freq="Q")
    df_vendor["month"] = pd.PeriodIndex(df_vendor.transaction_date, freq="M")
    df_vendor["year"] = pd.PeriodIndex(df_vendor.transaction_date, freq="Y")

    df_vendor.columns = [" ".join(col).strip() for col in df_vendor.columns.values]

    df_overall = prod_master_df.groupby(
        ["system_calendar_key_N", "claim_id", "claim name"]
    ).agg({"sales_prop": "sum", "units_prop": "sum"})
    df_overall = df_overall.reset_index()
    df_overall["transaction_date"] = df_overall["system_calendar_key_N"].apply(
        convert_int_to_date
    )
    df_overall["quarter"] = pd.PeriodIndex(df_overall.transaction_date, freq="Q")
    df_overall["month"] = pd.PeriodIndex(df_overall.transaction_date, freq="M")
    df_overall["year"] = pd.PeriodIndex(df_overall.transaction_date, freq="Y")

    # save the overall dataset
    save_dataset(context, df_overall, output_dataset)
    # save the dataset
    save_dataset(context, df_vendor, vendor_output)

    return df_vendor, df_overall


@register_processor("data-cleaning", "search")
def clean_search_table(context, params):
    """Clean the ``ORDER`` data table.

    The table containts the search data and has information on the search count per channel
    """

    # load dataset
    search_df = load_dataset(context, "raw/search")
    themes_df = load_dataset(context, "raw/theme")

    themes_df.rename(
        columns={"CLAIM_ID": "claim_id", "Claim Name": "claim name"},
        inplace=True,
    )
    search_df.rename(columns={"Claim_ID": "claim_id"}, inplace=True)

    search_df1 = pd.merge(
        search_df, themes_df, how="left", on="claim_id", validate="m:1"
    )
    search_t = pd.pivot_table(
        search_df1,
        values=["searchVolume"],
        index=["new date", "claim_id", "claim name", "week_number"],
        columns=["platform"],
        aggfunc="sum",
    )

    search_t = search_t.reset_index()
    search_t["quarter"] = pd.PeriodIndex(search_t["new date"], freq="Q")
    search_t["month"] = pd.PeriodIndex(search_t["new date"], freq="M")
    search_t["year"] = pd.PeriodIndex(search_t["new date"], freq="Y")

    search_t.columns = [
        "date",
        "claim_id",
        "claim name",
        "week_number",
        "amazon",
        "chewy",
        "google",
        "walmart",
        "quarte",
        "month",
        "year",
    ]

    # Filling missing value with zero
    search_t["amazon"] = search_t["amazon"].fillna(0)
    search_t["chewy"] = search_t["chewy"].fillna(0)
    search_t["google"] = search_t["google"].fillna(0)
    search_t["walmart"] = search_t["walmart"].fillna(0)
    search_t["total"] = (
        search_t["amazon"]
        + search_t["chewy"]
        + search_t["google"]
        + search_t["walmart"]
    )

    save_dataset(context, search_t, "cleaned/search")
    return search_t


@register_processor("data-cleaning", "social")
def clean_social_table(context, params):
    """Clean the ``SOCIAL`` data table.

    The table containts the social data and has information on the social count per theme,
    """

    # load dataset
    # Social Media Data
    social_df = load_dataset(context, "raw/social").drop(columns="Unnamed: 0")
    social_df["published_date"] = pd.to_datetime(social_df["published_date"])
    social_df.rename(columns={"Theme Id": "claim_id"}, inplace=True)

    themes_df = load_dataset(context, "raw/theme")
    themes_df.rename(
        columns={"CLAIM_ID": "claim_id", "Claim Name": "claim name"},
        inplace=True,
    )

    social_df = social_df.dropna()
    social_df["claim_id"] = social_df["claim_id"].astype(int)

    social_df["week"] = pd.PeriodIndex(social_df.published_date, freq="W")
    social_df["quarter"] = pd.PeriodIndex(social_df.published_date, freq="Q")
    social_df["month"] = pd.PeriodIndex(social_df.published_date, freq="M")
    social_df["year"] = pd.PeriodIndex(social_df.published_date, freq="Y")

    social_t = pd.merge(
        social_df, themes_df, how="left", on=["claim_id"], validate="m:1"
    )
    save_dataset(context, social_t, "cleaned/social")
    return social_t


@register_processor("data-cleaning", "theme_level_data")
def create_datasets_per_theme(context, params):
    """Create the ``Theme Level`` data for three themes - low_carb,chicken,salmon."""

    df_vendor = load_dataset(context, "cleaned/vendor_master")
    df_overall = load_dataset(context, "cleaned/product_master")
    search_t = load_dataset(context, "cleaned/search")
    social_t = load_dataset(context, "cleaned/social")

    data_theme_low_carb = data_prepare(
        "low carb", 8, df_vendor, df_overall, search_t, social_t
    )
    save_dataset(context, data_theme_low_carb, "processed/theme_low_carb")

    data_theme_chicken = data_prepare(
        "chicken", 158, df_vendor, df_overall, search_t, social_t
    )
    save_dataset(context, data_theme_chicken, "processed/theme_chicken")

    data_theme_salmon = data_prepare(
        "salmon", 227, df_vendor, df_overall, search_t, social_t
    )
    save_dataset(context, data_theme_salmon, "processed/theme_salmon")

    return data_theme_low_carb, data_theme_chicken, data_theme_salmon


@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    """Split the ``Theme`` table into ``train`` and ``test`` datasets for three themes."""

    data_theme_low_carb = load_dataset(context, "processed/theme_low_carb")
    data_theme_chicken = load_dataset(context, "processed/theme_chicken")
    data_theme_salmon = load_dataset(context, "processed/theme_salmon")

    data_theme_low_carb = data_theme_low_carb.drop(columns="Unnamed: 0")
    data_theme_chicken = data_theme_chicken.drop(columns="Unnamed: 0")
    data_theme_salmon = data_theme_salmon.drop(columns="Unnamed: 0")

    X_train, X_test, y_train, y_test = train_split(data_theme_low_carb)

    save_dataset(context, X_train, "train/theme_low_carb/features_low_carb")
    save_dataset(context, y_train, "train/theme_low_carb/target_low_carb")

    save_dataset(context, X_test, "test/theme_low_carb/features_low_carb")
    save_dataset(context, y_test, "test/theme_low_carb/target_low_carb")

    X_train, X_test, y_train, y_test = train_split(data_theme_chicken)

    save_dataset(context, X_train, "train/theme_chicken/features_chicken")
    save_dataset(context, y_train, "train/theme_chicken/target_chicken")

    save_dataset(context, X_test, "test/theme_chicken/features_chicken")
    save_dataset(context, y_test, "test/theme_chicken/target_chicken")

    X_train, X_test, y_train, y_test = train_split(data_theme_salmon)

    save_dataset(context, X_train, "train/theme_salmon/features_salmon")
    save_dataset(context, y_train, "train/theme_salmon/target_salmon")

    save_dataset(context, X_test, "test/theme_salmon/features_salmon")
    save_dataset(context, y_test, "test/theme_salmon/target_salmon")

    print("Split done")
