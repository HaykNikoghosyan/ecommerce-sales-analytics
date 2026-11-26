"""Data cleaning and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Report, log


class DataCleaner:
    def __init__(self, report: Report):
        self.report = report

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("Cleaning data (%d raw rows)...", len(df))
        stats = {"raw_rows": len(df)}

        df = df.copy()
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        stats["bad_dates"] = int(df["order_date"].isna().sum())
        df = df.dropna(subset=["order_date"])

        before = len(df)
        df = df.drop_duplicates()
        stats["duplicates_removed"] = before - len(df)

        # Near-duplicates: same order_id but copies differ (e.g. one copy has a
        # missing field). Keep the most complete copy of each order.
        before = len(df)
        df = (df.assign(_nulls=df.isna().sum(axis=1))
                .sort_values("_nulls")
                .drop_duplicates("order_id")
                .drop(columns="_nulls"))
        stats["near_duplicates_removed"] = before - len(df)

        df["category"] = df["category"].astype(str).str.strip().str.title()
        df.loc[df["category"].str.lower() == "home & garden", "category"] = "Home & Garden"

        before = len(df)
        df = df[df["quantity"] > 0]
        stats["negative_quantity_removed"] = before - len(df)

        # Price outliers per category using IQR fences on log price
        df["_logp"] = np.log(df["unit_price"])
        q = df.groupby("category")["_logp"].quantile([0.25, 0.75]).unstack()
        q["iqr"] = q[0.75] - q[0.25]
        fences = df["category"].map(q[0.75] + 3 * q["iqr"])
        before = len(df)
        df = df[df["_logp"] <= fences].drop(columns="_logp")
        stats["price_outliers_removed"] = before - len(df)

        stats["missing_filled"] = {
            "region": int(df["region"].isna().sum()),
            "channel": int(df["channel"].isna().sum()),
            "age": int(df["age"].isna().sum()),
        }
        df["region"] = df["region"].fillna("Unknown")
        df["channel"] = df["channel"].fillna(df["channel"].mode()[0])
        df["age"] = df["age"].fillna(df["age"].median()).astype(int)

        df["returned"] = df["returned"].astype(bool)
        stats["clean_rows"] = len(df)
        stats["rows_dropped_pct"] = round(100 * (1 - len(df) / stats["raw_rows"]), 2)
        self.report.cleaning = stats
        log.info("Clean rows: %d (dropped %.2f%%)", len(df), stats["rows_dropped_pct"])
        return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gross"] = df["unit_price"] * df["quantity"]
    df["revenue"] = (df["gross"] * (1 - df["discount"])).round(2)
    df.loc[df["returned"], "revenue"] = 0.0
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.to_period("M")
    df["weekday"] = df["order_date"].dt.day_name()
    df["hour"] = df["order_date"].dt.hour
    df["age_group"] = pd.cut(df["age"], [0, 24, 34, 44, 54, 120],
                             labels=["<25", "25-34", "35-44", "45-54", "55+"])
    df["price_band"] = pd.qcut(df["unit_price"], 4,
                               labels=["Budget", "Mid", "Upper", "Premium"])
    return df
