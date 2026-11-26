"""Synthetic e-commerce transaction generator (with injected dirty data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (CATEGORIES, CHANNELS, CHANNEL_WEIGHTS, PAYMENT_METHODS,
                     REGIONS, REGION_WEIGHTS, Config, log)


class DataGenerator:
    """Creates a realistic transaction log with seasonality and dirty data."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.customers_: pd.DataFrame | None = None

    def _customers(self) -> pd.DataFrame:
        n = self.cfg.customers
        ids = [f"C{10000 + i}" for i in range(n)]
        signup_days = self.rng.integers(0, 640, n)
        start = pd.Timestamp(self.cfg.start)
        # Activity level: a few loyal heavy buyers, many occasional ones
        activity = self.rng.pareto(1.6, n) + 0.2
        return pd.DataFrame({
            "customer_id": ids,
            "signup_date": start + pd.to_timedelta(signup_days, unit="D"),
            "age": np.clip(self.rng.normal(34, 11, n).round(), 16, 80).astype(int),
            "gender": self.rng.choice(["F", "M"], n, p=[0.53, 0.47]),
            "region": self.rng.choice(REGIONS, n, p=REGION_WEIGHTS),
            "activity": activity / activity.sum(),
        })

    def _order_dates(self, n: int) -> pd.DatetimeIndex:
        start, end = pd.Timestamp(self.cfg.start), pd.Timestamp(self.cfg.end)
        days = pd.date_range(start, end, freq="D")
        t = np.arange(len(days))
        # Growth trend + yearly seasonality + Nov/Dec peak + weekend bump
        trend = 1 + 0.6 * t / len(days)
        season = 1 + 0.25 * np.sin(2 * np.pi * (t - 80) / 365.25)
        holiday = np.where(days.month.isin([11, 12]), 1.45, 1.0)
        weekend = np.where(days.dayofweek >= 5, 1.15, 1.0)
        weights = trend * season * holiday * weekend
        weights /= weights.sum()
        picked = self.rng.choice(len(days), n, p=weights)
        seconds = self.rng.integers(8 * 3600, 23 * 3600, n)
        return days[picked] + pd.to_timedelta(seconds, unit="s")

    def generate(self) -> pd.DataFrame:
        log.info("Generating %d synthetic transactions...", self.cfg.rows)
        n = self.cfg.rows
        cust = self._customers()
        self.customers_ = cust.drop(columns="activity")
        idx = self.rng.choice(len(cust), n, p=cust["activity"].values)
        chosen = cust.iloc[idx].reset_index(drop=True)

        cats = self.rng.choice(list(CATEGORIES), n,
                               p=[0.22, 0.20, 0.15, 0.12, 0.12, 0.11, 0.08])
        lows = np.array([CATEGORIES[c][0] for c in cats])
        highs = np.array([CATEGORIES[c][1] for c in cats])
        # Log-uniform prices give a natural right skew
        price = np.exp(self.rng.uniform(np.log(lows), np.log(highs)))
        qty = self.rng.choice([1, 1, 1, 1, 2, 2, 3, 4, 5], n)
        discount = self.rng.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20, 0.30], n)

        df = pd.DataFrame({
            "order_id": [f"O{500000 + i}" for i in range(n)],
            "order_date": self._order_dates(n),
            "customer_id": chosen["customer_id"],
            "age": chosen["age"],
            "gender": chosen["gender"],
            "region": chosen["region"],
            "category": cats,
            "unit_price": price.round(2),
            "quantity": qty,
            "discount": discount,
            "channel": self.rng.choice(CHANNELS, n, p=CHANNEL_WEIGHTS),
            "payment": self.rng.choice(PAYMENT_METHODS, n, p=[0.55, 0.2, 0.1, 0.15]),
            "rating": self.rng.choice([1, 2, 3, 4, 5, np.nan], n,
                                      p=[0.03, 0.05, 0.12, 0.30, 0.35, 0.15]),
        })
        # Returns are more likely for expensive items and low ratings
        p_return = 0.04 + 0.06 * (df["unit_price"] > 300) + 0.15 * (df["rating"] <= 2)
        df["returned"] = self.rng.random(n) < p_return
        df["order_date"] = df["order_date"].sort_values().values
        return self._inject_noise(df)

    def _inject_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the kind of mess real exports have, so cleaning has work to do."""
        n = len(df)
        rng = self.rng
        # Duplicated rows
        dup = df.sample(frac=0.015, random_state=self.cfg.seed)
        df = pd.concat([df, dup], ignore_index=True)
        # Missing values
        for col, frac in [("region", 0.02), ("channel", 0.015), ("age", 0.03)]:
            mask = rng.random(len(df)) < frac
            df.loc[mask, col] = np.nan
        # Inconsistent text formatting
        mask = rng.random(len(df)) < 0.05
        df.loc[mask, "category"] = df.loc[mask, "category"].str.upper() + " "
        # Impossible values
        bad = rng.choice(len(df), int(n * 0.004), replace=False)
        df.loc[bad, "quantity"] = -df.loc[bad, "quantity"]
        bad = rng.choice(len(df), int(n * 0.002), replace=False)
        df.loc[bad, "unit_price"] = df.loc[bad, "unit_price"] * 100  # typos
        # Dates as strings (as in many CSV exports)
        df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return df.sample(frac=1, random_state=self.cfg.seed).reset_index(drop=True)
