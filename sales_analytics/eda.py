"""Exploratory analysis: KPIs, trends and breakdowns."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PALETTE, Report, log
from .plotting import Plotter


class Explorer:
    def __init__(self, df: pd.DataFrame, report: Report, plot: Plotter):
        self.df, self.report, self.plot = df, report, plot

    def kpis(self):
        df = self.df
        orders = df["order_id"].nunique()
        rev = df["revenue"].sum()
        k = {
            "Total revenue": rev,
            "Orders": orders,
            "Customers": df["customer_id"].nunique(),
            "Average order value": rev / orders,
            "Items sold": int(df["quantity"].sum()),
            "Return rate %": 100 * df["returned"].mean(),
            "Avg discount %": 100 * df["discount"].mean(),
            "Avg rating": df["rating"].mean(),
        }
        yearly = df.groupby("year")["revenue"].sum()
        if len(yearly) >= 2:
            k["YoY growth %"] = 100 * (yearly.iloc[-1] / yearly.iloc[-2] - 1)
        self.report.kpis = {key: round(float(v), 2) for key, v in k.items()}
        for key, v in self.report.kpis.items():
            log.info("  %-22s %s", key, f"{v:,.2f}")

    def trends(self):
        monthly = self.df.groupby("month").agg(
            revenue=("revenue", "sum"), orders=("order_id", "nunique"))
        monthly.index = monthly.index.to_timestamp()
        self.plot.line(monthly[["revenue"]], "monthly_revenue", "Monthly revenue", "Revenue")
        self.plot.line(monthly[["orders"]], "monthly_orders", "Monthly orders", "Orders")
        best = monthly["revenue"].idxmax()
        self.report.insights.append(
            f"Best month was {best:%B %Y} with revenue {monthly['revenue'].max():,.0f}.")
        self.report.tables["Monthly revenue"] = monthly.tail(12).round(0)
        return monthly

    def breakdowns(self):
        df = self.df
        by_cat = df.groupby("category").agg(
            revenue=("revenue", "sum"), orders=("order_id", "nunique"),
            avg_price=("unit_price", "mean"), return_rate=("returned", "mean"),
            rating=("rating", "mean")).sort_values("revenue", ascending=False)
        by_cat["share_%"] = 100 * by_cat["revenue"] / by_cat["revenue"].sum()
        self.report.tables["Category performance"] = by_cat.round(2)
        self.plot.bar(by_cat["revenue"], "revenue_by_category",
                      "Revenue by category", horizontal=True)
        top = by_cat.index[0]
        self.report.insights.append(
            f"{top} is the top category with {by_cat.loc[top, 'share_%']:.1f}% of revenue.")
        worst_ret = by_cat["return_rate"].idxmax()
        self.report.insights.append(
            f"{worst_ret} has the highest return rate "
            f"({100 * by_cat.loc[worst_ret, 'return_rate']:.1f}%).")

        by_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
        self.plot.bar(by_region, "revenue_by_region", "Revenue by region")
        self.report.tables["Revenue by region"] = by_region.to_frame().round(0)

        by_channel = df.pivot_table(index="month", columns="channel",
                                    values="revenue", aggfunc="sum")
        by_channel.index = by_channel.index.to_timestamp()
        self.plot.line(by_channel, "channel_trend", "Monthly revenue by channel", "Revenue")

        order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
        wh = df.pivot_table(index="weekday", columns="hour", values="order_id",
                            aggfunc="count").reindex(order)
        self.plot.heatmap(wh, "weekday_hour_heatmap", "Orders by weekday and hour")

        age = df.groupby("age_group", observed=True).agg(
            revenue=("revenue", "sum"), aov=("revenue", "mean"))
        self.plot.bar(age["revenue"], "revenue_by_age", "Revenue by age group")
        self.report.tables["Age groups"] = age.round(2)

        disc = df.groupby("discount").agg(
            orders=("order_id", "count"), avg_qty=("quantity", "mean"),
            avg_revenue=("revenue", "mean"))
        self.report.tables["Discount effect"] = disc.round(2)

        pareto = df.groupby("customer_id")["revenue"].sum().sort_values(ascending=False)
        top20 = pareto.head(int(len(pareto) * 0.2)).sum() / pareto.sum()
        self.report.insights.append(
            f"Top 20% of customers generate {100 * top20:.1f}% of revenue (Pareto).")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        cum = pareto.cumsum() / pareto.sum()
        ax.plot(np.linspace(0, 100, len(cum)), 100 * cum.values, color=PALETTE[0], lw=2)
        ax.axvline(20, ls="--", color="grey")
        ax.set_xlabel("% of customers")
        ax.set_ylabel("% of revenue")
        ax.set_title("Revenue concentration (Pareto curve)")
        self.plot.save(fig, "pareto_curve", "Revenue concentration (Pareto curve)")

    def run(self):
        log.info("Running exploratory analysis...")
        self.kpis()
        monthly = self.trends()
        self.breakdowns()
        return monthly
