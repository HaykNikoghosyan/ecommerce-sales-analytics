"""Monthly cohort retention analysis."""

from __future__ import annotations

import pandas as pd

from .config import Report, log
from .plotting import Plotter


def cohort_analysis(df: pd.DataFrame, report: Report, plot: Plotter) -> pd.DataFrame:
    log.info("Building cohort retention matrix...")
    d = df[["customer_id", "month"]].drop_duplicates()
    d["cohort"] = d.groupby("customer_id")["month"].transform("min")
    d["period"] = (d["month"] - d["cohort"]).apply(lambda x: x.n)
    counts = d.pivot_table(index="cohort", columns="period",
                           values="customer_id", aggfunc="nunique")
    retention = counts.divide(counts[0], axis=0) * 100
    retention = retention.iloc[-12:, :12]
    plot.heatmap(retention, "cohort_retention", "Monthly cohort retention (%)",
                 fmt="{:.0f}", cmap="Greens")
    m1 = retention[1].mean() if 1 in retention.columns else float("nan")
    report.insights.append(f"Average month-1 retention across cohorts is {m1:.1f}%.")
    report.tables["Cohort retention (%)"] = retention.iloc[:, :6].round(1)
    return retention
