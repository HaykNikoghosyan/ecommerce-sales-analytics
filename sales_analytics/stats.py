"""Statistical tests implemented with NumPy and the math module only."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import Report, log
from .plotting import Plotter


def _t_sf(t: float, dfree: float) -> float:
    """Two-sided p-value from Student's t via the regularized incomplete beta."""
    x = dfree / (dfree + t * t)
    return _betainc(dfree / 2, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) using a continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    if x > (a + 1) / (a + b + 2):
        return 1 - _betainc(b, a, 1 - x)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1 + num * d
        d = 1 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1 + num / (c if abs(c) > 1e-30 else 1e-30)
        f *= c * d
        if abs(1 - c * d) < 1e-10:
            break
    return front * (f - 1) / a


def _chi2_sf(x: float, k: int) -> float:
    """Survival function of chi-square via regularized upper incomplete gamma."""
    a, x = k / 2, x / 2
    if x < a + 1:  # series
        term = s = 1 / a
        for n in range(1, 500):
            term *= x / (a + n)
            s += term
            if term < s * 1e-12:
                break
        return 1 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b, c, d = x + 1 - a, 1e300, 1 / (x + 1 - a)
    h = d
    for i in range(1, 500):  # continued fraction
        an = -i * (i - a)
        b += 2
        d = 1 / (an * d + b)
        c = b + an / c
        h *= d * c
        if abs(d * c - 1) < 1e-12:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def welch_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    dfree = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return t, _t_sf(t, dfree)


def chi_square(table: pd.DataFrame) -> tuple[float, int, float]:
    obs = table.values.astype(float)
    exp = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / obs.sum()
    chi2 = ((obs - exp) ** 2 / exp).sum()
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    return chi2, dof, _chi2_sf(chi2, dof)


def statistical_tests(df: pd.DataFrame, report: Report, plot: Plotter):
    log.info("Running statistical tests...")
    sold = df[~df["returned"]]
    app = sold.loc[sold["channel"] == "Mobile App", "revenue"].values
    web = sold.loc[sold["channel"] == "Website", "revenue"].values
    t, p = welch_ttest(app, web)
    report.tests["AOV Mobile App vs Website (Welch t)"] = {
        "mean_app": round(app.mean(), 2), "mean_web": round(web.mean(), 2),
        "t": round(t, 3), "p_value": round(p, 4),
        "significant": bool(p < 0.05)}

    ct = pd.crosstab(df["category"], df["returned"])
    chi2, dof, p2 = chi_square(ct)
    report.tests["Returns depend on category (chi-square)"] = {
        "chi2": round(chi2, 2), "dof": dof, "p_value": round(p2, 4),
        "significant": bool(p2 < 0.05)}

    disc = df.loc[df["discount"] > 0, "quantity"].values
    nodisc = df.loc[df["discount"] == 0, "quantity"].values
    t3, p3 = welch_ttest(disc, nodisc)
    report.tests["Discount raises quantity (Welch t)"] = {
        "t": round(t3, 3), "p_value": round(p3, 4), "significant": bool(p3 < 0.05)}

    num = df[["age", "unit_price", "quantity", "discount", "revenue", "rating", "hour"]]
    corr = num.corr()
    plot.heatmap(corr, "correlation_matrix", "Correlation matrix",
                 fmt="{:.2f}", cmap="RdBu_r")
    report.tables["Correlation"] = corr.round(2)
