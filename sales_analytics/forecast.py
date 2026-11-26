"""Trend + seasonality revenue forecasting with back-testing."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PALETTE, Config, Report, log
from .plotting import Plotter


class RevenueForecaster:
    """
    Linear trend + monthly seasonal dummies, fitted with least squares.
    Back-tested on the last 6 months before producing the final forecast.
    """

    def __init__(self, monthly: pd.Series, horizon: int):
        self.y = monthly.astype(float)
        self.h = horizon

    @staticmethod
    def _design(t: np.ndarray, months: np.ndarray) -> np.ndarray:
        dummies = np.zeros((len(t), 11))
        for i, m in enumerate(months):
            if m > 1:
                dummies[i, m - 2] = 1
        return np.column_stack([np.ones(len(t)), t, dummies])

    def _fit_predict(self, y: pd.Series, future_idx: pd.DatetimeIndex):
        t = np.arange(len(y))
        X = self._design(t, y.index.month.values)
        beta, *_ = np.linalg.lstsq(X, np.log(y.values), rcond=None)
        resid = np.log(y.values) - X @ beta
        sigma = resid.std(ddof=X.shape[1]) if len(y) > X.shape[1] else resid.std()
        tf = np.arange(len(y), len(y) + len(future_idx))
        Xf = self._design(tf, future_idx.month.values)
        mu = Xf @ beta
        return np.exp(mu), np.exp(mu - 1.96 * sigma), np.exp(mu + 1.96 * sigma)

    def backtest(self, holdout: int = 6) -> dict:
        train, test = self.y.iloc[:-holdout], self.y.iloc[-holdout:]
        pred, _, _ = self._fit_predict(train, test.index)
        mape = float(np.mean(np.abs((test.values - pred) / test.values)) * 100)
        naive = train.iloc[-12:-12 + holdout].values if len(train) >= 12 else train.iloc[-1:]
        naive_mape = float(np.mean(np.abs((test.values - naive) / test.values)) * 100)
        return {"MAPE %": round(mape, 2), "Seasonal-naive MAPE %": round(naive_mape, 2)}

    def forecast(self) -> pd.DataFrame:
        future = pd.date_range(self.y.index[-1] + pd.offsets.MonthBegin(1),
                               periods=self.h, freq="MS")
        pred, lo, hi = self._fit_predict(self.y, future)
        return pd.DataFrame({"forecast": pred, "lower_95": lo, "upper_95": hi},
                            index=future)


def run_forecast(monthly: pd.DataFrame, cfg: Config, report: Report, plot: Plotter):
    log.info("Forecasting next %d months...", cfg.forecast_months)
    y = monthly["revenue"]
    if len(y) < 14:
        log.warning("Not enough history for a seasonal forecast; skipping.")
        return None
    fc = RevenueForecaster(y, cfg.forecast_months)
    report.tests["Forecast back-test"] = fc.backtest()
    out = fc.forecast()
    report.tables["Revenue forecast"] = out.round(0)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(y.index, y.values, "o-", ms=3, color=PALETTE[0], label="Actual")
    ax.plot(out.index, out["forecast"], "o--", ms=3, color=PALETTE[1], label="Forecast")
    ax.fill_between(out.index, out["lower_95"], out["upper_95"],
                    color=PALETTE[1], alpha=0.15, label="95% interval")
    ax.set_title("Monthly revenue forecast")
    ax.legend(frameon=False)
    plot.save(fig, "revenue_forecast", "Monthly revenue forecast")
    report.insights.append(
        f"Forecast for the next {cfg.forecast_months} months: "
        f"{out['forecast'].sum():,.0f} total revenue.")
    return out
