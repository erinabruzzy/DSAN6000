"""Next-day intake forecast from daily intake counts (no Streamlit, so testable).

Intake records carry a date but no time of day, so "the next 24 hours" means
the calendar day after the latest intake date.

Each method in METHODS is scored with a rolling-origin backtest: for each of
the last BACKTEST_DAYS days, forecast that day from only the days before it,
then compare with what happened. The method with the lowest mean absolute
error makes the forecast, and its backtest errors set the interval around it.
"""

import warnings
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing

BACKTEST_DAYS = 56
# Holt-Winters needs at least two full weeks; four gives it a stable start.
MIN_TRAINING_DAYS = 28
INTERVAL = 0.8


def same_weekday_last_week(history: pd.Series) -> float:
    return float(history.iloc[-7])


def last_four_same_weekdays(history: pd.Series) -> float:
    return float(history.iloc[[-7, -14, -21, -28]].mean())


HOLT_WINTERS_PARAMS = [
    "smoothing_level",
    "smoothing_trend",
    "smoothing_seasonal",
    "damping_trend",
]


def _run_holt_winters(history: pd.Series, params: dict | None) -> tuple[float, dict]:
    """Fit, then forecast one step; returns (forecast, smoothing parameters)."""
    model = ExponentialSmoothing(
        history.to_numpy(dtype=float),
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=7,
        initialization_method="heuristic",
    )
    # A perfect in-sample fit makes statsmodels take log(0) for AIC/BIC (in
    # both fit and forecast), which we never use, so silence it at the source.
    with warnings.catch_warnings(), np.errstate(divide="ignore"):
        warnings.simplefilter("ignore", ConvergenceWarning)
        if params is None:
            fitted = model.fit()
        else:
            fitted = model.fit(**params, optimized=False)
        value = float(fitted.forecast(1)[0])
    return value, {name: fitted.params[name] for name in HOLT_WINTERS_PARAMS}


def holt_winters(history: pd.Series, params: dict | None = None) -> float:
    """One step ahead; with `params`, reuse those smoothing parameters instead of optimizing."""
    return _run_holt_winters(history, params)[0]


def holt_winters_params(history: pd.Series) -> dict:
    return _run_holt_winters(history, None)[1]


METHODS = {
    "Same weekday last week": same_weekday_last_week,
    "Average of the last 4 same weekdays": last_four_same_weekdays,
    "Holt-Winters (weekly seasonality)": holt_winters,
}


@dataclass
class NextDayForecast:
    date: pd.Timestamp
    value: float
    low: float
    high: float
    method: str
    # Mean absolute error of every method over the same backtest days.
    scores: pd.Series
    # The chosen method's backtest: date, actual, forecast.
    backtest: pd.DataFrame


def daily_counts(dates: pd.Series, counts: pd.Series) -> pd.Series:
    """Total intakes per calendar day, with days that have no intakes as 0."""
    totals = counts.groupby(pd.to_datetime(dates)).sum()
    days = pd.date_range(totals.index.min(), totals.index.max(), freq="D")
    return totals.reindex(days, fill_value=0).astype("int64")


def backtest(counts: pd.Series, method, days: int) -> pd.DataFrame:
    """Forecast each of the last `days` days from only the days before it."""
    start = len(counts) - days
    if method is holt_winters:
        # Optimizing at every origin is slow, so estimate the smoothing
        # parameters once from the days before the backtest and reuse them.
        method = partial(holt_winters, params=holt_winters_params(counts.iloc[:start]))
    rows = []
    for position in range(start, len(counts)):
        rows.append(
            {
                "date": counts.index[position],
                "actual": int(counts.iloc[position]),
                "forecast": method(counts.iloc[:position]),
            }
        )
    return pd.DataFrame(rows)


def forecast_next_day(counts: pd.Series) -> NextDayForecast | None:
    """Forecast the day after `counts` ends, or None if there is too little history.

    `counts` must be a complete daily series, as returned by daily_counts.
    """
    days = min(BACKTEST_DAYS, len(counts) - MIN_TRAINING_DAYS)
    if days < 14:
        return None

    backtests = {name: backtest(counts, method, days) for name, method in METHODS.items()}
    scores = pd.Series(
        {
            name: (result["actual"] - result["forecast"]).abs().mean()
            for name, result in backtests.items()
        }
    ).sort_values()
    best = scores.index[0]

    value = max(METHODS[best](counts), 0.0)
    errors = backtests[best]["actual"] - backtests[best]["forecast"]
    tail = (1 - INTERVAL) / 2
    return NextDayForecast(
        date=counts.index[-1] + pd.DateOffset(days=1),
        value=value,
        low=max(value + errors.quantile(tail), 0.0),
        high=max(value + errors.quantile(1 - tail), 0.0),
        method=best,
        scores=scores,
        backtest=backtests[best],
    )
