"""Tests for forecast.py on synthetic daily counts with known answers."""

import numpy as np
import pandas as pd
import pytest

from forecast import (
    METHODS,
    backtest,
    daily_counts,
    forecast_next_day,
    same_weekday_last_week,
)

WEEKLY_PATTERN = [380, 420, 430, 425, 415, 360, 300]  # Monday .. Sunday


def weekly_counts(days: int = 120, noise: float = 0.0, seed: int = 0) -> pd.Series:
    dates = pd.date_range("2026-03-02", periods=days, freq="D")  # starts on a Monday
    rng = np.random.default_rng(seed)
    values = [WEEKLY_PATTERN[d.dayofweek] + rng.normal(0, noise) for d in dates]
    return pd.Series(np.round(values).astype("int64"), index=dates)


def test_daily_counts_sums_groups_and_fills_missing_days():
    dates = pd.Series(["2026-03-01", "2026-03-01", "2026-03-03"])
    records = pd.Series([5, 7, 4])

    counts = daily_counts(dates, records)

    assert list(counts.index) == list(pd.date_range("2026-03-01", "2026-03-03"))
    assert list(counts) == [12, 0, 4]


def test_exact_weekly_pattern_is_forecast_exactly():
    counts = weekly_counts()
    result = forecast_next_day(counts)

    expected = WEEKLY_PATTERN[result.date.dayofweek]
    assert result.date == pd.Timestamp("2026-06-30")  # the day after the last count
    assert result.value == pytest.approx(expected, abs=1)
    assert result.scores.iloc[0] == pytest.approx(0, abs=1)
    assert result.low <= result.value <= result.high


def test_scores_every_method_and_picks_the_lowest_error():
    result = forecast_next_day(weekly_counts(noise=15))

    assert set(result.scores.index) == set(METHODS)
    assert result.method == result.scores.idxmin()
    assert len(result.backtest) == 56


def test_interval_widens_with_noise():
    quiet = forecast_next_day(weekly_counts(noise=2))
    noisy = forecast_next_day(weekly_counts(noise=40))
    assert (noisy.high - noisy.low) > (quiet.high - quiet.low)


def test_backtest_never_sees_the_day_it_forecasts():
    counts = weekly_counts(days=60)
    changed = counts.copy()
    changed.iloc[-1] = 10_000

    original = backtest(counts, same_weekday_last_week, days=14)
    after = backtest(changed, same_weekday_last_week, days=14)

    assert list(original["forecast"]) == list(after["forecast"])
    assert after["actual"].iloc[-1] == 10_000


def test_forecast_is_never_negative():
    counts = pd.Series(
        [0] * 60, index=pd.date_range("2026-03-02", periods=60, freq="D"), dtype="int64"
    )
    result = forecast_next_day(counts)
    assert result.value >= 0 and result.low >= 0


def test_too_little_history_returns_none():
    assert forecast_next_day(weekly_counts(days=35)) is None
