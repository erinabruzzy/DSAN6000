"""Tests for census.py on small synthetic reports with known answers."""

import os

import pandas as pd
import pytest

from census import GROUPS, SUBGROUPS, breakdown, group_trend, unusual_changes


def make_report(days: int = 120, start: str = "2026-01-01") -> pd.DataFrame:
    """A report whose counts change by exactly +1, -1, +1, ... each day.

    Subgroups add up to their groups and groups add up to the total, as in the
    real DHS report.
    """
    dates = pd.date_range(start, periods=days, freq="D")
    wobble = pd.Series([i % 2 for i in range(days)])
    report = pd.DataFrame(
        {
            "census_date": dates,
            "single_adult_men": 20_000 + wobble,
            "single_adult_women": 7_000 + wobble,
            "family_adults": 23_000 + wobble,
            "family_children": 27_000 + wobble,
            "adult_family_individuals": 5_000 + wobble,
        }
    )
    report["single_adults"] = report["single_adult_men"] + report["single_adult_women"]
    report["family_individuals"] = report["family_adults"] + report["family_children"]
    report["individuals"] = report[list(GROUPS)].sum(axis=1)
    return report


def test_breakdown_subgroups_cover_everyone():
    row = make_report().iloc[-1]
    table = breakdown(row)

    assert list(table["label"]) == [f"{g} · {s}" for g, s, _ in SUBGROUPS]
    assert table["individuals"].sum() == row["individuals"]
    assert table["share"].sum() == pytest.approx(1.0)


def test_no_alert_on_an_ordinary_day():
    report = make_report()
    assert unusual_changes(report, report["census_date"].iloc[-1]).empty


def test_alert_on_a_spike_names_the_series_and_direction():
    report = make_report()
    last = report.index[-1]
    report.loc[last, ["single_adult_men", "single_adults", "individuals"]] += 500

    alerts = unusual_changes(report, report.at[last, "census_date"])

    assert set(alerts["series"]) == {"Single adults", "Total individuals"}
    assert (alerts["change"] > 0).all()
    assert (alerts["z"] >= 3).all()


def test_spike_day_is_not_part_of_its_own_baseline():
    """A huge one-day drop would inflate the spread if it counted toward the baseline."""
    report = make_report()
    last = report.index[-1]
    report.loc[last, ["adult_family_individuals", "individuals"]] -= 40

    alerts = unusual_changes(report, report.at[last, "census_date"])

    assert "Adult families" in set(alerts["series"])


def test_no_alert_after_a_gap_in_dates():
    """A change across a missing day is not a day-over-day change."""
    report = make_report().drop(index=118)  # the day before the last
    last = report.index[-1]
    report.loc[last, ["single_adult_men", "single_adults", "individuals"]] += 500

    assert unusual_changes(report, report.at[last, "census_date"]).empty


def test_no_alert_without_enough_history():
    report = make_report(days=10)
    report.loc[report.index[-1], ["single_adults", "individuals"]] += 500
    assert unusual_changes(report, report["census_date"].iloc[-1]).empty


def test_monthly_trend_averages_and_counts_days():
    report = make_report(days=45)  # all of January, half of February

    trend = group_trend(report, monthly=True)

    assert list(trend["date"]) == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")]
    assert list(trend["days_reported"]) == [31, 14]
    january = report.loc[report["census_date"].dt.month == 1, "single_adults"]
    assert trend.at[0, "single_adults"] == round(january.mean())


def test_daily_trend_respects_the_date_window():
    report = make_report()
    start, end = pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-28")

    trend = group_trend(report, start, end)

    assert trend["date"].min() == start
    assert trend["date"].max() == end
    assert list(trend.columns) == ["date", *GROUPS]


REAL_TABLE = os.getenv("DHS_DAILY_TEST_PATH")


@pytest.mark.skipif(not REAL_TABLE, reason="set DHS_DAILY_TEST_PATH to a curated dhs_daily copy")
def test_real_report_groups_add_up():
    """The breakdown assumes the DHS groups and subgroups are exact partitions."""
    report = pd.read_parquet(REAL_TABLE)
    assert (report[list(GROUPS)].sum(axis=1) == report["individuals"]).all()
    for group_column in GROUPS:
        parts = [column for group, _, column in SUBGROUPS if GROUPS[group_column] == group]
        assert (report[parts].sum(axis=1) == report[group_column]).all()
