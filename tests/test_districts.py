"""Tests for districts.py on a small synthetic district table."""

import os

import pandas as pd
import pytest

from districts import FACILITIES, FACILITY_TYPES, by_area, month_on_or_before, to_long

OCT = pd.Timestamp("2026-10-31")
SEP = pd.Timestamp("2026-09-30")


def row(date, borough, district, **counts):
    return {
        "report_date": date,
        "borough": borough,
        "community_district": district,
        **{column: counts.get(column, 0) for column in FACILITIES},
    }


TABLE = pd.DataFrame(
    [
        row(OCT, "Bronx", 1, adult_shelter=100, adult_hotel=50, family_shelter=300),
        row(OCT, "Bronx", 2, family_cluster=20, adult_family_hotel=30),
        row(OCT, "Queens", 1, family_hotel=600),
        row(OCT, "Westchester", None, family_shelter=10),
        row(SEP, "Bronx", 1, adult_shelter=999),
    ]
)


def test_month_on_or_before():
    assert month_on_or_before(TABLE, pd.Timestamp("2026-11-26")) == OCT
    assert month_on_or_before(TABLE, pd.Timestamp("2026-10-15")) == SEP
    assert month_on_or_before(TABLE, pd.Timestamp("2026-01-01")) is None


def test_boroughs_add_up_and_split_by_facility_type():
    wide = by_area(TABLE, OCT).set_index("area")

    assert list(wide.index) == ["Queens", "Bronx", "Westchester"]  # largest first
    assert wide.at["Bronx", "total"] == 500
    assert wide.at["Bronx", "Adult facilities"] == 150
    assert wide.at["Bronx", "Family with children facilities"] == 320
    assert wide.at["Bronx", "Adult family facilities"] == 30
    assert wide["share"].sum() == pytest.approx(1)
    # Hotels: adult_hotel 50 + adult_family_hotel 30 of 500.
    assert wide.at["Bronx", "hotel_share"] == pytest.approx(80 / 500)
    assert wide.at["Queens", "hotel_share"] == 1


def test_districts_within_a_borough_and_outside_the_city():
    bronx = by_area(TABLE, OCT, "Bronx")
    assert list(bronx["area"]) == ["District 1", "District 2"]
    assert list(bronx["total"]) == [450, 50]

    westchester = by_area(TABLE, OCT, "Westchester")
    assert list(westchester["area"]) == ["Westchester"]


def test_empty_area_has_no_hotel_share():
    table = pd.DataFrame([row(OCT, "Brooklyn", 10)])
    wide = by_area(table, OCT, "Brooklyn")
    assert wide.at[0, "total"] == 0 and pd.isna(wide.at[0, "hotel_share"])


def test_to_long_keeps_every_individual():
    wide = by_area(TABLE, OCT)
    long = to_long(wide)
    assert len(long) == len(wide) * len(FACILITY_TYPES)
    assert long["individuals"].sum() == wide["total"].sum()


@pytest.mark.skipif(
    "DHS_DISTRICT_TEST_PATH" not in os.environ or "DHS_DAILY_TEST_PATH" not in os.environ,
    reason="set DHS_DISTRICT_TEST_PATH and DHS_DAILY_TEST_PATH to check the real tables",
)
def test_real_month_totals_match_the_daily_report():
    districts = pd.read_parquet(os.environ["DHS_DISTRICT_TEST_PATH"])
    districts["report_date"] = pd.to_datetime(districts["report_date"])
    daily = pd.read_parquet(os.environ["DHS_DAILY_TEST_PATH"])
    daily["census_date"] = pd.to_datetime(daily["census_date"])
    daily = daily.set_index("census_date")["individuals"]

    for report_date in districts["report_date"].drop_duplicates().nlargest(12):
        if report_date in daily.index:
            total = by_area(districts, report_date)["total"].sum()
            assert total == pytest.approx(daily[report_date], rel=0.002)
