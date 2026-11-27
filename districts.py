"""Calculations over the curated dhs_district table (no Streamlit, so testable).

`table` arguments are curated/dhs_district as a pandas DataFrame with
`report_date` as datetime64 values: one row per community district per month,
counted on the month's last day. Rows outside the city (Westchester) have no
community district.

Facility types are where people are housed, not who they are: some adult
families live in family-with-children facilities, so these totals differ from
the household groups in the daily report even though the overall total matches.
"""

import pandas as pd

# Curated count column -> (facility type, kind of building).
FACILITIES = {
    "adult_shelter": ("Adult facilities", "Shelter"),
    "adult_hotel": ("Adult facilities", "Commercial hotel"),
    "family_shelter": ("Family with children facilities", "Shelter"),
    "family_hotel": ("Family with children facilities", "Commercial hotel"),
    "family_cluster": ("Family with children facilities", "Cluster"),
    "adult_family_shelter": ("Adult family facilities", "Shelter"),
    "adult_family_hotel": ("Adult family facilities", "Commercial hotel"),
}
FACILITY_TYPES = list(dict.fromkeys(facility_type for facility_type, _ in FACILITIES.values()))
HOTEL_COLUMNS = [column for column, (_, kind) in FACILITIES.items() if kind == "Commercial hotel"]


def month_on_or_before(table: pd.DataFrame, day: pd.Timestamp) -> pd.Timestamp | None:
    """The latest report date on or before `day`, or None if there is none."""
    dates = table.loc[table["report_date"] <= day, "report_date"]
    return dates.max() if not dates.empty else None


def by_area(
    table: pd.DataFrame, report_date: pd.Timestamp, borough: str | None = None
) -> pd.DataFrame:
    """Individuals per area on `report_date`, largest first.

    Areas are boroughs, or with `borough`, that borough's community districts.
    Each row has every FACILITIES column, a total per FACILITY_TYPES entry,
    `total`, `share` of the areas' combined total, and `hotel_share`.
    """
    month = table.loc[table["report_date"] == report_date]
    if borough is None:
        month = month.assign(area=month["borough"])
    else:
        month = month.loc[month["borough"] == borough]
        district = month["community_district"]
        month = month.assign(
            area=district.map(lambda value: f"District {int(value)}", na_action="ignore").fillna(
                month["borough"]
            )
        )
    wide = month.groupby("area", as_index=False)[list(FACILITIES)].sum()
    for facility_type in FACILITY_TYPES:
        columns = [c for c, (kind, _) in FACILITIES.items() if kind == facility_type]
        wide[facility_type] = wide[columns].sum(axis=1)
    wide["total"] = wide[FACILITY_TYPES].sum(axis=1)
    overall = wide["total"].sum()
    wide["share"] = wide["total"] / overall if overall else 0.0
    wide["hotel_share"] = (wide[HOTEL_COLUMNS].sum(axis=1) / wide["total"]).where(
        wide["total"] > 0
    )
    return wide.sort_values(["total", "area"], ascending=[False, True]).reset_index(drop=True)


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """by_area output as one row per (area, facility type) for a stacked bar chart."""
    return wide.melt(
        id_vars=["area", "total", "share", "hotel_share"],
        value_vars=FACILITY_TYPES,
        var_name="facility_type",
        value_name="individuals",
    )
