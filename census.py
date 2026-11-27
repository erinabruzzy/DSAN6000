"""Calculations over the curated DHS daily census (no Streamlit, so testable).

`report` arguments are the curated dhs_daily table as a pandas DataFrame with
`census_date` as datetime64 values, one row per day.
"""

import pandas as pd

# Household groups that together make up total individuals in shelter.
GROUPS = {
    "single_adults": "Single adults",
    "family_individuals": "Families with children",
    "adult_family_individuals": "Adult families",
}

# Each group split one level further: (group, subgroup, column).
SUBGROUPS = [
    ("Single adults", "Men", "single_adult_men"),
    ("Single adults", "Women", "single_adult_women"),
    ("Families with children", "Adults", "family_adults"),
    ("Families with children", "Children", "family_children"),
    ("Adult families", "Individuals", "adult_family_individuals"),
]

ALERT_COLUMNS = {"individuals": "Total individuals", **GROUPS}

# A day-over-day change is unusual when it is this many standard deviations
# from the mean change over the previous ALERT_WINDOW days. At 3 SD each series
# flags about 1% of days in the 2021-2026 history (2 SD flagged about 5%).
ALERT_Z = 3.0
ALERT_WINDOW = 90


def breakdown(row: pd.Series) -> pd.DataFrame:
    """One census day split into the SUBGROUPS, with each one's share of the total."""
    table = pd.DataFrame(SUBGROUPS, columns=["group", "subgroup", "column"])
    table["individuals"] = [int(row[column]) for column in table["column"]]
    table["share"] = table["individuals"] / int(row["individuals"])
    table["label"] = table["group"] + " · " + table["subgroup"]
    return table.drop(columns="column")


def unusual_changes(
    report: pd.DataFrame,
    as_of: pd.Timestamp,
    window: int = ALERT_WINDOW,
    threshold: float = ALERT_Z,
) -> pd.DataFrame:
    """Day-over-day changes on `as_of` that fall outside the recent normal range.

    The baseline is the `window` daily changes before `as_of`, so the day being
    judged never counts toward its own baseline. Returns one row per flagged
    series (empty if none, or if there is too little history to judge).
    """
    daily = report.set_index("census_date").sort_index()
    history = daily.loc[:as_of]
    # Changes are only comparable when consecutive rows are one day apart.
    # (.dt.days avoids a NumPy 2.5 deprecation warning from Timedelta comparisons.)
    consecutive = history.index.to_series().diff().dt.days == 1
    changes = history[list(ALERT_COLUMNS)].diff().where(consecutive, axis=0)
    if as_of not in changes.index or changes.loc[as_of].isna().all():
        return _empty_alerts()

    baseline = changes.iloc[:-1].tail(window)
    minimum_days = window // 3
    flagged = []
    for column, label in ALERT_COLUMNS.items():
        past = baseline[column].dropna()
        change = changes.at[as_of, column]
        spread = past.std()
        if len(past) < minimum_days or pd.isna(change) or not spread:
            continue
        z = (change - past.mean()) / spread
        if abs(z) >= threshold:
            flagged.append(
                {
                    "series": label,
                    "change": int(change),
                    "typical_change": float(past.abs().median()),
                    "z": float(z),
                }
            )
    return pd.DataFrame(flagged) if flagged else _empty_alerts()


def _empty_alerts() -> pd.DataFrame:
    return pd.DataFrame(columns=["series", "change", "typical_change", "z"])


# Look-back periods for changes_since, in days.
HORIZONS = {"previous day": 1, "a week earlier": 7, "four weeks earlier": 28}


def changes_since(
    report: pd.DataFrame, as_of: pd.Timestamp, horizons: dict[str, int] = HORIZONS
) -> pd.DataFrame:
    """Each ALERT_COLUMNS series on `as_of`, and its change since each horizon.

    A change is missing (NA) when the report has no row for that earlier date.
    """
    daily = report.set_index("census_date").sort_index()
    rows = []
    for column, label in ALERT_COLUMNS.items():
        row = {"series": label, "count": int(daily.at[as_of, column])}
        for name, days in horizons.items():
            earlier = as_of - pd.DateOffset(days=days)
            row[name] = (
                int(daily.at[as_of, column] - daily.at[earlier, column])
                if earlier in daily.index
                else pd.NA
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.astype({name: "Int64" for name in horizons})


def group_trend(
    report: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    monthly: bool = False,
) -> pd.DataFrame:
    """GROUPS over time in wide form: one row per date (or month), one column per group.

    With `monthly`, each value is the average daily count over the days reported
    in that month, and `days_reported` says how many days that was, so a partial
    month can be labeled.
    """
    daily = report.set_index("census_date").sort_index().loc[start:end, list(GROUPS)]
    if not monthly:
        return daily.rename_axis("date").reset_index()
    months = daily.groupby(daily.index.to_period("M"))
    trend = months.mean().round().astype("int64")
    trend["days_reported"] = months.size()
    trend.index = trend.index.to_timestamp()
    return trend.rename_axis("date").reset_index()
