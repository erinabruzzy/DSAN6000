import os
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st
from pyspark.errors import AnalysisException
from pyspark.sql import functions as F

from app_spark import BUCKET, get_spark, load_or_none
from census import (
    ALERT_WINDOW,
    ALERT_Z,
    GROUPS,
    SUBGROUPS,
    breakdown,
    changes_since,
    group_trend,
    unusual_changes,
)
from districts import (
    FACILITIES,
    FACILITY_TYPES,
    by_area,
    month_on_or_before,
    to_long,
)
from forecast import (
    INTERVAL,
    MIN_TRAINING_DAYS,
    NextDayForecast,
    daily_counts,
    forecast_next_day,
    latest_day_summary,
)
from records import (
    LIST_COLUMNS,
    RECORD_COLUMNS,
    decode_photo,
    detail_sections,
    escape_markdown,
    format_value,
)

st.set_page_config(page_title="NYC Shelter Snapshot", layout="wide")

# Curated Parquet tables written by the jobs in etl/.
DEFAULT_INTAKE_PATH = f"s3a://{BUCKET}/curated/intakes"  # intake_to_parquet.py
DEFAULT_DETAILS_PATH = f"s3a://{BUCKET}/curated/intake_details"  # intake_to_parquet.py
DEFAULT_REPORT_PATH = f"s3a://{BUCKET}/curated/dhs_daily"  # dhs_daily_ingest.py
DEFAULT_DISTRICT_PATH = f"s3a://{BUCKET}/curated/dhs_district"  # dhs_district_ingest.py
ALL_HOUSEHOLDS = "All household types"
REPORT_COLUMNS = list(
    dict.fromkeys(
        ["census_date", "individuals", "adults", "children", *GROUPS]
        + [column for _, _, column in SUBGROUPS]
    )
)
# Streamlit's chart theme sets its own axis labelExpr, which overrides an axis
# `format`, so numeric axes set labelExpr themselves (60000 -> "60,000").
COUNT_LABELS = "format(datum.value, ',')"
# DHS publishes each census about two days later; warn when it is later than this.
STALE_AFTER_DAYS = 4

# Slots 1-3 of the validated categorical palette (blue, orange, aqua), each
# stepped for its theme, plus the text and hover-rule colors for that theme.
THEME_COLORS = {
    "light": {
        "series": ["#257de0", "#f47230", "#00b894"],
        "text": "#31333f",
        "rule": "#8c8c8c",
        "surface": "#ffffff",
    },
    "dark": {
        "series": ["#338ced", "#e76624", "#00a888"],
        "text": "#fafafa",
        "rule": "#7a7a7a",
        "surface": "#0e1117",
    },
}
INTAKE_COLUMNS = [
    "synthetic_data",
    "intake_date",
    "program_type",
    "household_type",
    "age",
    "children_count",
]


@st.cache_data(ttl=3600, show_spinner="Loading the DHS daily report...")
def load_daily_report(path: str) -> pd.DataFrame:
    return get_spark().read.parquet(path).select(*REPORT_COLUMNS).toPandas()


@st.cache_data(ttl=3600, show_spinner="Aggregating synthetic intake records...")
def load_intake_summary(path: str) -> tuple[pd.DataFrame, int]:
    spark = get_spark()
    # Parquet is columnar, so selecting six columns reads only those columns.
    records = spark.read.parquet(path).select(*INTAKE_COLUMNS)
    summary = (
        records.groupBy(
            "synthetic_data", "intake_date", "program_type", "household_type"
        )
        .agg(
            F.count(F.lit(1)).alias("intake_records"),
            F.count("age").alias("age_count"),
            F.sum(F.col("age").cast("long")).alias("age_sum"),
            F.sum(F.col("children_count").cast("long")).alias(
                "children_recorded"
            ),
        )
        .toPandas()
    )

    is_synthetic = summary["synthetic_data"].fillna(False).astype(bool)
    excluded_records = int(summary.loc[~is_synthetic, "intake_records"].sum() or 0)
    synthetic_summary = summary.loc[is_synthetic].drop(columns="synthetic_data").copy()
    numeric_columns = [
        "intake_records",
        "age_count",
        "age_sum",
        "children_recorded",
    ]
    synthetic_summary.loc[:, numeric_columns] = synthetic_summary.loc[
        :, numeric_columns
    ].fillna(0)
    synthetic_summary["intake_date"] = pd.to_datetime(
        synthetic_summary["intake_date"], errors="coerce"
    )
    return synthetic_summary, excluded_records


@st.cache_data(ttl=3600, show_spinner="Loading the census by community district...")
def load_district_report(path: str) -> pd.DataFrame:
    table = get_spark().read.parquet(path).toPandas()
    table["report_date"] = pd.to_datetime(table["report_date"])
    return table


def _read_partition(path: str, partition: str):
    """Read one partition folder of a curated table, or None if it doesn't exist.

    Reading the folder directly (with basePath) reads only that partition,
    whatever type Spark would infer for the partition values.
    """
    try:
        return get_spark().read.option("basePath", path).parquet(f"{path}/{partition}")
    except AnalysisException as error:
        if error.getCondition() != "PATH_NOT_FOUND":
            raise
        return None


@st.cache_data(ttl=3600, max_entries=20, show_spinner="Loading that day's intake records...")
def load_day_records(path: str, day: date) -> pd.DataFrame:
    """Every synthetic intake record on `day` (curated/intakes is partitioned by month)."""
    month = _read_partition(path, f"intake_month={day:%Y-%m}")
    if month is None:
        return pd.DataFrame(columns=RECORD_COLUMNS)
    return (
        month.filter((F.col("intake_date") == F.lit(day)) & F.col("synthetic_data"))
        .select(*RECORD_COLUMNS)
        .toPandas()
    )


@st.cache_data(ttl=3600, max_entries=50, show_spinner="Loading the intake note and photo...")
def load_record_details(path: str, day: date, intake_id: str) -> dict | None:
    """The note and photo for one record (curated/intake_details is partitioned by day)."""
    records = _read_partition(path, f"intake_date={day:%Y-%m-%d}")
    if records is None:
        return None
    rows = (
        records.filter(F.col("intake_id") == F.lit(intake_id))
        .select("intake_note", "intake_photo_b64")
        .limit(1)
        .collect()
    )
    return rows[0].asDict() if rows else None


def load_or_stop(loader, path: str, etl_script: str):
    """Run a loader, or explain which ETL job to run if its table is missing."""
    result = load_or_none(loader, path)
    if result is None:
        st.error(
            f"No curated table at `{path}`. Run "
            f"`uv run python etl/{etl_script}` on the Spark master first "
            "(see README)."
        )
        st.stop()
    return result


def format_count(value: object) -> str:
    if pd.isna(value):
        return "Unavailable"
    return f"{int(value):,}"


def theme_colors() -> dict:
    return THEME_COLORS.get(st.context.theme.type, THEME_COLORS["light"])


def breakdown_chart(table: pd.DataFrame, colors: dict) -> alt.LayerChart:
    """Horizontal bars, one per subgroup, labeled with their counts."""
    base = alt.Chart(table).encode(
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=300)),
        x=alt.X(
            "individuals:Q",
            title="Individuals in shelter",
            axis=alt.Axis(labelExpr=COUNT_LABELS, tickCount=5),
            scale=alt.Scale(nice=True),
        ),
        tooltip=[
            alt.Tooltip("group:N", title="Group"),
            alt.Tooltip("subgroup:N", title="Subgroup"),
            alt.Tooltip("individuals:Q", title="Individuals", format=","),
            alt.Tooltip("share:Q", title="Share of everyone in shelter", format=".1%"),
        ],
    )
    bars = base.mark_bar(
        color=colors["series"][0], cornerRadiusEnd=4, height={"band": 0.7}
    )
    labels = base.mark_text(align="left", dx=6, color=colors["text"]).encode(
        text=alt.Text("individuals:Q", format=",")
    )
    return (bars + labels).properties(height=alt.Step(40))


def sync_click(chart_key: str, param: str, state_key: str, options: list, default) -> None:
    """Copy a chart click into a selectbox's state before the selectbox renders.

    Only a new click is copied, so the selectbox stays the single source of
    truth and works without a pointer. Clearing the click picks `default`.
    """
    clicked = st.session_state.get(chart_key, {}).get("selection", {}).get(param, [])
    value = clicked[0][param] if clicked else None
    last_key = f"{state_key}_last_click"
    if value != st.session_state.get(last_key):
        st.session_state[last_key] = value
        st.session_state[state_key] = value or default
    if st.session_state.get(state_key) not in options:
        st.session_state[state_key] = default


def area_chart(wide: pd.DataFrame, selected: str | None, colors: dict) -> alt.LayerChart:
    """Stacked bars of individuals per area by facility type, totals at the ends.

    With `selected`, the bars are clickable (selection "area") and the other
    areas are dimmed.
    """
    long = to_long(wide).assign(
        facility_order=lambda frame: frame["facility_type"].map(FACILITY_TYPES.index),
        highlighted=lambda frame: True if selected is None else frame["area"] == selected,
    )
    y = alt.Y("area:N", sort=list(wide["area"]), title=None, axis=alt.Axis(labelLimit=300))
    bars = (
        alt.Chart(long)
        .mark_bar(
            height={"band": 0.7},
            stroke=colors["surface"],
            strokeWidth=1,
            cursor="default" if selected is None else "pointer",
        )
        .encode(
            y=y,
            x=alt.X(
                "individuals:Q",
                title="Individuals in shelter",
                stack="zero",
                axis=alt.Axis(labelExpr=COUNT_LABELS, tickCount=5),
            ),
            color=alt.Color(
                "facility_type:N",
                scale=alt.Scale(domain=FACILITY_TYPES, range=colors["series"]),
                legend=alt.Legend(orient="top", title=None, labelLimit=0),
            ),
            order=alt.Order("facility_order:Q"),
            opacity=alt.condition("datum.highlighted", alt.value(1), alt.value(0.35)),
            tooltip=[
                alt.Tooltip("area:N", title="Area"),
                alt.Tooltip("facility_type:N", title="Facility type"),
                alt.Tooltip("individuals:Q", title="Individuals", format=","),
                alt.Tooltip("total:Q", title="Area total", format=","),
                alt.Tooltip("share:Q", title="Share of total", format=".1%"),
                alt.Tooltip("hotel_share:Q", title="Area share in commercial hotels", format=".0%"),
            ],
        )
    )
    if selected is not None:
        bars = bars.add_params(alt.selection_point(name="area", fields=["area"]))
    labels = (
        alt.Chart(wide)
        .mark_text(align="left", dx=6, color=colors["text"])
        .encode(y=y, x=alt.X("total:Q"), text=alt.Text("total:Q", format=","))
    )
    return (bars + labels).properties(
        height=alt.Step(34), padding={"left": 5, "top": 5, "bottom": 5, "right": 50}
    )


def household_chart(table: pd.DataFrame, selected: str, colors: dict) -> alt.LayerChart:
    """Clickable bars of intake records per household type; `selected` is highlighted."""
    table = table.assign(highlighted=(table["household_type"] == selected) | (selected == ALL_HOUSEHOLDS))
    click = alt.selection_point(name="household_type", fields=["household_type"])
    base = alt.Chart(table).encode(
        y=alt.Y("household_type:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)),
        x=alt.X(
            "intake_records:Q",
            title="Intake records",
            axis=alt.Axis(labelExpr=COUNT_LABELS, tickCount=5),
        ),
        tooltip=[
            alt.Tooltip("household_type:N", title="Household type"),
            alt.Tooltip("intake_records:Q", title="Intake records", format=","),
            alt.Tooltip("average_recorded_age:Q", title="Average recorded age", format=".1f"),
            alt.Tooltip("children_recorded:Q", title="Children recorded", format=","),
        ],
    )
    bars = (
        base.mark_bar(color=colors["series"][0], cornerRadiusEnd=4, height={"band": 0.7}, cursor="pointer")
        .encode(opacity=alt.condition("datum.highlighted", alt.value(1), alt.value(0.35)))
        .add_params(click)
    )
    labels = base.mark_text(align="left", dx=6, color=colors["text"]).encode(
        text=alt.Text("intake_records:Q", format=",")
    )
    return (bars + labels).properties(height=alt.Step(40))


@st.cache_data(ttl=3600, show_spinner="Backtesting the intake forecast...")
def cached_forecast(counts: pd.Series) -> NextDayForecast | None:
    return forecast_next_day(counts)


def forecast_chart(result: NextDayForecast, colors: dict) -> alt.LayerChart:
    """Backtest days (actual vs one-day-ahead forecast) plus the next day with its range."""
    names = ["Actual intakes", "Forecast from the days before"]
    wide = result.backtest.copy()
    upcoming = pd.DataFrame(
        {
            "date": [result.date],
            "forecast": [result.value],
            "low": [result.low],
            "high": [result.high],
        }
    )
    long = pd.concat(
        [
            wide.assign(series=names[0], count=wide["actual"]),
            wide.assign(series=names[1], count=wide["forecast"]),
            upcoming.assign(series=names[1], count=upcoming["forecast"]),
        ]
    )[["date", "series", "count"]]

    x = alt.X("date:T", title=None, axis=alt.Axis(format="%b %d", labelOverlap=True))
    y = alt.Y(
        "count:Q",
        title="Intakes per day",
        scale=alt.Scale(zero=False),
        axis=alt.Axis(labelExpr=COUNT_LABELS),
    )
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=names, range=colors["series"][:2]),
        legend=alt.Legend(orient="top", title=None, labelLimit=0),
    )
    hover = alt.selection_point(
        fields=["date"], nearest=True, on="pointerover", clear="pointerout", empty=False
    )
    shown_on_hover = alt.condition(hover, alt.value(1), alt.value(0))

    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(x=x, y=y, color=color)
    points = (
        alt.Chart(long)
        .mark_point(filled=True, size=70, stroke=colors["surface"], strokeWidth=2)
        .encode(x=x, y=y, color=color, opacity=shown_on_hover)
    )
    # Tooltip text is preformatted per row: a backtest day has an actual count
    # and an error, the next day has no actual yet but has a range.
    hover_rows = pd.DataFrame(
        {
            "date": [*wide["date"], result.date],
            "actual_text": [f"{value:,}" for value in wide["actual"]] + ["Not known yet"],
            "forecast_text": [f"{value:,.0f}" for value in wide["forecast"]]
            + [f"{result.value:,.0f}"],
            "note_text": [
                f"Off by {abs(actual - forecast):,.0f}"
                for actual, forecast in zip(wide["actual"], wide["forecast"])
            ]
            + [f"{INTERVAL:.0%} range {result.low:,.0f} – {result.high:,.0f}"],
        }
    )
    rule = (
        alt.Chart(hover_rows)
        .mark_rule(color=colors["rule"], strokeWidth=1)
        .encode(
            x=x,
            opacity=shown_on_hover,
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%b %d, %Y"),
                alt.Tooltip("actual_text:N", title=names[0]),
                alt.Tooltip("forecast_text:N", title="Forecast"),
                alt.Tooltip("note_text:N", title="Detail"),
            ],
        )
        .add_params(hover)
    )
    # The next day: its range as a vertical bar, the forecast as a ringed dot.
    next_range = (
        alt.Chart(upcoming)
        .mark_rule(color=colors["series"][1], strokeWidth=2)
        .encode(x=x, y="low:Q", y2="high:Q")
    )
    next_point = (
        alt.Chart(upcoming)
        .mark_point(
            filled=True,
            size=90,
            color=colors["series"][1],
            stroke=colors["surface"],
            strokeWidth=2,
        )
        .encode(x=x, y=alt.Y("forecast:Q"))
    )
    next_label = (
        alt.Chart(upcoming)
        .mark_text(align="left", dx=10, color=colors["text"])
        .encode(
            x=x,
            y=alt.Y("forecast:Q"),
            text=alt.Text("forecast:Q", format=",.0f"),
        )
    )
    # The hover rule goes last so marks drawn above it can't swallow the pointer.
    return alt.layer(lines, next_range, points, next_point, next_label, rule).properties(
        height=320, padding={"left": 5, "top": 5, "bottom": 5, "right": 40}
    )


def trend_chart(wide: pd.DataFrame, colors: dict, monthly: bool) -> alt.LayerChart:
    """One line per household group, with a hover rule listing every group's value."""
    names = list(GROUPS.values())
    long = wide.melt(
        id_vars="date", value_vars=list(GROUPS), var_name="column", value_name="count"
    )
    long["group"] = long["column"].map(GROUPS)

    x = alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y", labelOverlap=True))
    y = alt.Y(
        "count:Q",
        title="Average daily individuals" if monthly else "Individuals",
        axis=alt.Axis(labelExpr=COUNT_LABELS),
    )
    color = alt.Color(
        "group:N",
        scale=alt.Scale(domain=names, range=colors["series"]),
        legend=alt.Legend(orient="top", title=None),
    )
    hover = alt.selection_point(
        fields=["date"], nearest=True, on="pointerover", clear="pointerout", empty=False
    )
    shown_on_hover = alt.condition(hover, alt.value(1), alt.value(0))

    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(x=x, y=y, color=color)
    points = (
        alt.Chart(long)
        .mark_point(filled=True, size=70, stroke=colors["surface"], strokeWidth=2)
        .encode(x=x, y=y, color=color, opacity=shown_on_hover)
    )
    tooltip = [
        alt.Tooltip(
            "date:T",
            title="Month" if monthly else "Date",
            format="%B %Y" if monthly else "%b %d, %Y",
        ),
        *[
            alt.Tooltip(f"{column}:Q", title=name, format=",")
            for column, name in GROUPS.items()
        ],
    ]
    if monthly:
        tooltip.append(alt.Tooltip("days_reported:Q", title="Days reported"))
    rule = (
        alt.Chart(wide)
        .mark_rule(color=colors["rule"], strokeWidth=1)
        .encode(x=x, opacity=shown_on_hover, tooltip=tooltip)
        .add_params(hover)
    )
    # Name each line at its last point, so identity never rests on color alone.
    end_labels = (
        alt.Chart(long.loc[long["date"] == long["date"].max()])
        .mark_text(align="left", dx=8, color=colors["text"])
        .encode(x=x, y=y, text="group:N")
    )
    # The hover rule goes last so marks drawn above it can't swallow the pointer.
    return alt.layer(lines, points, end_labels, rule).properties(
        height=380, padding={"left": 5, "top": 5, "bottom": 5, "right": 170}
    )


st.title("NYC Shelter Population Snapshot")
st.caption(
    "Synthetic data only. DHS census totals describe reported shelter population; "
    "intake statistics describe intake records, not current residents."
)

spark = get_spark()
st.caption(f"Spark master: `{spark.sparkContext.master}`")

report_path = os.getenv("DHS_DAILY_PATH", DEFAULT_REPORT_PATH)
report = load_or_stop(load_daily_report, report_path, "dhs_daily_ingest.py")
report["census_date"] = pd.to_datetime(report["census_date"], errors="coerce")
report = report.dropna(subset=["census_date"]).sort_values("census_date")
if report.empty:
    st.error("The DHS daily report contains no usable census dates.")
    st.stop()

report_dates = report["census_date"].dt.date.tolist()
selected_report_date = st.sidebar.selectbox(
    "DHS census date",
    report_dates,
    index=len(report_dates) - 1,
    format_func=lambda value: value.strftime("%B %d, %Y"),
)
selected_report = report.loc[
    report["census_date"].dt.date == selected_report_date
].iloc[-1]
selected_day = pd.Timestamp(selected_report["census_date"])
colors = theme_colors()

latest_day = report["census_date"].max()
days_behind = (pd.Timestamp.today().normalize() - latest_day).days
if days_behind > STALE_AFTER_DAYS:
    st.warning(
        f"The newest census in the table is from {latest_day:%B %d, %Y}, "
        f"{days_behind} days ago. DHS usually publishes within two days, so check "
        "that `etl/dhs_daily_ingest.py` is still running.",
        icon=":material/schedule:",
    )

st.subheader("Latest data at a glance")
st.caption(
    "The newest day in each dataset, whatever dates the sidebar selects. Both "
    "sources are daily, so there is nothing finer than a day: DHS publishes each "
    "census about two days later, and intake records have no time of day."
)
intake_path = os.getenv("INTAKE_PARQUET_PATH", DEFAULT_INTAKE_PATH)
census_box, intake_box = st.columns([3, 2])
with census_box.container(border=True):
    latest_changes = changes_since(report, latest_day)
    total = latest_changes.iloc[0]
    st.markdown(f"**DHS census, {latest_day:%A, %B %d, %Y}**")
    st.metric(
        "Individuals in shelter",
        f"{total['count']:,}",
        None if pd.isna(total["previous day"]) else f"{total['previous day']:+,} vs previous day",
        delta_color="off",
    )
    st.dataframe(
        latest_changes,
        column_config={
            "series": "Group",
            "count": st.column_config.NumberColumn("Individuals", format="localized"),
            "previous day": st.column_config.NumberColumn("Change: 1 day", format="%+d"),
            "a week earlier": st.column_config.NumberColumn("7 days", format="%+d"),
            "four weeks earlier": st.column_config.NumberColumn("28 days", format="%+d"),
        },
        hide_index=True,
        width="stretch",
    )
    st.caption("Changes since 1, 7, and 28 days earlier; blank if DHS has no report for that day.")
with intake_box.container(border=True):
    glance_intakes = load_or_none(load_intake_summary, intake_path)
    glance_dated = (
        glance_intakes[0].dropna(subset=["intake_date"]) if glance_intakes else pd.DataFrame()
    )
    if glance_dated.empty:
        st.markdown("**Synthetic intake records**")
        st.caption("No intake records yet; see the intake section below.")
    else:
        glance_counts = daily_counts(glance_dated["intake_date"], glance_dated["intake_records"])
        newest = latest_day_summary(glance_counts)
        st.markdown(f"**Synthetic intake records, {newest.date:%A, %B %d, %Y}**")
        st.metric(
            "Intake records",
            f"{newest.count:,}",
            None
            if newest.weekday_average is None
            else f"{newest.count - newest.weekday_average:+,.0f} vs the last 4 {newest.date:%A}s",
            delta_color="off",
        )
        lines = []
        if newest.weekday_average is not None:
            lines.append(
                f"The last 4 {newest.date:%A}s averaged {newest.weekday_average:,.0f} intakes."
            )
        upcoming = cached_forecast(glance_counts)
        if upcoming is not None:
            lines.append(
                f"Forecast for {upcoming.date:%A, %B %d}: **{upcoming.value:,.0f}** "
                f"({INTERVAL:.0%} range {upcoming.low:,.0f} – {upcoming.high:,.0f}); "
                "details in the intake section below."
            )
        st.markdown("  \n".join(lines))

st.subheader(f"DHS reported population — {selected_report_date:%B %d, %Y}")
previous_rows = report.loc[report["census_date"] == selected_day - pd.DateOffset(days=1)]
previous_report = previous_rows.iloc[-1] if not previous_rows.empty else None
population_columns = st.columns(3)
for column, (field, label) in zip(
    population_columns,
    [("individuals", "Individuals in shelter"), ("adults", "Adults"), ("children", "Children")],
):
    delta = None
    if previous_report is not None:
        delta = f"{int(selected_report[field] - previous_report[field]):+,} vs previous day"
    # Neutral delta color: the arrow shows direction without judging it.
    column.metric(label, format_count(selected_report[field]), delta, delta_color="off")

st.markdown("**Unusual day-over-day changes**")
alerts = unusual_changes(report, selected_day)
if alerts.empty:
    st.caption(
        f"None on this date. A change is flagged when it is {ALERT_Z:g} or more "
        "standard deviations from the average daily change over the previous "
        f"{ALERT_WINDOW} days."
    )
for alert in alerts.itertuples():
    direction = "rose" if alert.change > 0 else "fell"
    st.warning(
        f"**{alert.series}** {direction} by {abs(alert.change):,} from the previous "
        f"day. A typical daily change is about {alert.typical_change:,.0f} "
        f"({alert.z:+.1f} standard deviations).",
        icon=":material/warning:",
    )

st.markdown("**Who is in shelter**")
population_breakdown = breakdown(selected_report)
st.altair_chart(breakdown_chart(population_breakdown, colors), width="stretch")
with st.expander("Table view"):
    st.dataframe(
        population_breakdown[["group", "subgroup", "individuals", "share"]],
        column_config={
            "group": "Group",
            "subgroup": "Subgroup",
            "individuals": st.column_config.NumberColumn("Individuals", format="localized"),
            "share": st.column_config.NumberColumn("Share of total", format="percent"),
        },
        hide_index=True,
        width="stretch",
    )

st.markdown("**Household groups over time**")
trend_view = st.segmented_control(
    "Trend view",
    ["Daily, last 12 months", "Monthly average, all years"],
    default="Daily, last 12 months",
    label_visibility="collapsed",
)
monthly = trend_view == "Monthly average, all years"
if monthly:
    trend = group_trend(report, end=selected_day, monthly=True)
    st.caption(
        "Average daily count in each month through the selected census date. "
        "The latest month averages only the days reported so far."
    )
else:
    # DateOffset avoids a NumPy 2.5 deprecation warning raised by pd.Timedelta in pandas 2.3.
    trend = group_trend(report, selected_day - pd.DateOffset(days=365), selected_day)
    st.caption("Daily counts during the 12 months through the selected census date.")
st.altair_chart(trend_chart(trend, colors, monthly), width="stretch")
with st.expander("Table view"):
    st.dataframe(
        trend.sort_values("date", ascending=False),
        column_config={
            "date": st.column_config.DateColumn(
                "Month" if monthly else "Date",
                format="MMMM YYYY" if monthly else "MMM D, YYYY",
            ),
            **{
                column: st.column_config.NumberColumn(name, format="localized")
                for column, name in GROUPS.items()
            },
            "days_reported": "Days reported",
        },
        hide_index=True,
        width="stretch",
    )

st.markdown("**Where people are sheltered**")
district_path = os.getenv("DHS_DISTRICT_PATH", DEFAULT_DISTRICT_PATH)
district_report = load_or_none(load_district_report, district_path)
district_month = (
    None if district_report is None else month_on_or_before(district_report, selected_day)
)
if district_report is None:
    st.info(
        f"No census by community district at `{district_path}` yet. Run "
        "`uv run python etl/dhs_district_ingest.py` on the Spark master (see README)."
    )
elif district_month is None:
    st.info(f"DHS has no census by community district on or before {selected_day:%B %d, %Y}.")
else:
    st.caption(
        f"DHS publishes this census monthly. Showing {district_month:%B %d, %Y}, the latest "
        "on or before the selected census date; its total matches the daily report for "
        "that day. Facility types describe where people are housed, not their household "
        "group: some adult families live in family-with-children facilities."
    )
    boroughs = by_area(district_report, district_month)
    borough_options = list(boroughs["area"])
    sync_click("borough_chart", "area", "drill_borough", borough_options, borough_options[0])
    st.altair_chart(
        area_chart(boroughs, st.session_state["drill_borough"], colors),
        width="stretch",
        on_select="rerun",
        selection_mode="area",
        key="borough_chart",
    )
    st.caption("Click a borough to see its community districts.")
    district_borough = st.selectbox("Borough", borough_options, key="drill_borough")
    district_rows = by_area(district_report, district_month, district_borough)
    st.markdown(f"**{district_borough} by community district**")
    st.altair_chart(area_chart(district_rows, None, colors), width="stretch")
    with st.expander("Table view"):
        facility_labels = {
            column: f"{facility_type.removesuffix(' facilities')} · {kind}"
            for column, (facility_type, kind) in FACILITIES.items()
        }
        st.dataframe(
            pd.concat(
                [
                    boroughs.assign(level="Borough"),
                    district_rows.assign(level=f"{district_borough} district"),
                ]
            )[["level", "area", "total", "share", "hotel_share", *FACILITIES]],
            column_config={
                "level": "Level",
                "area": "Area",
                "total": st.column_config.NumberColumn("Individuals", format="localized"),
                "share": st.column_config.NumberColumn("Share of level", format="percent"),
                "hotel_share": st.column_config.NumberColumn(
                    "In commercial hotels", format="percent"
                ),
                **{
                    column: st.column_config.NumberColumn(label, format="localized")
                    for column, label in facility_labels.items()
                },
            },
            hide_index=True,
            width="stretch",
        )

st.divider()
st.subheader("Synthetic intake-record profile")
intakes, excluded_records = load_or_stop(
    load_intake_summary, intake_path, "intake_to_parquet.py"
)
undated_records = int(
    intakes.loc[intakes["intake_date"].isna(), "intake_records"].sum()
)
valid_intakes = intakes.dropna(subset=["intake_date"]).copy()
if valid_intakes.empty:
    st.info("There are no synthetic intake records with a usable intake date.")
    st.stop()

st.markdown("**Next-day intake forecast**")
intake_counts = daily_counts(valid_intakes["intake_date"], valid_intakes["intake_records"])
next_day = cached_forecast(intake_counts)
if next_day is None:
    st.info(
        f"The forecast needs at least {MIN_TRAINING_DAYS + 14} days of intake "
        f"records; there are {len(intake_counts)}."
    )
else:
    forecast_columns = st.columns(3)
    forecast_columns[0].metric(
        f"Forecast for {next_day.date:%B %d, %Y}", f"{next_day.value:,.0f} intakes"
    )
    forecast_columns[1].metric(
        f"{INTERVAL:.0%} range", f"{next_day.low:,.0f} – {next_day.high:,.0f}"
    )
    forecast_columns[2].metric(
        f"Average error, last {len(next_day.backtest)} days",
        f"±{next_day.scores[next_day.method]:,.1f}",
    )
    st.caption(
        "Intake records have a date but no time of day, so this forecasts the day "
        f"after the latest intake ({intake_counts.index[-1]:%B %d, %Y}), using all "
        f"synthetic records regardless of the filters. Method: **{next_day.method}**, "
        "which had the lowest average error when each of the last "
        f"{len(next_day.backtest)} days was forecast from only the days before it. "
        f"The range covers {INTERVAL:.0%} of those errors."
    )
    st.altair_chart(forecast_chart(next_day, colors), width="stretch")
    with st.expander("How the methods compared"):
        st.dataframe(
            next_day.scores.rename("Average error (intakes per day)")
            .rename_axis("Method")
            .reset_index(),
            column_config={
                "Average error (intakes per day)": st.column_config.NumberColumn(
                    format="%.1f"
                )
            },
            hide_index=True,
            width="stretch",
        )
        st.dataframe(
            next_day.backtest.sort_values("date", ascending=False),
            column_config={
                "date": st.column_config.DateColumn("Date", format="MMM D, YYYY"),
                "actual": "Actual intakes",
                "forecast": st.column_config.NumberColumn(
                    "Forecast from the days before", format="%.0f"
                ),
            },
            hide_index=True,
            width="stretch",
        )

minimum_date = valid_intakes["intake_date"].min().date()
maximum_date = valid_intakes["intake_date"].max().date()
default_start = max(minimum_date, maximum_date - timedelta(days=29))
selected_date_range = st.sidebar.date_input(
    "Intake date range",
    value=(default_start, maximum_date),
    min_value=minimum_date,
    max_value=maximum_date,
)
if isinstance(selected_date_range, tuple):
    start_date, end_date = selected_date_range
else:
    start_date = end_date = selected_date_range

program_options = sorted(
    value for value in valid_intakes["program_type"].dropna().unique()
)
selected_program = st.sidebar.selectbox(
    "Program type", ["All programs", *program_options]
)

filtered_intakes = valid_intakes.loc[
    valid_intakes["intake_date"].dt.date.between(start_date, end_date)
]
if selected_program != "All programs":
    filtered_intakes = filtered_intakes.loc[
        filtered_intakes["program_type"] == selected_program
    ]
if filtered_intakes.empty:
    st.info("No synthetic intake records match these filters.")
    st.stop()

intake_count = int(filtered_intakes["intake_records"].sum())
age_count = int(filtered_intakes["age_count"].sum())
age_sum = int(filtered_intakes["age_sum"].sum())
children_count = int(filtered_intakes["children_recorded"].sum())
average_age = age_sum / age_count if age_count else None

intake_columns = st.columns(3)
intake_columns[0].metric("Intake records", format_count(intake_count))
intake_columns[1].metric(
    "Average recorded age",
    f"{average_age:.1f}" if average_age is not None else "Unavailable",
)
intake_columns[2].metric(
    "Children recorded across intakes", format_count(children_count)
)

if excluded_records:
    st.warning(
        f"{excluded_records:,} records without `synthetic_data = true` were excluded."
    )
if undated_records:
    st.caption(
        f"{undated_records:,} synthetic intake records with missing dates are excluded "
        "from the date-filtered statistics."
    )

by_household = (
    filtered_intakes.assign(
        household_type=filtered_intakes["household_type"].fillna("Not reported")
    )
    .groupby("household_type", as_index=False)
    .agg(
        intake_records=("intake_records", "sum"),
        age_count=("age_count", "sum"),
        age_sum=("age_sum", "sum"),
        children_recorded=("children_recorded", "sum"),
    )
)
by_household["average_recorded_age"] = by_household["age_sum"].div(
    by_household["age_count"].where(by_household["age_count"] > 0)
)
# A click on a bar picks the household type for the record list below. The
# click is copied into the selectbox (only when it changes), so the selectbox
# is the single source of truth and works without a pointer.
household_options = [ALL_HOUSEHOLDS, *sorted(by_household["household_type"])]
sync_click("household_chart", "household_type", "drill_household", household_options, ALL_HOUSEHOLDS)

st.altair_chart(
    household_chart(by_household, st.session_state["drill_household"], colors),
    width="stretch",
    on_select="rerun",
    selection_mode="household_type",
    key="household_chart",
)
st.caption("Click a bar to list that household type's records below.")
st.dataframe(
    by_household[
        [
            "household_type",
            "intake_records",
            "average_recorded_age",
            "children_recorded",
        ]
    ],
    column_config={
        "household_type": "Household type",
        "intake_records": st.column_config.NumberColumn("Intake records", format="localized"),
        "average_recorded_age": st.column_config.NumberColumn(
            "Average recorded age", format="%.1f"
        ),
        "children_recorded": st.column_config.NumberColumn(
            "Children recorded", format="localized"
        ),
    },
    hide_index=True,
    width="stretch",
)

st.divider()
st.subheader("Individual intake records")
st.caption(
    "Every record is synthetic. Pick a day and household type, then select a row "
    "to open the full record with its intake note and photo."
)
picker_columns = st.columns(2)
# No key, so the picker resets whenever the sidebar date range changes.
record_day = picker_columns[0].date_input(
    "Intake date",
    value=end_date,
    min_value=start_date,
    max_value=end_date,
    format="MM/DD/YYYY",
)
record_household = picker_columns[1].selectbox(
    "Household type", household_options, key="drill_household"
)

day_records = load_day_records(intake_path, record_day)
day_records["household_type"] = day_records["household_type"].fillna("Not reported")
if selected_program != "All programs":
    day_records = day_records.loc[day_records["program_type"] == selected_program]
if record_household != ALL_HOUSEHOLDS:
    day_records = day_records.loc[day_records["household_type"] == record_household]
day_records = day_records.sort_values("intake_id").reset_index(drop=True)

if day_records.empty:
    st.info(f"No synthetic intake records match these choices on {record_day:%B %d, %Y}.")
    st.stop()

st.markdown(f"**{len(day_records):,} records on {record_day:%B %d, %Y}**")
selection = st.dataframe(
    day_records[list(LIST_COLUMNS)],
    column_config={
        column: (
            st.column_config.NumberColumn(label, format="%d")
            if column in ("age", "children_count")
            else label
        )
        for column, label in LIST_COLUMNS.items()
    },
    hide_index=True,
    width="stretch",
    height=320,
    on_select="rerun",
    selection_mode="single-row",
    # A new key per list clears the selected row when the list changes.
    key=f"records_{record_day}_{record_household}_{selected_program}",
)
selected_rows = selection.selection.rows
if not selected_rows:
    st.caption("Select a row to open that record.")
    st.stop()

record = day_records.iloc[selected_rows[0]]
details_path = os.getenv("INTAKE_DETAILS_PATH", DEFAULT_DETAILS_PATH)
details = load_record_details(details_path, record_day, record["intake_id"])

with st.container(border=True):
    st.markdown(f"#### {escape_markdown(str(record['client_display_name']))}")
    st.caption(f"Intake `{record['intake_id']}` · {record_day:%B %d, %Y} · {record['program_type']}")
    photo_column, facts_column = st.columns([1, 3])
    photo = decode_photo(details["intake_photo_b64"]) if details else None
    if photo is not None:
        photo_column.image(photo, caption="Intake photo (synthetic)", width=220)
    else:
        photo_column.caption("No photo could be shown for this record.")
    fact_columns = facts_column.columns(3)
    fact_columns[0].metric("Age", format_value(record["age"]))
    fact_columns[1].metric("Children", format_value(record["children_count"]))
    facts_column.markdown(
        "  \n".join(
            f"**{label}:** {escape_markdown(format_value(record[column]))}"
            for column, label in [
                ("household_type", "Household type"),
                ("contact_type", "Contact type"),
                ("presenting_reason", "Presenting reason"),
            ]
        )
    )

    section_columns = st.columns(2)
    for column, (title, table) in zip(section_columns, detail_sections(record).items()):
        column.markdown(f"**{title}**")
        column.dataframe(table, hide_index=True, width="stretch")

    st.markdown("**Intake note**")
    if details is None:
        st.info(
            f"No note or photo found in `{details_path}` for this record. Run "
            "`uv run python etl/intake_to_parquet.py` to rebuild the details table."
        )
    elif details["intake_note"]:
        st.markdown(escape_markdown(details["intake_note"]))
    else:
        st.caption("No note was recorded.")
