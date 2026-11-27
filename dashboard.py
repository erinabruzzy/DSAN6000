import os
from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st
from pyspark.errors import AnalysisException
from pyspark.sql import SparkSession, functions as F

from census import (
    ALERT_WINDOW,
    ALERT_Z,
    GROUPS,
    SUBGROUPS,
    breakdown,
    group_trend,
    unusual_changes,
)
from forecast import (
    INTERVAL,
    MIN_TRAINING_DAYS,
    NextDayForecast,
    daily_counts,
    forecast_next_day,
)

st.set_page_config(page_title="NYC Shelter Snapshot", layout="wide")

BUCKET = "dsan6000-nyc-dashboard-data"
# Curated Parquet tables written by the jobs in etl/.
DEFAULT_INTAKE_PATH = f"s3a://{BUCKET}/curated/intakes"  # intake_to_parquet.py
DEFAULT_REPORT_PATH = f"s3a://{BUCKET}/curated/dhs_daily"  # dhs_daily_ingest.py
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


@st.cache_resource
def get_spark() -> SparkSession:
    master = os.getenv("SPARK_MASTER_URL", "local[*]")
    return (
        SparkSession.builder.appName("NYC Shelter Dashboard")
        .master(master)
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.5.0")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider",
        )
        .getOrCreate()
    )


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


def load_or_stop(loader, path: str, etl_script: str):
    """Run a loader, or explain which ETL job to run if its table is missing."""
    try:
        return loader(path)
    except AnalysisException as error:
        if error.getCondition() != "PATH_NOT_FOUND":
            raise
        st.error(
            f"No curated table at `{path}`. Run "
            f"`uv run python etl/{etl_script}` on the Spark master first "
            "(see README)."
        )
        st.stop()


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

st.divider()
st.subheader("Synthetic intake-record profile")
intake_path = os.getenv("INTAKE_PARQUET_PATH", DEFAULT_INTAKE_PATH)
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
st.bar_chart(by_household.set_index("household_type")["intake_records"])
st.dataframe(
    by_household[
        [
            "household_type",
            "intake_records",
            "average_recorded_age",
            "children_recorded",
        ]
    ].rename(
        columns={
            "household_type": "Household type",
            "intake_records": "Intake records",
            "average_recorded_age": "Average recorded age",
            "children_recorded": "Children recorded",
        }
    ),
    hide_index=True,
    width="stretch",
)
