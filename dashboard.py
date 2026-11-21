import os
from datetime import timedelta

import pandas as pd
import streamlit as st
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    BooleanType,
    DateType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

st.set_page_config(page_title="NYC Shelter Snapshot", layout="wide")

BUCKET = "dsan6000-nyc-dashboard-data"
DEFAULT_INTAKE_PATH = f"s3a://{BUCKET}/source/nyc-shelters/intake_log.csv"
DEFAULT_REPORT_PATH = f"s3a://{BUCKET}/DHS_Daily_Report_20261120.csv"

INTAKE_SCHEMA = StructType(
    [
        StructField("intake_id", StringType(), True),
        StructField("synthetic_data", BooleanType(), True),
        StructField("client_display_name", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("household_type", StringType(), True),
        StructField("children_count", IntegerType(), True),
        StructField("program_type", StringType(), True),
        StructField("contact_type", StringType(), True),
        StructField("intake_date", DateType(), True),
        StructField("presenting_reason", StringType(), True),
        StructField("prior_living_situation_reported", StringType(), True),
        StructField("reported_housing_instability_days", IntegerType(), True),
        StructField("employment_status_reported", StringType(), True),
        StructField("income_status_reported", StringType(), True),
        StructField("primary_barrier", StringType(), True),
        StructField("immediate_needs", StringType(), True),
        StructField("potential_referral", StringType(), True),
        StructField("referral_status", StringType(), True),
        StructField("client_goal", StringType(), True),
        StructField("follow_up_days", IntegerType(), True),
        StructField("follow_up_target_date", DateType(), True),
        StructField("housing_eligibility_verified", BooleanType(), True),
        StructField("staff_role", StringType(), True),
        StructField("note_word_count", IntegerType(), True),
        StructField("intake_note", StringType(), True),
        StructField("intake_photo_b64", StringType(), True),
    ]
)


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
    spark = get_spark()
    report = spark.read.option("header", "true").csv(path)
    report = report.toDF(*[column.strip() for column in report.columns])
    return (
        report.select(
            F.to_date(F.col("Date of Census"), "MM/dd/yyyy").alias("census_date"),
            F.col("Total Adults in Shelter").cast("long").alias("adults"),
            F.col("Total Children in Shelter").cast("long").alias("children"),
            F.col("Total Individuals in Shelter").cast("long").alias("individuals"),
            F.col("Total Single Adults in Shelter")
            .cast("long")
            .alias("single_adults"),
            F.col("Total Individuals in Families with Children in Shelter")
            .cast("long")
            .alias("family_individuals"),
            F.col("Individuals in Adult Families in Shelter")
            .cast("long")
            .alias("adult_family_individuals"),
        )
        .filter(F.col("census_date").isNotNull())
        .toPandas()
    )


@st.cache_data(
    ttl=3600, show_spinner="Scanning and aggregating synthetic intake records..."
)
def load_intake_summary(path: str) -> tuple[pd.DataFrame, int]:
    spark = get_spark()
    records = (
        spark.read.schema(INTAKE_SCHEMA)
        .option("header", "true")
        .option("multiLine", "true")
        .csv(path)
        .select(
            "synthetic_data",
            "intake_date",
            "program_type",
            "household_type",
            "age",
            "children_count",
        )
    )
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


def format_count(value: object) -> str:
    if pd.isna(value):
        return "Unavailable"
    return f"{int(value):,}"


st.title("NYC Shelter Population Snapshot")
st.caption(
    "Synthetic data only. DHS census totals describe reported shelter population; "
    "intake statistics describe intake records, not current residents."
)

spark = get_spark()
st.caption(f"Spark master: `{spark.sparkContext.master}`")

report_path = os.getenv("DHS_DAILY_REPORT_PATH", DEFAULT_REPORT_PATH)
report = load_daily_report(report_path)
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

st.subheader(f"DHS reported population — {selected_report_date:%B %d, %Y}")
population_columns = st.columns(3)
population_columns[0].metric(
    "Individuals in shelter", format_count(selected_report["individuals"])
)
population_columns[1].metric("Adults", format_count(selected_report["adults"]))
population_columns[2].metric("Children", format_count(selected_report["children"]))

population_breakdown = pd.DataFrame(
    {
        "Group": [
            "Single adults",
            "Individuals in families with children",
            "Individuals in adult families",
        ],
        "Individuals": [
            selected_report["single_adults"],
            selected_report["family_individuals"],
            selected_report["adult_family_individuals"],
        ],
    }
).set_index("Group")
st.bar_chart(population_breakdown)

trend_end = pd.Timestamp(selected_report["census_date"])
# DateOffset avoids a NumPy 2.5 deprecation warning raised by pd.Timedelta in pandas 2.3.
trend_start = trend_end - pd.DateOffset(days=365)
population_trend = report.loc[
    report["census_date"].between(trend_start, trend_end),
    ["census_date", "adults", "children"],
].set_index("census_date")
st.caption(
    "Daily reported adult and child counts during the 12 months through the selected census date."
)
st.line_chart(population_trend)

st.divider()
st.subheader("Synthetic intake-record profile")
intake_path = os.getenv("INTAKE_S3_PATH", DEFAULT_INTAKE_PATH)
intakes, excluded_records = load_intake_summary(intake_path)
undated_records = int(
    intakes.loc[intakes["intake_date"].isna(), "intake_records"].sum()
)
valid_intakes = intakes.dropna(subset=["intake_date"]).copy()
if valid_intakes.empty:
    st.info("There are no synthetic intake records with a usable intake date.")
    st.stop()

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
