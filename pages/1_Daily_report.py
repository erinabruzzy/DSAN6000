"""The generated daily reports, one per day (written by etl/daily_report.py)."""

import os

import pandas as pd
import streamlit as st

from app_spark import BUCKET, get_spark, load_or_none

st.set_page_config(page_title="Daily report", layout="wide")

DEFAULT_PATH = f"s3a://{BUCKET}/curated/daily_reports"  # etl/daily_report.py


@st.cache_data(ttl=900, show_spinner="Loading the daily reports...")
def load_reports(path: str) -> pd.DataFrame:
    """Every saved report's date, warnings, and Markdown (small: one row per day)."""
    reports = (
        get_spark()
        .read.parquet(path)
        .select("report_date", "generated_at", "warning_count", "warnings", "markdown")
        .toPandas()
    )
    reports["report_date"] = pd.to_datetime(reports["report_date"]).dt.date
    return reports.sort_values("report_date", ascending=False).reset_index(drop=True)


path = os.getenv("DAILY_REPORTS_PATH", DEFAULT_PATH)
reports = load_or_none(load_reports, path)
if reports is None or reports.empty:
    st.title("Daily report")
    st.info(
        f"No reports at `{path}` yet. Run `uv run python etl/daily_report.py` on the "
        "Spark master after the daily ingest (see README)."
    )
    st.stop()

dates = list(reports["report_date"])
warning_counts = dict(zip(reports["report_date"], reports["warning_count"]))


def label(day) -> str:
    count = int(warning_counts[day])
    return f"{day:%B %d, %Y}" + (f" ({count} warning{'s' if count != 1 else ''})" if count else "")


chosen = st.sidebar.selectbox("Report date", dates, format_func=label)
report = reports.loc[reports["report_date"] == chosen].iloc[0]

if chosen != dates[0]:
    st.info(f"This is an earlier report. The newest is from {dates[0]:%B %d, %Y}.")
count = int(report["warning_count"])
if count:
    st.warning(
        f"{count} warning{'s' if count != 1 else ''} in this report; see Warnings below.",
        icon=":material/warning:",
    )

st.markdown(report["markdown"])
st.caption(f"Generated {pd.Timestamp(report['generated_at']):%B %d, %Y at %H:%M} UTC.")
st.download_button(
    "Download this report (Markdown)",
    report["markdown"],
    file_name=f"nyc-shelter-report-{chosen}.md",
    mime="text/markdown",
)
