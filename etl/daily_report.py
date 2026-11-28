"""Generate the once-daily shelter report and save it to S3.

Reads the curated tables, fetches the NYC weather forecast (Open-Meteo),
active National Weather Service alerts, and recent news (GDELT, or Google
News RSS when GDELT is rate-limited), and saves
the report as Markdown together with the raw API responses, so each day's
report and what it was based on are preserved:

  curated/daily_reports/report_date=YYYY-MM-DD/

Rerunning on the same day replaces only that day's report. An API that fails
is noted in the report instead of stopping it; only a missing daily census
stops the job. Run it after etl/dhs_daily_ingest.py (see README for cron):

  export SPARK_MASTER_URL="spark://<MASTER_PRIVATE_IP>:7077"
  uv run python etl/daily_report.py
"""

import argparse
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import pandas as pd
from pyspark.errors import AnalysisException
from pyspark.sql import functions as F

# The report logic lives at the repository root, shared with the dashboard.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daily_report import (  # noqa: E402
    NYC_LATITUDE,
    NYC_LONGITUDE,
    build_report,
    merge_news,
    parse_alerts,
    parse_news,
    parse_news_rss,
    parse_weather,
)
from forecast import daily_counts, forecast_next_day  # noqa: E402
from spark_session import BUCKET, build_spark, load_aws_credentials  # noqa: E402

# A generic client name: the APIs ask for one, and it identifies no one.
USER_AGENT = "DSAN6000-nyc-shelter-report (course project)"
TIMEOUT_SECONDS = 30
# Below this many relevant GDELT headlines, Google News is added.
MIN_HEADLINES = 3

WEATHER_URL = "https://api.open-meteo.com/v1/forecast?" + urlencode(
    {
        "latitude": NYC_LATITUDE,
        "longitude": NYC_LONGITUDE,
        "daily": ",".join(
            [
                "temperature_2m_min",
                "temperature_2m_max",
                "apparent_temperature_min",
                "apparent_temperature_max",
                "precipitation_probability_max",
                "precipitation_sum",
                "snowfall_sum",
                "wind_speed_10m_max",
            ]
        ),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "America/New_York",
        "forecast_days": 3,
    }
)
# A place name keeps results about the city rather than anywhere called New York.
NYC_PLACES = '("New York City" OR NYC OR Bronx OR Brooklyn OR Queens OR Manhattan OR "Staten Island")'
ALERTS_URL = f"https://api.weather.gov/alerts/active?point={NYC_LATITUDE},{NYC_LONGITUDE}"
NEWS_URL = "https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode(
    {
        "query": f"(homeless OR shelter OR encampment) {NYC_PLACES} sourcelang:english",
        "mode": "artlist",
        "format": "json",
        "timespan": "2d",
        "maxrecords": 50,
        "sort": "datedesc",
    }
)


GOOGLE_NEWS_URL = "https://news.google.com/rss/search?" + urlencode(
    {"q": f"homeless {NYC_PLACES} when:2d", "hl": "en-US", "gl": "US", "ceid": "US:en"}
)


def fetch_text(name: str, url: str) -> str | None:
    """The response body, or None (logged) if the request fails. Never raises."""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, UnicodeDecodeError) as error:
        print(f"{name}: unavailable ({type(error).__name__}: {error})")
        return None


def fetch_json(name: str, url: str, retries: int = 0) -> tuple[dict | None, str]:
    """(parsed JSON or None, raw text or the error). Never raises."""
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                text = response.read().decode("utf-8")
            return json.loads(text), text
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as error:
            # GDELT allows one request every 5 seconds and answers faster ones
            # with 429 or a plain-text notice, so back off and retry.
            if attempt < retries:
                time.sleep(10 * (attempt + 1))
                continue
            print(f"{name}: unavailable ({type(error).__name__}: {error})")
            return None, f"error: {type(error).__name__}: {error}"
    return None, "error: no attempt made"


def read_optional(spark, path: str):
    try:
        return spark.read.parquet(path)
    except AnalysisException as error:
        if error.getCondition() != "PATH_NOT_FOUND":
            raise
        print(f"No table at {path}; the report will note it as missing.")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        default=f"s3a://{BUCKET}",
        help="Bucket (or local directory) that holds curated/.",
    )
    parser.add_argument("--aws-profile", default="default")
    args = parser.parse_args()
    curated = f"{args.root}/curated"

    today = pd.Timestamp(dt.datetime.now(ZoneInfo("America/New_York")).date())
    spark = build_spark(load_aws_credentials(args.aws_profile), "Daily Shelter Report")
    print(f"Spark master: {spark.sparkContext.master}")

    daily = read_optional(spark, f"{curated}/dhs_daily")
    if daily is None:
        raise SystemExit("The daily census is required; run etl/dhs_daily_ingest.py first.")
    census = daily.toPandas()
    census["census_date"] = pd.to_datetime(census["census_date"])

    district_table = read_optional(spark, f"{curated}/dhs_district")
    districts = None
    if district_table is not None:
        districts = district_table.toPandas()
        districts["report_date"] = pd.to_datetime(districts["report_date"])

    intakes = read_optional(spark, f"{curated}/intakes")
    intake_counts = next_day = None
    if intakes is not None:
        per_day = (
            intakes.filter(F.col("synthetic_data") & F.col("intake_date").isNotNull())
            .groupBy("intake_date")
            .count()
            .toPandas()
        )
        if not per_day.empty:
            intake_counts = daily_counts(per_day["intake_date"], per_day["count"])
            next_day = forecast_next_day(intake_counts)

    weather_payload, weather_raw = fetch_json("Weather", WEATHER_URL)
    alerts_payload, alerts_raw = fetch_json("NWS alerts", ALERTS_URL)
    # GDELT rate-limits by IP and often refuses even a single request, and its
    # matches can be sparse, so add Google News RSS when it returns too few.
    news_payload, news_raw = fetch_json("GDELT news", NEWS_URL, retries=1)
    news, news_source = parse_news(news_payload), "GDELT"
    if news is None or len(news) < MIN_HEADLINES:
        rss = fetch_text("Google News", GOOGLE_NEWS_URL)
        google = parse_news_rss(rss)
        if google:
            news = merge_news(news, google)
            news_source = "Google News" if news_payload is None else "GDELT and Google News"
            news_raw = json.dumps({"gdelt": news_raw, "google_news_rss": rss})

    report = build_report(
        today,
        census,
        districts,
        intake_counts,
        next_day,
        parse_weather(weather_payload),
        parse_alerts(alerts_payload),
        news,
        news_source,
    )

    row = pd.DataFrame(
        [
            {
                "generated_at": pd.Timestamp.now(tz="UTC").tz_localize(None),
                "census_date": census["census_date"].max().date(),
                "warning_count": len(report.warnings),
                "warnings": json.dumps(report.warnings),
                "markdown": report.markdown,
                "weather_json": weather_raw,
                "alerts_json": alerts_raw,
                "news_source": news_source,
                # GDELT's JSON, or {"gdelt": ..., "google_news_rss": ...} when
                # Google News was added.
                "news_raw": news_raw,
                "report_date": today.date(),
            }
        ]
    )
    # Replace only today's partition when the job reruns.
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    output = f"{curated}/daily_reports"
    (
        spark.createDataFrame(row)
        .coalesce(1)
        .write.mode("overwrite")
        .partitionBy("report_date")
        .parquet(output)
    )
    saved = spark.read.parquet(f"{output}/report_date={today.date()}").count()
    if saved != 1:
        raise SystemExit(f"Expected 1 report for {today.date()} in {output}, found {saved}.")
    print(f"Wrote the {today.date()} report to {output} ({len(report.warnings)} warnings)")
    print(report.markdown)
    spark.stop()


if __name__ == "__main__":
    main()
