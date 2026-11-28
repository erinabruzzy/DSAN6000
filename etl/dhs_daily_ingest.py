"""Pull the NYC DHS Daily Report from NYC Open Data into S3.

The full history is about 2,100 rows (one per day since March 2021), so each
run re-fetches all of it. That keeps the job idempotent: a missed day (for
example while the AWS Academy lab is stopped) is filled in by the next run, and
any corrections DHS makes to past days are picked up. Each run writes:

  raw/dhs_daily/fetched_date=YYYY-MM-DD/  the API response as fetched (CSV)
  curated/dhs_daily/                      typed Parquet, one row per census date

Run on the Spark master (see README for the credential setup and cron line):

  export SPARK_MASTER_URL="spark://<MASTER_PRIVATE_IP>:7077"
  uv run python etl/dhs_daily_ingest.py
"""

import argparse
import datetime as dt

from pyspark.sql import DataFrame, functions as F

from open_data import fetch_dataset
from schemas import DHS_DAILY_COLUMNS
from spark_session import BUCKET, build_spark, load_aws_credentials

API_URL = "https://data.cityofnewyork.us/resource/k46n-sa2m.csv"


def curate(raw: DataFrame) -> DataFrame:
    """Type the raw strings: census_date as a date, every count as a long."""
    date_field, *count_fields = DHS_DAILY_COLUMNS
    return raw.select(
        # Values look like 2026-11-25T00:00:00.000; the time is always midnight.
        F.try_to_date(F.substring(date_field, 1, 10), "yyyy-MM-dd").alias(
            DHS_DAILY_COLUMNS[date_field]
        ),
        *[
            F.col(field).try_cast("long").alias(DHS_DAILY_COLUMNS[field])
            for field in count_fields
        ],
    )


def check_curated(curated: DataFrame, raw_rows: int) -> None:
    """Refuse to write if typing lost rows or values, or dates repeat."""
    nulls = curated.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in curated.columns]
    ).first()
    bad = {column: count for column, count in nulls.asDict().items() if count}
    if bad:
        raise SystemExit(f"Values that failed to parse, by column: {bad}")

    distinct_dates = curated.select("census_date").distinct().count()
    if distinct_dates != raw_rows:
        raise SystemExit(
            f"{raw_rows:,} rows but {distinct_dates:,} distinct census dates."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument(
        "--output-root",
        default=f"s3a://{BUCKET}",
        help="Bucket (or local directory) that holds raw/ and curated/.",
    )
    parser.add_argument("--aws-profile", default="default")
    args = parser.parse_args()

    report = fetch_dataset(args.api_url, "date_of_census", DHS_DAILY_COLUMNS)
    print(f"Fetched {len(report):,} rows from {args.api_url}")

    spark = build_spark(load_aws_credentials(args.aws_profile), "DHS Daily Ingest")
    print(f"Spark master: {spark.sparkContext.master}")

    raw = spark.createDataFrame(report).cache()
    curated = curate(raw).cache()
    check_curated(curated, len(report))

    fetched_date = dt.date.today().isoformat()
    raw_path = f"{args.output_root}/raw/dhs_daily/fetched_date={fetched_date}"
    raw.coalesce(1).write.mode("overwrite").option("header", "true").csv(raw_path)
    print(f"Wrote {raw_path}")

    curated_path = f"{args.output_root}/curated/dhs_daily"
    curated.coalesce(1).write.mode("overwrite").parquet(curated_path)
    written = spark.read.parquet(curated_path).count()
    if written != len(report):
        raise SystemExit(
            f"Wrote {written:,} rows to {curated_path}, expected {len(report):,}."
        )
    latest = curated.agg(F.max("census_date")).first()[0]
    print(f"Wrote {curated_path}: {written:,} rows through {latest}")

    spark.stop()


if __name__ == "__main__":
    main()
