"""Pull the DHS census by borough, community district, and facility type into S3.

NYC Open Data dataset veav-vj3r: one row per community district per month,
as of the month's last day, since July 2018 (about 5,700 rows). Like the
daily job, each run re-fetches the whole history, so it is safe to run as
often as you like (DHS adds one month at a time). Each run writes:

  raw/dhs_district/fetched_date=YYYY-MM-DD/  the API response as fetched (CSV)
  curated/dhs_district/                      typed Parquet, one row per district per month

Run on the Spark master (see README for the credential setup and cron line):

  export SPARK_MASTER_URL="spark://<MASTER_PRIVATE_IP>:7077"
  uv run python etl/dhs_district_ingest.py
"""

import argparse
import datetime as dt

from pyspark.sql import DataFrame, functions as F

from open_data import fetch_dataset
from schemas import DHS_DISTRICT_COUNTS, DHS_DISTRICT_KEYS
from spark_session import BUCKET, build_spark, load_aws_credentials

API_URL = "https://data.cityofnewyork.us/resource/veav-vj3r.csv"
KEY_COLUMNS = list(DHS_DISTRICT_KEYS.values())


def _blank(field: str):
    return F.trim(F.col(field)) == ""


def curate(raw: DataFrame) -> DataFrame:
    """Type the raw strings. A blank count becomes 0; anything unparseable becomes null.

    try_cast and try_to_date return null instead of failing (Spark 4 runs in
    ANSI mode), so check_curated can report every bad value at once.
    """
    return raw.select(
        # Values look like 2026-10-31T00:00:00.000; the time is always midnight.
        F.try_to_date(F.substring("report_date", 1, 10), "yyyy-MM-dd").alias("report_date"),
        F.trim(F.col("borough")).alias("borough"),
        # Blank (outside the city) stays null; a non-number is caught by the check below.
        F.when(_blank("community_districts"), F.lit(None).cast("int"))
        .otherwise(F.col("community_districts").try_cast("int"))
        .alias("community_district"),
        F.when(_blank("community_districts"), F.lit(False))
        .otherwise(F.col("community_districts").try_cast("int").isNull())
        .alias("_bad_district"),
        *[
            F.when(_blank(field), F.lit(0).cast("long"))
            .otherwise(F.col(field).try_cast("long"))
            .alias(column)
            for field, column in DHS_DISTRICT_COUNTS.items()
        ],
    )


def check_curated(curated: DataFrame, raw_rows: int) -> DataFrame:
    """Refuse to write on unparseable values, negative counts, or repeated rows."""
    counts = list(DHS_DISTRICT_COUNTS.values())
    problems = curated.select(
        F.sum(F.col("report_date").isNull().cast("int")).alias("report_date"),
        F.sum((F.col("borough") == "").cast("int")).alias("borough"),
        F.sum(F.col("_bad_district").cast("int")).alias("community_district"),
        *[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in counts],
        F.sum(sum(F.when(F.col(c) < 0, 1).otherwise(0) for c in counts)).alias("negative"),
    ).first()
    bad = {column: count for column, count in problems.asDict().items() if count}
    if bad:
        raise SystemExit(f"Values that failed to parse or are negative, by column: {bad}")

    # distinct() treats nulls as equal, so Westchester rows are compared too.
    distinct_keys = curated.select(*KEY_COLUMNS).distinct().count()
    if distinct_keys != raw_rows:
        raise SystemExit(
            f"{raw_rows:,} rows but {distinct_keys:,} distinct (month, borough, district) keys."
        )
    return curated.drop("_bad_district")


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

    rows = fetch_dataset(
        args.api_url,
        "report_date, borough, community_districts",
        [*DHS_DISTRICT_KEYS, *DHS_DISTRICT_COUNTS],
    )
    print(f"Fetched {len(rows):,} rows from {args.api_url}")

    spark = build_spark(load_aws_credentials(args.aws_profile), "DHS District Ingest")
    print(f"Spark master: {spark.sparkContext.master}")

    raw = spark.createDataFrame(rows).cache()
    curated = check_curated(curate(raw).cache(), len(rows))

    fetched_date = dt.date.today().isoformat()
    raw_path = f"{args.output_root}/raw/dhs_district/fetched_date={fetched_date}"
    raw.coalesce(1).write.mode("overwrite").option("header", "true").csv(raw_path)
    print(f"Wrote {raw_path}")

    curated_path = f"{args.output_root}/curated/dhs_district"
    curated.coalesce(1).write.mode("overwrite").parquet(curated_path)
    written = spark.read.parquet(curated_path).count()
    if written != len(rows):
        raise SystemExit(f"Wrote {written:,} rows to {curated_path}, expected {len(rows):,}.")
    latest = curated.agg(F.max("report_date")).first()[0]
    print(f"Wrote {curated_path}: {written:,} rows through {latest}")

    spark.stop()


if __name__ == "__main__":
    main()
