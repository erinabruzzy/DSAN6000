"""Convert the raw intake-log CSV into curated Parquet tables.

Reads the single multiline CSV once and writes two tables:

  curated/intakes/         every field except the note and photo (for the
                           dashboard), partitioned by intake_month
  curated/intake_details/  intake_id, note, photo (for drill-down), partitioned
                           by intake_date, so looking up one record reads one
                           day's folder (about 15 MB) instead of a month's

Run on the Spark master (see README for the credential setup):

  export SPARK_MASTER_URL="spark://<MASTER_PRIVATE_IP>:7077"
  uv run python etl/intake_to_parquet.py
"""

import argparse

from pyspark import StorageLevel
from pyspark.sql import functions as F

from schemas import DETAIL_COLUMNS, INTAKE_SCHEMA
from spark_session import BUCKET, build_spark, load_aws_credentials

DEFAULT_SOURCE = f"s3a://{BUCKET}/source/nyc-shelters/intake_log.csv"
DEFAULT_OUTPUT = f"s3a://{BUCKET}/curated"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--aws-profile", default="default")
    parser.add_argument(
        "--partitions",
        type=int,
        default=24,
        help="Partitions to spread the single-file CSV across after reading.",
    )
    args = parser.parse_args()

    spark = build_spark(
        load_aws_credentials(args.aws_profile), "NYC Shelter Intake ETL"
    )
    print(f"Spark master: {spark.sparkContext.master}")
    print(f"Source: {args.source}")

    raw = (
        spark.read.schema(INTAKE_SCHEMA)
        .option("header", "true")
        .option("multiLine", "true")
        .csv(args.source)
    )

    # The EDA found synthetic_data is always true; keep the guard anyway so
    # non-synthetic rows can never reach the curated tables.
    records = (
        raw.filter(F.col("synthetic_data") == F.lit(True))
        .withColumn("intake_month", F.date_format("intake_date", "yyyy-MM"))
        .repartition(args.partitions)
        # A multiline CSV can only be read by one task; cache the parsed rows
        # so both tables are written from a single scan.
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    kept_rows = records.count()  # materializes the cache
    print(f"Synthetic rows kept: {kept_rows:,}")

    intakes_path = f"{args.output}/intakes"
    details_path = f"{args.output}/intake_details"

    (
        records.drop(*DETAIL_COLUMNS)
        .repartition("intake_month")
        .write.mode("overwrite")
        .partitionBy("intake_month")
        .parquet(intakes_path)
    )
    print(f"Wrote {intakes_path}")

    (
        records.select("intake_id", "intake_date", *DETAIL_COLUMNS)
        .repartition("intake_date")
        .write.mode("overwrite")
        .partitionBy("intake_date")
        .parquet(details_path)
    )
    print(f"Wrote {details_path}")

    # Verify the writes by reading the row counts back.
    for name, path in [("intakes", intakes_path), ("intake_details", details_path)]:
        written = spark.read.parquet(path).count()
        status = "OK" if written == kept_rows else "MISMATCH"
        print(f"{name}: {written:,} rows [{status}]")
        if written != kept_rows:
            raise SystemExit(f"{name} row count does not match the source.")

    records.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
