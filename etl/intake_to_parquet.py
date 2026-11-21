"""Convert the raw intake-log CSV into curated Parquet tables.

Reads the single multiline CSV once and writes two tables, both partitioned
by intake month:

  curated/intakes/         every field except the note and photo (for the dashboard)
  curated/intake_details/  intake_id, intake_date, note, photo (for drill-down)

Run on the Spark master (see README for the credential setup):

  export SPARK_MASTER_URL="spark://<MASTER_PRIVATE_IP>:7077"
  python3 etl/intake_to_parquet.py
"""

import argparse
import configparser
import os
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F

from schemas import DETAIL_COLUMNS, INTAKE_SCHEMA

BUCKET = "dsan6000-nyc-dashboard-data"
DEFAULT_SOURCE = f"s3a://{BUCKET}/source/nyc-shelters/intake_log.csv"
DEFAULT_OUTPUT = f"s3a://{BUCKET}/curated"


def load_aws_credentials(profile: str) -> dict[str, str]:
    """Read AWS Academy credentials from ~/.aws/credentials (never printed)."""
    path = Path(os.getenv("AWS_SHARED_CREDENTIALS_FILE", "~/.aws/credentials"))
    parser = configparser.ConfigParser()
    if not parser.read(path.expanduser()) or profile not in parser:
        raise SystemExit(
            f"No [{profile}] profile in {path}. Paste the AWS Details block "
            "from AWS Academy first (see README)."
        )
    section = parser[profile]
    try:
        return {
            "access_key": section["aws_access_key_id"],
            "secret_key": section["aws_secret_access_key"],
            "session_token": section.get("aws_session_token", ""),
        }
    except KeyError as missing:
        raise SystemExit(f"{path} [{profile}] is missing {missing}.") from None


def build_spark(credentials: dict[str, str]) -> SparkSession:
    builder = (
        SparkSession.builder.appName("NYC Shelter Intake ETL")
        .master(os.getenv("SPARK_MASTER_URL", "local[*]"))
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.5.0")
        .config("spark.hadoop.fs.s3a.access.key", credentials["access_key"])
        .config("spark.hadoop.fs.s3a.secret.key", credentials["secret_key"])
    )
    if credentials["session_token"]:
        builder = builder.config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider",
        ).config("spark.hadoop.fs.s3a.session.token", credentials["session_token"])
    else:
        builder = builder.config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
    return builder.getOrCreate()


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

    spark = build_spark(load_aws_credentials(args.aws_profile))
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
        records.select("intake_id", "intake_date", "intake_month", *DETAIL_COLUMNS)
        .repartition("intake_month")
        .write.mode("overwrite")
        .partitionBy("intake_month")
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
