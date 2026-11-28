"""Spark session with S3 write access, shared by the ETL jobs."""

import configparser
import os
from pathlib import Path

from pyspark.sql import SparkSession

BUCKET = "dsan6000-nyc-dashboard-data"


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


def build_spark(credentials: dict[str, str], app_name: str) -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
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
