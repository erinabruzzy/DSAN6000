"""Spark access shared by the dashboard's pages (read-only, anonymous S3)."""

import os

import streamlit as st
from pyspark.errors import AnalysisException
from pyspark.sql import SparkSession

BUCKET = "dsan6000-nyc-dashboard-data"


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


def load_or_none(loader, path: str):
    """Run a loader, or return None if its table doesn't exist yet."""
    try:
        return loader(path)
    except AnalysisException as error:
        if error.getCondition() != "PATH_NOT_FOUND":
            raise
        return None
