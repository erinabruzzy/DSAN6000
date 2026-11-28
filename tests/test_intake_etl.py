import csv
import sys
from datetime import date
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

sys.path.insert(0, str(Path(__file__).parents[1] / "etl"))
from intake_to_parquet import write_curated_tables
from schemas import INTAKE_SCHEMA


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("intake-etl-test")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_writes_narrow_tables_and_preserves_multiline_notes(spark, tmp_path):
    source = tmp_path / "intakes.csv"
    output = tmp_path / "curated"
    columns = [field.name for field in INTAKE_SCHEMA.fields]
    included = {
        "intake_id": "test-included",
        "synthetic_data": "true",
        "intake_date": "2026-01-02",
        "intake_note": "first line\nsecond line",
        "intake_photo_b64": "synthetic-photo",
    }
    excluded = {
        "intake_id": "test-excluded",
        "synthetic_data": "false",
        "intake_date": "2026-01-03",
    }
    with source.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerow({**{column: "" for column in columns}, **included})
        writer.writerow({**{column: "" for column in columns}, **excluded})

    assert write_curated_tables(spark, str(source), str(output)) == 1

    intakes = spark.read.parquet(str(output / "intakes")).collect()
    details_path = output / "intake_details" / f"intake_date={date(2026, 1, 2)}"
    details = spark.read.parquet(str(details_path)).collect()

    assert [row["intake_id"] for row in intakes] == ["test-included"]
    assert intakes[0]["intake_month"] == "2026-01"
    assert [row["intake_id"] for row in details] == ["test-included"]
    assert details[0]["intake_note"] == "first line\nsecond line"
    assert details[0]["intake_photo_b64"] == "synthetic-photo"
