"""Shared Spark schemas for the NYC shelter data."""

from pyspark.sql.types import (
    BooleanType,
    DateType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Matches intakes_codebook.csv and the types inferred in Intake_Log_EDA.ipynb.
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

# Large per-record fields that live in the details table, not the core table.
DETAIL_COLUMNS = ["intake_note", "intake_photo_b64"]

# NYC Open Data "DHS Daily Report" (k46n-sa2m) API field -> curated column.
# date_of_census becomes census_date (a date); every other field is a count.
DHS_DAILY_COLUMNS = {
    "date_of_census": "census_date",
    "total_adults_in_shelter": "adults",
    "total_children_in_shelter": "children",
    "total_individuals_in_shelter": "individuals",
    "single_adult_men_in_shelter": "single_adult_men",
    "single_adult_women_in_shelter": "single_adult_women",
    "total_single_adults_in_shelter": "single_adults",
    "families_with_children_in_shelter": "families_with_children",
    "adults_in_families_with_children_in_shelter": "family_adults",
    "children_in_families_with_children_in_shelter": "family_children",
    # The trailing underscore is in the source field name.
    "total_individuals_in_families_with_children_in_shelter_": "family_individuals",
    "adult_families_in_shelter": "adult_families",
    "individuals_in_adult_families_in_shelter": "adult_family_individuals",
}
