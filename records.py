"""Helpers for showing individual intake records (no Streamlit or Spark, so testable).

Every intake record is synthetic. The codebook says `intake_photo_b64` is a
Base64-encoded image but not which format, so decode_photo checks the bytes.
"""

import base64
import binascii

import numpy as np
import pandas as pd

# Columns of curated/intakes listed for one day's records, in display order.
LIST_COLUMNS = {
    "intake_id": "Intake ID",
    "client_display_name": "Name",
    "age": "Age",
    "household_type": "Household type",
    "children_count": "Children",
    "contact_type": "Contact type",
    "presenting_reason": "Presenting reason",
    "referral_status": "Referral status",
}

# Every other curated/intakes field, grouped for the single-record view.
DETAIL_SECTIONS = {
    "Situation": {
        "prior_living_situation_reported": "Prior living situation",
        "reported_housing_instability_days": "Days of housing instability",
        "employment_status_reported": "Employment",
        "income_status_reported": "Income",
        "primary_barrier": "Primary barrier",
        "immediate_needs": "Immediate needs",
    },
    "Plan": {
        "client_goal": "Client goal",
        "potential_referral": "Potential referral",
        "referral_status": "Referral status",
        "follow_up_days": "Follow-up in (days)",
        "follow_up_target_date": "Follow-up target date",
        "housing_eligibility_verified": "Housing eligibility verified",
        "staff_role": "Staff role",
    },
}

RECORD_COLUMNS = list(
    dict.fromkeys(
        [
            *LIST_COLUMNS,
            "intake_date",
            "program_type",
            *[column for section in DETAIL_SECTIONS.values() for column in section],
        ]
    )
)

# Leading bytes of the image formats st.image can show.
_SIGNATURES = [b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a"]


def decode_photo(encoded: object) -> bytes | str | None:
    """Image bytes (or SVG text) from a Base64 string, or None if it isn't an image.

    Accepts plain Base64 or a `data:image/...;base64,` URI.
    """
    if not isinstance(encoded, str) or not encoded.strip():
        return None
    text = encoded.strip()
    if text.startswith("data:"):
        text = text.partition(",")[2]
    try:
        data = base64.b64decode(text, validate=False)
    except (binascii.Error, ValueError):
        return None
    if any(data.startswith(signature) for signature in _SIGNATURES):
        return data
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return data
    head = data[:512].lstrip().lower()
    if head.startswith(b"<svg") or (head.startswith(b"<?xml") and b"<svg" in head):
        return data.decode("utf-8", errors="replace")
    return None


def format_value(value: object) -> str:
    """A record field as display text (missing values say so)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "Not recorded"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if hasattr(value, "strftime"):
        return value.strftime("%B %d, %Y")
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value)


_MARKDOWN_SPECIAL = set("\\`*_{}[]()#+-.!|<>~$")


def escape_markdown(text: str) -> str:
    """Plain text that st.markdown shows as written (no formatting, links or math)."""
    return "".join(f"\\{char}" if char in _MARKDOWN_SPECIAL else char for char in text)


def detail_sections(record: pd.Series) -> dict[str, pd.DataFrame]:
    """The record's DETAIL_SECTIONS as two-column (Field, Value) tables."""
    return {
        title: pd.DataFrame(
            {
                "Field": list(fields.values()),
                "Value": [format_value(record.get(column)) for column in fields],
            }
        )
        for title, fields in DETAIL_SECTIONS.items()
    }
