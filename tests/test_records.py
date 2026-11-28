"""Tests for records.py."""

import base64
import datetime

import numpy as np
import pandas as pd

from records import (
    DETAIL_SECTIONS,
    RECORD_COLUMNS,
    decode_photo,
    detail_sections,
    escape_markdown,
    format_value,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16


def encode(data: bytes) -> str:
    return base64.b64encode(data).decode()


def test_decodes_plain_base64_images():
    assert decode_photo(encode(PNG)) == PNG
    assert decode_photo(encode(JPEG)) == JPEG


def test_decodes_a_data_uri():
    assert decode_photo(f"data:image/png;base64,{encode(PNG)}") == PNG


def test_decodes_svg_as_text():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'
    assert decode_photo(encode(svg)) == svg.decode()


def test_rejects_missing_or_non_image_values():
    assert decode_photo(None) is None
    assert decode_photo(np.nan) is None
    assert decode_photo("   ") is None
    assert decode_photo(encode(b"just some text")) is None
    assert decode_photo("not base64 at all!!") is None


def test_format_value():
    assert format_value(None) == "Not recorded"
    assert format_value(np.nan) == "Not recorded"
    assert format_value(True) == "Yes"
    assert format_value(np.bool_(False)) == "No"
    assert format_value(datetime.date(2026, 3, 5)) == "March 05, 2026"
    assert format_value(1200.0) == "1,200"
    assert format_value(np.int64(42)) == "42"
    assert format_value("Shelter") == "Shelter"


def test_detail_sections_cover_every_field_once_in_order():
    record = pd.Series({column: f"value of {column}" for column in RECORD_COLUMNS})
    sections = detail_sections(record)

    assert list(sections) == list(DETAIL_SECTIONS)
    for title, fields in DETAIL_SECTIONS.items():
        assert list(sections[title]["Field"]) == list(fields.values())
        assert list(sections[title]["Value"]) == [f"value of {c}" for c in fields]
    assert len(RECORD_COLUMNS) == len(set(RECORD_COLUMNS))


def test_escape_markdown_leaves_no_formatting():
    assert escape_markdown("Needs *urgent* care: $40 #1") == r"Needs \*urgent\* care: \$40 \#1"
    assert escape_markdown("plain words") == "plain words"
