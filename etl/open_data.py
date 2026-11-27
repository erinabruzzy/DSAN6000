"""Download a whole NYC Open Data (Socrata) dataset as strings."""

from urllib.parse import urlencode

import pandas as pd

PAGE_SIZE = 50_000


def fetch_dataset(api_url: str, order: str, expected_fields) -> pd.DataFrame:
    """Every row as strings (blanks stay ""), paging in case the dataset outgrows a page."""
    pages = []
    offset = 0
    while True:
        query = urlencode({"$order": order, "$limit": PAGE_SIZE, "$offset": offset})
        page = pd.read_csv(f"{api_url}?{query}", dtype=str, keep_default_na=False)
        pages.append(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    rows = pd.concat(pages, ignore_index=True)

    missing = sorted(set(expected_fields) - set(rows.columns))
    if missing:
        raise SystemExit(f"The API response is missing expected fields: {missing}")
    if rows.empty:
        raise SystemExit("The API returned no rows; leaving S3 unchanged.")
    return rows
