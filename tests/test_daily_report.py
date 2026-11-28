"""Tests for daily_report.py with small synthetic tables and API payloads."""

import pandas as pd

from census import ALERT_COLUMNS, GROUPS, SUBGROUPS
from daily_report import (
    build_report,
    merge_news,
    parse_alerts,
    parse_news,
    parse_news_rss,
    parse_weather,
    source_status,
    weather_hazards,
)
from districts import FACILITIES

TODAY = pd.Timestamp("2026-11-27")


def census(days=400, end="2026-11-25", jump=0):
    dates = pd.date_range(end=end, periods=days, freq="D")
    table = pd.DataFrame({"census_date": dates})
    columns = {"individuals", "adults", "children", *GROUPS, *(c for _, _, c in SUBGROUPS)}
    for column in columns:
        table[column] = 1000 + (pd.Series(range(days)) % 2)
    table.loc[table.index[-1], "individuals"] += jump
    return table


def weather_payload(feels_low=40.0, snow=0.0):
    return {
        "daily": {
            "time": ["2026-11-27", "2026-11-28"],
            "temperature_2m_min": [38.0, 35.0],
            "temperature_2m_max": [50.0, 45.0],
            "apparent_temperature_min": [feels_low, 33.0],
            "apparent_temperature_max": [48.0, 42.0],
            "precipitation_probability_max": [10, 80],
            "precipitation_sum": [0.0, 0.4],
            "snowfall_sum": [snow, 0.0],
            "wind_speed_10m_max": [12.0, 18.0],
        }
    }


def test_parse_weather_and_hazards():
    calm = parse_weather(weather_payload())
    assert list(calm["low"]) == [38.0, 35.0]
    assert weather_hazards(calm) == []

    cold = weather_hazards(parse_weather(weather_payload(feels_low=25.0, snow=3.0)))
    assert any("Code Blue" in hazard for hazard in cold)
    assert any("snow" in hazard for hazard in cold)
    assert parse_weather(None) is None and weather_hazards(None) == []


def test_parse_alerts_dedupes_and_orders_by_severity():
    payload = {
        "features": [
            {"properties": {"event": "Wind Advisory", "headline": "Windy", "severity": "Minor"}},
            {"properties": {"event": "Winter Storm Warning", "headline": "Snow", "severity": "Severe"}},
            {"properties": {"event": "Wind Advisory", "headline": "Windy", "severity": "Minor"}},
        ]
    }
    alerts = parse_alerts(payload)
    assert [a["event"] for a in alerts] == ["Winter Storm Warning", "Wind Advisory"]
    assert parse_alerts({"features": []}) == [] and parse_alerts(None) is None


def test_parse_news_keeps_web_links_once_newest_first():
    payload = {
        "articles": [
            {"title": "Older shelter story", "url": "https://a.example/1", "domain": "a.example", "seendate": "20261125T100000Z"},
            {"title": "Newer  shelter story", "url": "https://b.example/x_(1)", "domain": "b.example", "seendate": "20261127T100000Z"},
            {"title": "newer shelter story", "url": "https://c.example/2", "domain": "c.example", "seendate": "20261127T110000Z"},
            {"title": "Bad link shelter", "url": "javascript:alert(1)", "domain": "d.example", "seendate": "20261127T120000Z"},
            {"title": "Storm floods roads", "url": "https://e.example", "domain": "e.example", "seendate": "20261127T130000Z"},
        ]
    }
    news = parse_news(payload)
    assert [item["title"] for item in news] == ["Newer shelter story", "Older shelter story"]
    assert news[0]["url"] == "https://b.example/x_%281%29"


def test_source_status():
    status = source_status(
        {
            "DHS daily census": pd.Timestamp("2026-11-25"),
            "DHS census by community district": pd.Timestamp("2026-09-30"),
            "Synthetic intake records": pd.Timestamp("2026-11-11"),
            "Something else": None,
        },
        TODAY,
    ).set_index("source")["status"]
    assert status["DHS daily census"] == "Current"
    assert status["DHS census by community district"] == "Overdue"  # 58 days > 45
    assert status["Synthetic intake records"] == "Loaded once (synthetic)"
    assert status["Something else"] == "Missing"


def districts():
    row = {"report_date": pd.Timestamp("2026-10-31"), "borough": "Bronx", "community_district": 1}
    return pd.DataFrame([{**row, **{column: 10 for column in FACILITIES}}])


def test_report_with_every_source():
    counts = pd.Series(400, index=pd.date_range(end="2026-11-11", periods=60, freq="D"))
    news = parse_news(
        {"articles": [{"title": "A *shelter* story", "url": "https://a.example", "domain": "a.example", "seendate": "20261127T100000Z"}]}
    )
    report = build_report(
        TODAY, census(), districts(), counts, None, parse_weather(weather_payload()), [], news
    )
    text = report.markdown
    assert text.startswith("# NYC shelter daily report: Friday, November 27, 2026")
    assert "No warnings today." in text and report.warnings == []
    for heading in [
        "## Occupancy on November 25, 2026",
        "## Where people are sheltered (October 31, 2026)",
        "## Synthetic intake records",
        "## Weather in New York City",
        "## In the news",
    ]:
        assert heading in text
    assert "from a year earlier" in text
    assert r"A \*shelter\* story" in text  # headlines can't add formatting
    # Occupancy table: header row plus one row per series.
    assert all(f"| {name} |" in text for name in ALERT_COLUMNS.values())


def test_report_warns_and_survives_missing_sources():
    report = build_report(
        TODAY,
        census(end="2026-11-15", jump=500),
        None,
        None,
        None,
        parse_weather(weather_payload(feels_low=20.0)),
        None,
        None,
    )
    joined = " ".join(report.warnings)
    assert "is 12 days old" in joined  # the daily census is overdue
    assert "could not be read" in joined  # districts and intakes are missing
    assert "**Total individuals** rose by" in joined  # the jump on the newest census day
    assert "Code Blue" in joined
    assert "## Warnings" in report.markdown and "Code Blue" in report.markdown
    assert "News headlines were unavailable today." in report.markdown
    assert "National Weather Service alerts were unavailable today." in report.markdown


RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Shelter story - The City</title><link>https://news.example/a</link>
<pubDate>Fri, 27 Nov 2026 14:35:45 GMT</pubDate><source url="https://thecity.nyc">The City</source></item>
<item><title>Older shelter story - Gothamist</title><link>https://news.example/b</link>
<pubDate>Wed, 25 Nov 2026 09:00:00 GMT</pubDate><source url="https://gothamist.com">Gothamist</source></item>
</channel></rss>"""


def test_parse_news_rss_matches_the_gdelt_shape():
    news = parse_news_rss(RSS)
    assert [item["title"] for item in news] == ["Shelter story", "Older shelter story"]
    assert news[0]["source"] == "The City"
    assert news[0]["date"] == pd.Timestamp("2026-11-27 14:35:45")
    assert parse_news_rss(None) is None and parse_news_rss("not xml") is None


def test_report_names_its_news_source():
    report = build_report(
        TODAY, census(), None, None, None, None, [], parse_news_rss(RSS), news_source="Google News"
    )
    assert "from Google News:" in report.markdown
    assert "[Shelter story](https://news.example/a) — The City, Nov 27" in report.markdown


def test_merge_news_dedupes_across_sources():
    gdelt = parse_news(
        {"articles": [{"title": "Shelter story", "url": "https://g.example", "domain": "g.example", "seendate": "20261126T100000Z"}]}
    )
    merged = merge_news(gdelt, parse_news_rss(RSS))
    assert [item["title"] for item in merged] == ["Shelter story", "Older shelter story"]
    assert merged[0]["source"] == "g.example"  # the first source's copy is kept
    assert merge_news(None, None) is None
    assert merge_news(None, []) == []
