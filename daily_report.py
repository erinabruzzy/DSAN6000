"""The once-daily generated report (no Spark, Streamlit, or network, so testable).

etl/daily_report.py fetches the data and API responses, then calls
build_report to turn them into Markdown. Everything here takes plain pandas
objects and parsed JSON, and a missing source (None) is reported, not fatal.

External sources (all keyless):
  Open-Meteo forecast   https://open-meteo.com/en/docs
  NWS active alerts     https://www.weather.gov/documentation/services-web-api
  GDELT DOC 2.0 news    https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
  Google News RSS       fallback when GDELT is rate-limited
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime

import pandas as pd

from census import ALERT_Z, changes_since, unusual_changes
from districts import by_area
from forecast import INTERVAL, NextDayForecast, latest_day_summary
from records import escape_markdown

NYC_LATITUDE, NYC_LONGITUDE = 40.7128, -74.0060

# NYC DHS declares Code Blue when it is 32°F or colder, including wind chill.
CODE_BLUE_F = 32
# NWS issues heat advisories in NYC from a heat index of about 95°F.
HEAT_F = 95
HEAVY_RAIN_IN = 1.0
HEAVY_SNOW_IN = 2.0
HIGH_WIND_MPH = 40

# How old each source's newest data may be before the report warns. The
# district census is monthly and published about a week after month-end.
# Sources not listed (the synthetic intake records) are loaded once.
STALE_DAYS = {"DHS daily census": 4, "DHS census by community district": 45}

MAX_HEADLINES = 8
# News searches match loosely, so a headline must mention one of these.
NEWS_KEYWORDS = re.compile(
    r"homeless|unhoused|shelter|encampment|evict|asylum|migrant|code blue|supportive housing",
    re.IGNORECASE,
)


@dataclass
class Report:
    markdown: str
    warnings: list[str] = field(default_factory=list)


# --- Parsing API responses ---------------------------------------------------


def parse_weather(payload: dict | None) -> pd.DataFrame | None:
    """Open-Meteo daily forecast as one row per day (°F, inches, mph)."""
    if not payload or "daily" not in payload:
        return None
    daily = payload["daily"]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "low": daily["temperature_2m_min"],
            "high": daily["temperature_2m_max"],
            "feels_low": daily["apparent_temperature_min"],
            "feels_high": daily["apparent_temperature_max"],
            "precip_chance": daily["precipitation_probability_max"],
            "precip_in": daily["precipitation_sum"],
            "snow_in": daily["snowfall_sum"],
            "wind_mph": daily["wind_speed_10m_max"],
        }
    )


def parse_alerts(payload: dict | None) -> list[dict] | None:
    """Active NWS alerts for the city, most severe first, each alert once."""
    if payload is None:
        return None
    order = {"Extreme": 0, "Severe": 1, "Moderate": 2, "Minor": 3}
    alerts = [
        {
            "event": props.get("event") or "Weather alert",
            "headline": props.get("headline") or "",
            "severity": props.get("severity") or "Unknown",
        }
        for props in (feature.get("properties", {}) for feature in payload.get("features", []))
    ]
    # The same alert is often issued once per forecast zone.
    unique = list({(a["event"], a["headline"]): a for a in alerts}.values())
    return sorted(unique, key=lambda a: order.get(a["severity"], 4))


def parse_news(payload: dict | None) -> list[dict] | None:
    """GDELT articles as title, url, source, date; newest first, no repeated titles."""
    if payload is None:
        return None
    return _clean_news(
        {
            "title": article.get("title"),
            "url": article.get("url"),
            "source": article.get("domain"),
            "date": pd.to_datetime(
                article.get("seendate"), format="%Y%m%dT%H%M%SZ", errors="coerce"
            ),
        }
        for article in payload.get("articles", [])
    )


def parse_news_rss(text: str | None) -> list[dict] | None:
    """Google News RSS items in the same shape as parse_news, or None if unreadable."""
    if text is None:
        return None
    try:
        channel = ET.fromstring(text).find("channel")
    except ET.ParseError:
        return None
    if channel is None:
        return None

    def published(item):
        try:
            return pd.Timestamp(parsedate_to_datetime(item.findtext("pubDate"))).tz_convert(None)
        except (TypeError, ValueError):
            return pd.NaT

    items = []
    for item in channel.findall("item"):
        source = item.findtext("source") or ""
        title = item.findtext("title") or ""
        # Google appends " - Source" to each title.
        if source and title.endswith(f" - {source}"):
            title = title[: -len(f" - {source}")]
        items.append({"title": title, "url": item.findtext("link"), "source": source, "date": published(item)})
    return _clean_news(items)


def merge_news(*lists: list[dict] | None) -> list[dict] | None:
    """Combine parsed headline lists (None if every source failed), newest first, once each."""
    available = [items for items in lists if items is not None]
    if not available:
        return None
    return _clean_news(item for items in available for item in items)


def _clean_news(raw_articles) -> list[dict]:
    seen, articles = set(), []
    for article in raw_articles:
        title = " ".join((article["title"] or "").split())
        url = (article["url"] or "").strip()
        # Only web links, with parentheses encoded so they can't end a Markdown link early.
        if (
            not NEWS_KEYWORDS.search(title)
            or title.lower() in seen
            or not url.startswith(("https://", "http://"))
        ):
            continue
        seen.add(title.lower())
        articles.append(
            {
                "title": title,
                "url": url.replace(" ", "%20").replace("(", "%28").replace(")", "%29"),
                "source": article["source"] or "",
                "date": article["date"],
            }
        )
    articles.sort(
        key=lambda a: a["date"] if pd.notna(a["date"]) else pd.Timestamp.min, reverse=True
    )
    return articles[:MAX_HEADLINES]


# --- Judging -----------------------------------------------------------------


def weather_hazards(days: pd.DataFrame | None) -> list[str]:
    """Plain-language hazards in the forecast that could raise shelter demand."""
    if days is None:
        return []
    hazards = []
    for _, day in days.iterrows():
        when = f"{day['date']:%A, %b %d}"
        if day["feels_low"] <= CODE_BLUE_F:
            hazards.append(
                f"{when}: feels as cold as {day['feels_low']:.0f}°F, at or below the "
                f"{CODE_BLUE_F}°F Code Blue threshold."
            )
        if day["feels_high"] >= HEAT_F:
            hazards.append(
                f"{when}: feels as hot as {day['feels_high']:.0f}°F (heat advisory level)."
            )
        if day["snow_in"] >= HEAVY_SNOW_IN:
            hazards.append(f"{when}: {day['snow_in']:.1f} in of snow forecast.")
        elif day["precip_in"] >= HEAVY_RAIN_IN:
            hazards.append(f"{when}: {day['precip_in']:.1f} in of rain forecast.")
        if day["wind_mph"] >= HIGH_WIND_MPH:
            hazards.append(f"{when}: winds up to {day['wind_mph']:.0f} mph.")
    return hazards


def source_status(latest: dict[str, pd.Timestamp | None], today: pd.Timestamp) -> pd.DataFrame:
    """Each source's newest date, its age in days, and whether it is overdue."""
    rows = []
    for source, newest in latest.items():
        if newest is None or pd.isna(newest):
            rows.append({"source": source, "newest": None, "age_days": None, "status": "Missing"})
            continue
        age = (today.normalize() - pd.Timestamp(newest).normalize()).days
        limit = STALE_DAYS.get(source)
        if limit is None:
            status = "Loaded once (synthetic)"
        else:
            status = "Overdue" if age > limit else "Current"
        rows.append(
            {"source": source, "newest": pd.Timestamp(newest), "age_days": age, "status": status}
        )
    return pd.DataFrame(rows)


# --- Building ----------------------------------------------------------------


def _signed(value) -> str:
    return "–" if pd.isna(value) else f"{int(value):+,}"


def _days(count) -> str:
    return f"{int(count)} day" + ("" if int(count) == 1 else "s")


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _warnings(status, census, latest_census, weather, alerts) -> list[str]:
    warnings = []
    for _, row in status.iterrows():
        if row["status"] == "Overdue":
            warnings.append(
                f"**{row['source']}** is {_days(row['age_days'])} old; check its ingest job."
            )
        elif row["status"] == "Missing":
            warnings.append(f"**{row['source']}** could not be read.")
    for _, alert in unusual_changes(census, latest_census).iterrows():
        direction = "rose" if alert["change"] > 0 else "fell"
        warnings.append(
            f"**{alert['series']}** {direction} by {abs(alert['change']):,} on "
            f"{latest_census:%b %d}; a typical daily change is about "
            f"{alert['typical_change']:,.0f} ({alert['z']:+.1f} standard deviations)."
        )
    warnings += [f"**Weather:** {hazard}" for hazard in weather_hazards(weather)]
    for alert in alerts or []:
        warnings.append(
            f"**NWS {escape_markdown(alert['event'])}:** {escape_markdown(alert['headline'])}"
        )
    return warnings


def build_report(
    today: pd.Timestamp,
    census: pd.DataFrame,
    districts: pd.DataFrame | None,
    intake_counts: pd.Series | None,
    next_day: NextDayForecast | None,
    weather: pd.DataFrame | None,
    alerts: list[dict] | None,
    news: list[dict] | None,
    news_source: str = "GDELT",
) -> Report:
    """The day's report as Markdown, plus the warnings it raised.

    `census` is curated dhs_daily (census_date as datetime64), `districts` is
    curated dhs_district (report_date as datetime64), and `intake_counts` is a
    complete daily series as returned by forecast.daily_counts. Pass None for
    any source that could not be read or fetched.
    """
    census = census.sort_values("census_date")
    latest_census = census["census_date"].max()
    latest_district = None if districts is None else districts["report_date"].max()
    latest_intake = None if intake_counts is None else intake_counts.index[-1]
    status = source_status(
        {
            "DHS daily census": latest_census,
            "DHS census by community district": latest_district,
            "Synthetic intake records": latest_intake,
        },
        today,
    )
    warnings = _warnings(status, census, latest_census, weather, alerts)

    sections = [
        f"# NYC shelter daily report: {today:%A, %B %d, %Y}",
        "Generated automatically from the DHS Daily Report, the DHS census by community "
        "district, the synthetic intake records, Open-Meteo, the National Weather Service, "
        f"and news from {news_source}. Intake figures describe synthetic records, not real people.",
        "## Warnings",
        "\n".join(f"- {w}" for w in warnings) if warnings else "No warnings today.",
    ]

    changes = changes_since(census, latest_census)
    total = changes.iloc[0]
    summary = f"**{total['count']:,}** individuals were in DHS shelter"
    if not pd.isna(total["previous day"]):
        summary += f", {_signed(total['previous day'])} from the previous day"
    year_rows = census.loc[
        census["census_date"] == latest_census - pd.DateOffset(years=1), "individuals"
    ]
    if not year_rows.empty:
        year_ago = int(year_rows.iloc[0])
        summary += (
            f" and {_signed(total['count'] - year_ago)} "
            f"({(total['count'] - year_ago) / year_ago:+.1%}) from a year earlier"
        )
    sections += [
        f"## Occupancy on {latest_census:%B %d, %Y}",
        summary + ".",
        _table(
            ["Group", "Individuals", "1 day", "7 days", "28 days"],
            [
                [
                    row["series"],
                    f"{row['count']:,}",
                    _signed(row["previous day"]),
                    _signed(row["a week earlier"]),
                    _signed(row["four weeks earlier"]),
                ]
                for _, row in changes.iterrows()
            ],
        ),
        "Changes are against 1, 7, and 28 days earlier. Unusual day-over-day changes "
        f"({ALERT_Z:g} or more standard deviations) appear under Warnings.",
    ]

    if districts is not None:
        boroughs = by_area(districts, latest_district)
        sections += [
            f"## Where people are sheltered ({latest_district:%B %d, %Y})",
            _table(
                ["Borough", "Individuals", "Share", "In commercial hotels"],
                [
                    [
                        escape_markdown(row["area"]),
                        f"{row['total']:,}",
                        f"{row['share']:.1%}",
                        "–" if pd.isna(row["hotel_share"]) else f"{row['hotel_share']:.0%}",
                    ]
                    for _, row in boroughs.iterrows()
                ],
            ),
        ]

    if intake_counts is not None:
        newest = latest_day_summary(intake_counts)
        line = (
            f"The newest intake day, {newest.date:%A, %B %d}, had **{newest.count:,}** records"
        )
        if newest.weekday_average is not None:
            line += (
                f" against an average of {newest.weekday_average:,.0f} on the last 4 "
                f"{newest.date:%A}s"
            )
        sections += ["## Synthetic intake records", line + "."]
        if next_day is not None:
            sections.append(
                f"Forecast for {next_day.date:%A, %B %d}: **{next_day.value:,.0f}** "
                f"({INTERVAL:.0%} range {next_day.low:,.0f} – {next_day.high:,.0f}); "
                f"method: {next_day.method}."
            )

    sections.append("## Weather in New York City")
    if weather is None:
        sections.append("The weather forecast was unavailable today.")
    else:
        sections.append(
            _table(
                ["Day", "Low / high", "Feels like", "Precipitation", "Wind"],
                [
                    [
                        f"{day['date']:%a %b %d}",
                        f"{day['low']:.0f}°F / {day['high']:.0f}°F",
                        f"{day['feels_low']:.0f}°F / {day['feels_high']:.0f}°F",
                        f"{day['precip_chance']:.0f}% chance, {day['precip_in']:.2f} in"
                        + (f" ({day['snow_in']:.1f} in snow)" if day["snow_in"] > 0 else ""),
                        f"{day['wind_mph']:.0f} mph",
                    ]
                    for _, day in weather.iterrows()
                ],
            )
        )
    if alerts is None:
        sections.append("National Weather Service alerts were unavailable today.")
    elif not alerts:
        sections.append("No active National Weather Service alerts for the city.")

    sections.append("## In the news")
    if news is None:
        sections.append("News headlines were unavailable today.")
    elif not news:
        sections.append("No recent articles matched.")
    else:
        sections.append(f"Recent articles about homelessness in New York, from {news_source}:")
        sections.append(
            "\n".join(
                f"- [{escape_markdown(item['title'])}]({item['url']})"
                f" — {escape_markdown(item['source'])}"
                + (f", {item['date']:%b %d}" if pd.notna(item["date"]) else "")
                for item in news
            )
        )

    sections += [
        "## Data sources",
        _table(
            ["Source", "Newest data", "Age", "Status"],
            [
                [
                    row["source"],
                    "–" if row["newest"] is None or pd.isna(row["newest"]) else f"{row['newest']:%b %d, %Y}",
                    "–" if pd.isna(row["age_days"]) else _days(row["age_days"]),
                    row["status"],
                ]
                for _, row in status.iterrows()
            ],
        ),
    ]
    return Report(markdown="\n\n".join(sections) + "\n", warnings=warnings)
