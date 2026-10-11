"""Client for the public Muisca REST API (the Laravel routes under /api/public).

This file knows nothing about AI. It calls the API and reshapes each response into a
small, clean dictionary, so the model receives only what it needs:

  - HTML is stripped from descriptions (the model reads text, and markup costs tokens).
  - Long descriptions are shortened in search results.
  - Field names are made self-explanatory (area_size -> area_sqft).
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Any

import httpx

from . import config

SUMMARY_LENGTH = 240  # characters of description kept per search result
DESCRIPTION_LENGTH = 3000  # characters of description kept for a single property
PAGE_SIZE = 10


class MuiscaApiError(Exception):
    """The API could not answer. The message is written to be shown to the model."""


# --------------------------------------------------------------------------- text


class _TextExtractor(HTMLParser):
    """Collects the text of an HTML fragment, turning block tags into line breaks."""

    BLOCK_TAGS = {"br", "p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def strip_html(value: str | None) -> str:
    """'<b>Sea view</b><br>2 beds &amp; a pool' -> 'Sea view\\n2 beds & a pool'"""
    if not value:
        return ""
    extractor = _TextExtractor()
    extractor.feed(value)
    extractor.close()
    text = html.unescape("".join(extractor.parts)).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def shorten(text: str, limit: int) -> str:
    """Cuts at a word boundary and marks the cut, so the model knows there is more."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",.;:") + "…"


def _to_number(value: Any) -> float | None:
    """The API sends prices as strings ("3000000.00")."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------- client


class MuiscaApi:
    def __init__(self, base_url: str | None = None, http: httpx.Client | None = None) -> None:
        self.base_url = (base_url or config.MUISCA_API_URL).rstrip("/")
        # Accept: application/json makes Laravel answer errors as JSON, not HTML pages.
        self.http = http or httpx.Client(timeout=10, headers={"Accept": "application/json"})

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/api/public{path}"
        try:
            response = self.http.get(url, params=params)
        except httpx.HTTPError as error:
            raise MuiscaApiError(f"Could not reach the Muisca API ({type(error).__name__}).") from error

        if response.is_success:
            return response.json()

        # Laravel error bodies look like {"message": "Property not found"}.
        try:
            message = response.json().get("message")
        except ValueError:
            message = None
        raise MuiscaApiError(message or f"The Muisca API answered with HTTP {response.status_code}.")

    # -- reference data ---------------------------------------------------------

    def search_options(self) -> dict[str, Any]:
        """The codes and ids the search filters accept. One tool call instead of three."""
        statuses = self._get("/property-statuses/lookup")
        types = self._get("/property-types/lookup")
        cities = self._get("/cities-with-properties")
        return {
            "statuses": [{"status_id": item["id"], "name": item["name"]} for item in statuses],
            "property_types": [{"type_id": item["id"], "name": item["name"]} for item in types],
            "cities_with_listings": [
                {
                    "city_id": item["city_id"],
                    "name": item["city_name"],
                    "state_code": item["state_code"],
                    "country_code": item["country_code"],
                    "listings": item["property_count"],
                }
                for item in cities
            ],
        }

    # -- properties -------------------------------------------------------------

    def search_properties(self, filters: dict[str, Any] | None = None, page: int = 1) -> dict[str, Any]:
        """`filters` uses the API's own parameter names (status_id, min_price, ...)."""
        params = {key: value for key, value in (filters or {}).items() if value is not None}
        params.update({"page": page, "per_page": PAGE_SIZE})
        data = self._get("/properties/search", params)

        pagination = data.get("pagination", {})
        return {
            "total": pagination.get("total", len(data.get("data", []))),
            "page": pagination.get("current_page", page),
            "total_pages": pagination.get("total_pages", 1),
            "properties": [self._summary(item) for item in data.get("data", [])],
        }

    @staticmethod
    def _summary(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": item["id"],
            "property_code": item.get("property_code"),
            "title": item.get("title"),
            "price": _to_number(item.get("price")),
            "status": item.get("status_name"),
            "type": item.get("type_name"),
            "bedrooms": item.get("bedroom"),
            "bathrooms": item.get("bathroom"),
            "area_sqft": item.get("area_size"),
            "address": item.get("formatted_address"),
            "city": item.get("city_name") or item.get("locality"),
            "state": item.get("administrative_area_level_1"),
            "country": item.get("country_name"),
            "agent_id": item.get("agent_id"),
            "featured": item.get("featured") == "Y",
            "summary": shorten(strip_html(item.get("description")), SUMMARY_LENGTH),
        }

    def get_property(self, property_id: int) -> dict[str, Any]:
        data = self._get(f"/properties/{int(property_id)}")
        item = data["property"]
        location = item.get("location") or {}
        agent = item.get("agent") or {}
        return {
            "id": item["id"],
            "property_code": item.get("property_code"),
            "title": item.get("title"),
            "price": _to_number(item.get("price")),
            "status": item.get("status_name"),
            "type": item.get("type_name"),
            "bedrooms": item.get("bedroom"),
            "bathrooms": item.get("bathroom"),
            "area_sqft": item.get("area_size"),
            "lot_size": item.get("lot_size"),
            "parking_spaces": item.get("parking"),
            "year_built": item.get("year_built"),
            "featured": bool(item.get("featured")),
            "location": {
                "address": location.get("formatted_address"),
                "city": location.get("city") or location.get("locality"),
                "state": location.get("state"),
                "country": location.get("country"),
            },
            "agent": {"id": agent.get("id"), "name": agent.get("name"), "email": agent.get("email")},
            # Each amenity may be a plain name or an object with a "name" field.
            "amenities": [a.get("name") if isinstance(a, dict) else a for a in item.get("amenities") or []],
            "photo_count": len(data.get("picture_list") or []),
            "description": shorten(strip_html(item.get("description")), DESCRIPTION_LENGTH),
        }

    # -- agents -----------------------------------------------------------------

    def list_agents(self) -> list[dict[str, Any]]:
        data = self._get("/agents")
        return [
            {
                "agent_id": item["id"],
                "name": item.get("name"),
                "email": item.get("email"),
                "office_phone": item.get("telefono_office"),
                "whatsapp": item.get("telefono_whatsapp"),
                "listings": item.get("properties_count"),
                "based_in": item.get("primary_city_name"),
                "bio": shorten(strip_html(item.get("bio")), SUMMARY_LENGTH),
            }
            for item in data.get("data", [])
        ]
