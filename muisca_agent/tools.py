"""The tools the model can call.

Each tool has two halves:

  1. A definition (name, description, input_schema) that is sent to the model. This is
     all the model knows about the tool, so the descriptions are written as carefully
     as a prompt: they say when to use the tool and what each argument means.
  2. A Python function that runs when the model asks for the tool.

`run_tool` connects the two.
"""

from __future__ import annotations

from typing import Any

from .muisca_api import MuiscaApi

SORT_OPTIONS = ["newest", "oldest", "price_asc", "price_desc", "sqft_asc", "sqft_desc"]


class ToolInputError(Exception):
    """The model sent arguments the tool cannot use. The message goes back to the model."""


# ------------------------------------------------------------------- definitions

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_search_options",
        "description": (
            "Returns the valid values for search_properties filters: status codes (for sale, "
            "for rent, sold), property type codes (house, apartment, land, ...) and the cities "
            "that currently have listings, each with its city_id. Call this before the first "
            "search in a conversation, because filters take codes and ids, not names."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "search_properties",
        "description": (
            "Searches the listings. Every filter is optional and filters are combined with AND. "
            "Returns up to 10 short summaries per page plus the total number of matches. "
            "Use get_property for the full details of one listing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status_id": {
                    "type": "string",
                    "description": "Status code from get_search_options, e.g. 'FSL' (for sale) or 'FRT' (for rent).",
                },
                "type_id": {
                    "type": "string",
                    "description": "Property type code from get_search_options, e.g. 'HOUS' or 'APRT'.",
                },
                "city_id": {"type": "integer", "description": "City id from get_search_options."},
                "state_code": {"type": "string", "description": "State or province code, e.g. 'FL'."},
                "country_code": {"type": "string", "description": "Two-letter country code, e.g. 'US' or 'PA'."},
                "min_price": {"type": "number", "description": "Lowest price to include."},
                "max_price": {"type": "number", "description": "Highest price to include."},
                "min_bedrooms": {"type": "integer", "description": "Minimum number of bedrooms (1 to 5)."},
                "min_bathrooms": {"type": "integer", "description": "Minimum number of bathrooms (1 to 5)."},
                "min_sqft": {"type": "number", "description": "Minimum interior area in square feet."},
                "max_sqft": {"type": "number", "description": "Maximum interior area in square feet."},
                "text": {
                    "type": "string",
                    "description": (
                        "Words to look for in the title, description and address, e.g. 'ocean view'. "
                        "Matches the exact phrase, so keep it to one or two words."
                    ),
                },
                "agent_id": {"type": "integer", "description": "Only listings handled by this agent."},
                "sort_by": {
                    "type": "string",
                    "enum": SORT_OPTIONS,
                    "description": "Order of results. Default is 'newest', with featured listings first.",
                },
                "page": {"type": "integer", "description": "Page of results, starting at 1."},
            },
            "required": [],
        },
    },
    {
        "name": "get_property",
        "description": (
            "Returns the full details of one listing: complete description, lot size, parking, "
            "year built, amenities, location and the agent's name and email. "
            "Takes the numeric id from search_properties."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"property_id": {"type": "integer", "description": "The listing's numeric id."}},
            "required": ["property_id"],
        },
    },
    {
        "name": "list_agents",
        "description": (
            "Returns the real estate agents with their contact details and how many listings each "
            "one handles. Use the agent_id with search_properties to see an agent's listings."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

# --------------------------------------------------------------------- execution

# Tool argument name -> the Laravel API's query parameter name.
_SEARCH_PARAMS = {
    "status_id": "status_id",
    "type_id": "type_id",
    "city_id": "city_id",
    "state_code": "state_code",
    "country_code": "country_code",
    "min_price": "min_price",
    "max_price": "max_price",
    "min_bedrooms": "bedrooms",
    "min_bathrooms": "bathrooms",
    "min_sqft": "min_sqft",
    "max_sqft": "max_sqft",
    "text": "search",
    "agent_id": "agent_id",
    "sort_by": "sort_by",
}


def _positive_int(value: Any, name: str) -> int:
    # bool is a subclass of int in Python, so it has to be excluded by hand.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) or value < 1:
        raise ToolInputError(f"'{name}' must be a whole number of 1 or more.")
    return int(value)


def _search_properties(api: MuiscaApi, args: dict[str, Any]) -> dict[str, Any]:
    unknown = set(args) - set(_SEARCH_PARAMS) - {"page"}
    if unknown:
        # Laravel ignores parameters it does not know, which would silently return
        # unfiltered results. Rejecting them lets the model correct itself.
        raise ToolInputError(f"Unknown filter(s): {', '.join(sorted(unknown))}.")
    if args.get("sort_by") not in (None, *SORT_OPTIONS):
        raise ToolInputError(f"'sort_by' must be one of: {', '.join(SORT_OPTIONS)}.")

    page = _positive_int(args.get("page", 1), "page")
    filters = {_SEARCH_PARAMS[name]: value for name, value in args.items() if name != "page"}
    return api.search_properties(filters, page=page)


def _get_property(api: MuiscaApi, args: dict[str, Any]) -> dict[str, Any]:
    if "property_id" not in args:
        raise ToolInputError("'property_id' is required.")
    return api.get_property(_positive_int(args["property_id"], "property_id"))


def run_tool(api: MuiscaApi, name: str, args: dict[str, Any]) -> Any:
    """Runs one tool call and returns its result as plain data (dicts and lists)."""
    if name == "get_search_options":
        return api.search_options()
    if name == "search_properties":
        return _search_properties(api, args)
    if name == "get_property":
        return _get_property(api, args)
    if name == "list_agents":
        return api.list_agents()
    raise ToolInputError(f"Unknown tool: {name}.")
