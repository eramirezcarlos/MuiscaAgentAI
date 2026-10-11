import httpx
import pytest

from muisca_agent.muisca_api import MuiscaApi, MuiscaApiError, shorten, strip_html


def test_strip_html_turns_markup_into_readable_text():
    raw = "Vista Mar Golf &amp; Beach.<br>THE VILLA<br><br><h6><b>GROUND FLOOR</b></h6><ul><li>&nbsp; &nbsp; TV room,</li></ul>"
    assert strip_html(raw) == "Vista Mar Golf & Beach.\nTHE VILLA\nGROUND FLOOR\nTV room,"
    assert strip_html(None) == ""


def test_shorten_cuts_at_a_word_and_marks_the_cut():
    assert shorten("short", 20) == "short"
    assert shorten("one two three four", 10) == "one two…"


def test_search_returns_clean_summaries(api):
    result = api.search_properties()

    assert (result["total"], result["page"], result["total_pages"]) == (25, 1, 3)
    villa, house, penthouse = result["properties"]

    assert house["id"] == 28
    assert house["price"] == 3_000_000.0  # the API sends "3000000.00"
    assert house["status"] == "For Sale" and house["type"] == "House"
    assert (house["bedrooms"], house["bathrooms"], house["area_sqft"]) == (5, 4, 1000)
    assert (house["city"], house["state"], house["country"]) == ("Miami", "Florida", "United States")
    assert house["featured"] is True and penthouse["featured"] is False

    assert "<" not in villa["summary"] and "&amp;" not in villa["summary"]
    assert len(villa["summary"]) <= 241


def test_search_sends_filters_as_query_parameters(api, muisca):
    api.search_properties({"status_id": "FSL", "city_id": 13, "max_price": 500000, "search": None}, page=2)

    assert muisca.last_params == {
        "status_id": "FSL",
        "city_id": "13",
        "max_price": "500000",
        "page": "2",
        "per_page": "10",
    }
    assert muisca.requests[-1].headers["accept"] == "application/json"


def test_get_property_returns_full_details(api):
    prop = api.get_property(28)

    assert prop["title"] == "Luxury Coastal Living"
    assert (prop["lot_size"], prop["parking_spaces"], prop["year_built"]) == (10000, 4, 1990)
    assert prop["location"] == {
        "address": "4821 Ocean Breeze Drive",
        "city": "Miami",
        "state": "Florida",
        "country": "United States",
    }
    assert prop["agent"] == {"id": 1, "name": "Demo Agent", "email": "agent@example.com"}
    assert prop["photo_count"] == 2
    assert prop["description"].startswith("Experience luxury coastal living in this stunning")


def test_missing_property_raises_the_api_message(api):
    with pytest.raises(MuiscaApiError, match="Property not found"):
        api.get_property(999)


def test_search_options_combines_the_three_lookups(api, muisca):
    options = api.search_options()

    assert {"status_id": "FRT", "name": "For Rent"} in options["statuses"]
    assert {"type_id": "HOUS", "name": "House"} in options["property_types"]
    assert options["cities_with_listings"][0] == {
        "city_id": 13,
        "name": "Miami",
        "state_code": "FL",
        "country_code": "US",
        "listings": 2,
    }
    assert len(muisca.requests) == 3


def test_list_agents(api):
    (agent,) = api.list_agents()
    assert agent["agent_id"] == 1 and agent["listings"] == 25 and agent["based_in"] == "Miami"
    assert agent["bio"] == "Ten years selling coastal homes."


def test_non_json_error_and_network_failure_become_api_errors():
    def broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="<html>Server Error</html>")

    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(MuiscaApiError, match="HTTP 500"):
        MuiscaApi("http://x.test", httpx.Client(transport=httpx.MockTransport(broken))).list_agents()
    with pytest.raises(MuiscaApiError, match="Could not reach"):
        MuiscaApi("http://x.test", httpx.Client(transport=httpx.MockTransport(offline))).list_agents()
