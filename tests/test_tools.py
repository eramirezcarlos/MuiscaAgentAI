import pytest

from muisca_agent.tools import TOOLS, ToolInputError, run_tool


def test_every_tool_definition_has_the_shape_the_api_expects():
    assert [tool["name"] for tool in TOOLS] == [
        "get_search_options",
        "search_properties",
        "get_property",
        "list_agents",
    ]
    for tool in TOOLS:
        assert set(tool) == {"name", "description", "input_schema"}
        assert len(tool["description"]) > 40
        assert tool["input_schema"]["type"] == "object"
        for name, schema in tool["input_schema"]["properties"].items():
            assert "type" in schema and "description" in schema, name


def test_search_translates_tool_arguments_to_api_parameters(api, muisca):
    result = run_tool(
        api,
        "search_properties",
        {"status_id": "FSL", "min_bedrooms": 3, "min_bathrooms": 2, "text": "ocean view", "sort_by": "price_asc", "page": 2},
    )

    assert result["total"] == 25
    assert muisca.last_params == {
        "status_id": "FSL",
        "bedrooms": "3",  # min_bedrooms -> bedrooms
        "bathrooms": "2",  # min_bathrooms -> bathrooms
        "search": "ocean view",  # text -> search
        "sort_by": "price_asc",
        "page": "2",
        "per_page": "10",
    }


def test_search_with_no_arguments_lists_everything(api, muisca):
    run_tool(api, "search_properties", {})
    assert muisca.last_params == {"page": "1", "per_page": "10"}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"city": "Miami"}, "Unknown filter"),
        ({"sort_by": "cheapest"}, "sort_by"),
        ({"page": 0}, "page"),
        ({"page": "two"}, "page"),
    ],
)
def test_search_rejects_bad_arguments_before_calling_the_api(api, muisca, args, message):
    with pytest.raises(ToolInputError, match=message):
        run_tool(api, "search_properties", args)
    assert muisca.requests == []


def test_get_property_validates_its_id(api, muisca):
    assert run_tool(api, "get_property", {"property_id": 28})["id"] == 28
    for bad in ({}, {"property_id": "28; DROP"}, {"property_id": -1}, {"property_id": True}):
        with pytest.raises(ToolInputError):
            run_tool(api, "get_property", bad)
    assert len(muisca.requests) == 1


def test_unknown_tool_is_an_error(api):
    with pytest.raises(ToolInputError, match="Unknown tool"):
        run_tool(api, "delete_property", {})
