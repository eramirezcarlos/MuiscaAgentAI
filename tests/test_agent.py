"""Tests of the agent loop, with a scripted model in place of Claude."""

import json

from conftest import FakeLLM, reply, tool_call

from muisca_agent.agent import STEP_LIMIT_MESSAGE, SYSTEM_PROMPT, MuiscaAgent
from muisca_agent.tools import TOOLS


def test_question_is_answered_through_two_rounds_of_tools(api, muisca):
    llm = FakeLLM(
        reply(tool_call("call_1", "get_search_options", {}), stop_reason="tool_use"),
        reply(
            "Let me search.",
            tool_call("call_2", "search_properties", {"status_id": "FSL", "city_id": 13}),
            stop_reason="tool_use",
        ),
        reply("I found 25 listings. The first is Luxury Coastal Living (PROP-001)."),
    )
    seen = []
    agent = MuiscaAgent(llm=llm, api=api, model="test-model")

    result = agent.ask("Houses for sale in Miami?", on_tool_call=lambda name, args: seen.append(name))

    assert result.answer == "I found 25 listings. The first is Luxury Coastal Living (PROP-001)."
    assert result.tool_calls == [
        {"name": "get_search_options", "input": {}},
        {"name": "search_properties", "input": {"status_id": "FSL", "city_id": 13}},
    ]
    assert seen == ["get_search_options", "search_properties"]
    assert (result.input_tokens, result.output_tokens) == (300, 60)  # summed over three calls
    assert result.stop_reason == "end_turn"

    # Every call carries the same model, system prompt and tools.
    assert len(llm.calls) == 3
    for call in llm.calls:
        assert call["model"] == "test-model" and call["system"] == SYSTEM_PROMPT and call["tools"] == TOOLS

    # The conversation grows by two messages per round: the model's turn, then our results.
    assert [len(call["messages"]) for call in llm.calls] == [1, 3, 5]
    final = llm.calls[2]["messages"]
    assert final[0] == {"role": "user", "content": "Houses for sale in Miami?"}
    assert final[1]["role"] == "assistant" and final[1]["content"][0].id == "call_1"
    assert final[3]["content"][1].id == "call_2"  # the model's text block is kept alongside

    # Each result is tied to its call by id, and the search really reached the API.
    (options_result,) = final[2]["content"]
    assert options_result["tool_use_id"] == "call_1" and "is_error" not in options_result
    assert json.loads(options_result["content"])["cities_with_listings"][0]["city_id"] == 13
    (search_result,) = final[4]["content"]
    assert search_result["tool_use_id"] == "call_2"
    assert json.loads(search_result["content"])["total"] == 25
    assert dict(muisca.requests[-1].url.params)["city_id"] == "13"


def test_several_tool_calls_in_one_reply_get_one_result_each(api):
    llm = FakeLLM(
        reply(
            tool_call("a", "get_property", {"property_id": 28}),
            tool_call("b", "list_agents", {}),
            stop_reason="tool_use",
        ),
        reply("Done."),
    )

    MuiscaAgent(llm=llm, api=api).ask("Tell me about listing 28 and the agents")

    results = llm.calls[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert all(r["type"] == "tool_result" for r in results)


def test_tool_failure_is_reported_to_the_model_and_the_loop_continues(api):
    llm = FakeLLM(
        reply(
            tool_call("a", "get_property", {"property_id": 999}),  # the API answers 404
            tool_call("b", "search_properties", {"city": "Miami"}),  # not a valid filter
            stop_reason="tool_use",
        ),
        reply("That listing doesn't exist."),
    )

    result = MuiscaAgent(llm=llm, api=api).ask("Show me listing 999")

    not_found, bad_filter = llm.calls[1]["messages"][2]["content"]
    assert not_found == {"type": "tool_result", "tool_use_id": "a", "content": "Property not found", "is_error": True}
    assert bad_filter["is_error"] is True and "Unknown filter" in bad_filter["content"]
    assert result.answer == "That listing doesn't exist."


def test_loop_stops_at_the_step_limit(api):
    endless = [reply(tool_call(f"c{i}", "list_agents", {}), stop_reason="tool_use") for i in range(5)]
    llm = FakeLLM(*endless)

    result = MuiscaAgent(llm=llm, api=api, max_steps=3).ask("loop forever")

    assert len(llm.calls) == 3
    assert result.answer == STEP_LIMIT_MESSAGE
    assert result.stop_reason == "tool_use"


def test_history_is_sent_before_the_new_question(api):
    llm = FakeLLM(reply("It has 5 bedrooms."))
    history = [
        {"role": "user", "content": "Any houses in Miami?"},
        {"role": "assistant", "content": "Yes: Luxury Coastal Living."},
    ]

    MuiscaAgent(llm=llm, api=api).ask("How many bedrooms?", history=history)

    assert llm.calls[0]["messages"] == [*history, {"role": "user", "content": "How many bedrooms?"}]
    assert len(history) == 2  # the caller's list is not modified


def test_cut_off_and_empty_answers_are_explained(api):
    cut = MuiscaAgent(llm=FakeLLM(reply("Here are the", stop_reason="max_tokens")), api=api).ask("q")
    assert cut.answer.startswith("Here are the") and "cut short" in cut.answer

    refused = MuiscaAgent(llm=FakeLLM(reply(stop_reason="refusal")), api=api).ask("q")
    assert refused.answer == "I can't help with that request."

    empty = MuiscaAgent(llm=FakeLLM(reply()), api=api).ask("q")
    assert "rephrasing" in empty.answer
