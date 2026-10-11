"""Command line entry point.

    python -m muisca_agent "3-bedroom houses for sale in Miami under 4 million"
    python -m muisca_agent                 # interactive chat that remembers the conversation
    python -m muisca_agent -v "..."        # also print each tool call and the tokens used
"""

from __future__ import annotations

import argparse
import json
import sys

import anthropic

from . import config
from .agent import AgentResult, MuiscaAgent


def _print_result(result: AgentResult, verbose: bool) -> None:
    print(result.answer)
    if verbose:
        print(f"\n[{result.input_tokens} input tokens, {result.output_tokens} output tokens]", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(prog="muisca_agent", description="Ask about Muisca Real Estate listings.")
    parser.add_argument("question", nargs="?", help="A question. Leave it out to start an interactive chat.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show tool calls and token usage.")
    args = parser.parse_args()

    def show_tool_call(name: str, tool_input: dict) -> None:
        print(f"[tool] {name} {json.dumps(tool_input, ensure_ascii=False)}", file=sys.stderr)

    if not config.api_key_is_set():
        print("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.", file=sys.stderr)
        return 1

    agent = MuiscaAgent()
    on_tool_call = show_tool_call if args.verbose else None

    try:
        if args.question:
            _print_result(agent.ask(args.question, on_tool_call=on_tool_call), args.verbose)
            return 0

        print("Ask about Muisca listings. Press Ctrl+D or type 'exit' to quit.")
        history: list[dict[str, str]] = []
        while True:
            try:
                question = input("\nyou> ").strip()
            except EOFError:
                print()
                return 0
            if question.lower() in {"exit", "quit"}:
                return 0
            if not question:
                continue
            result = agent.ask(question, history=history, on_tool_call=on_tool_call)
            print()
            _print_result(result, args.verbose)
            # Only the visible text is remembered; tool results are fetched fresh each turn.
            history += [{"role": "user", "content": question}, {"role": "assistant", "content": result.answer}]
    except anthropic.APIError as error:
        print(f"The Claude API returned an error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
