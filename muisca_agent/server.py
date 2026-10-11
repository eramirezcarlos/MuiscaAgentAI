"""HTTP wrapper around the agent, so a web page can use it.

    uvicorn muisca_agent.server:app --port 8100

    POST /chat    {"message": "...", "history": [{"role": "user", "content": "..."}, ...]}
    GET  /health

This endpoint has no authentication or rate limiting. Each request spends API credits,
so add both before exposing it to the internet.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import anthropic
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config
from .agent import MuiscaAgent

app = FastAPI(title="Muisca property assistant")

# CORS is what lets a browser page on another origin (the Next.js site) call this API.
if config.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_methods=["POST"],
        allow_headers=["Content-Type"],
    )


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=20)


class ToolCall(BaseModel):
    name: str
    input: dict


class ChatResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int


@lru_cache
def _shared_agent() -> MuiscaAgent:
    return MuiscaAgent()


def get_agent() -> MuiscaAgent:
    """One agent for the whole process, created on first use."""
    if not config.api_key_is_set():
        raise HTTPException(status_code=503, detail="The assistant is not configured (ANTHROPIC_API_KEY is missing).")
    return _shared_agent()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# A plain `def` endpoint runs in a worker thread, so the blocking calls inside the
# agent do not freeze the server for other visitors.
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, agent: MuiscaAgent = Depends(get_agent)) -> ChatResponse:
    history = [turn.model_dump() for turn in request.history]
    try:
        result = agent.ask(request.message, history=history)
    except anthropic.APIError as error:
        # Don't leak provider error details to visitors; the server log has them.
        raise HTTPException(status_code=502, detail="The assistant is unavailable right now.") from error
    return ChatResponse(
        answer=result.answer,
        tool_calls=[ToolCall(**call) for call in result.tool_calls],
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
