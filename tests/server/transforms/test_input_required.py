"""Generated resource and prompt tools preserve input-required control flow."""

from __future__ import annotations

import json
from typing import Any, Literal

import pytest
from mcp.shared.exceptions import MCPError
from mcp_types import ElicitRequest, ElicitRequestFormParams, InputRequiredResult
from mcp_types import ElicitResult as ElicitResponse

from fastmcp import Client, Context, FastMCP
from fastmcp.client.elicitation import ElicitResult
from fastmcp.server.transforms import PromptsAsTools, ResourcesAsTools

_CALLS = [
    pytest.param("read_resource", {"uri": "greeting://name"}, id="resource"),
    pytest.param("read_resource", {"uri": "greeting://Hello"}, id="template"),
    pytest.param(
        "get_prompt",
        {"name": "greeting", "arguments": {"prefix": "Hello"}},
        id="prompt",
    ),
]


def _ask(field: str, message: str, state: str) -> InputRequiredResult:
    return InputRequiredResult(
        result_type="input_required",
        input_requests={
            field: ElicitRequest(
                method="elicitation/create",
                params=ElicitRequestFormParams(
                    message=message,
                    requested_schema={
                        "type": "object",
                        "properties": {field: {"type": "string"}},
                        "required": [field],
                    },
                ),
            )
        },
        request_state=state,
    )


def _server(rounds: int = 1) -> FastMCP:
    mcp = FastMCP("input-required-transforms")

    def render(ctx: Context, prefix: str) -> str | InputRequiredResult:
        responses = ctx.input_responses
        if responses is None:
            return _ask("name", "Name?", prefix)
        field = "name" if "name" in responses else "suffix"
        answer = responses[field]
        assert isinstance(answer, ElicitResponse)
        if answer.action != "accept":
            return f"Input {answer.action}"
        assert answer.content is not None
        assert ctx.request_state is not None
        if field == "name":
            greeting = f"{ctx.request_state} {answer.content['name']}"
            if rounds == 2:
                return _ask("suffix", "Suffix?", greeting)
            return greeting
        return f"{ctx.request_state}{answer.content['suffix']}"

    @mcp.resource("greeting://name")
    def resource(ctx: Context) -> str | InputRequiredResult:
        return render(ctx, "Hello")

    @mcp.resource("greeting://{prefix}")
    def template(prefix: str, ctx: Context) -> str | InputRequiredResult:
        return render(ctx, prefix)

    @mcp.prompt
    def greeting(prefix: str, ctx: Context) -> str | InputRequiredResult:
        return render(ctx, prefix)

    mcp.add_transform(ResourcesAsTools(mcp))
    mcp.add_transform(PromptsAsTools(mcp))
    return mcp


def _expected(tool: str, text: str) -> str | dict[str, Any]:
    if tool == "get_prompt":
        return {"messages": [{"role": "user", "content": text}]}
    return text


@pytest.mark.parametrize("tool,arguments", _CALLS)
@pytest.mark.parametrize("rounds", [1, 2])
async def test_generated_tool_drives_input_rounds(
    tool: str, arguments: dict[str, Any], rounds: int
):
    """Formatting the empty control result must not swallow the question."""
    asked: list[str] = []

    async def answer(message, response_type, params, ctx):
        asked.append(message)
        content = (
            response_type(name="Ada")
            if message == "Name?"
            else response_type(suffix="!")
        )
        return ElicitResult(action="accept", content=content)

    async with Client(_server(rounds), elicitation_handler=answer) as client:
        result = await client.call_tool(tool, arguments)

    data = json.loads(result.data) if tool == "get_prompt" else result.data
    assert data == _expected(tool, "Hello Ada" if rounds == 1 else "Hello Ada!")
    assert not result.is_error
    assert asked == (["Name?"] if rounds == 1 else ["Name?", "Suffix?"])


@pytest.mark.parametrize("tool,arguments", _CALLS)
@pytest.mark.parametrize("action", ["decline", "cancel"])
async def test_generated_tool_delivers_non_accepting_answer(
    tool: str, arguments: dict[str, Any], action: Literal["decline", "cancel"]
):
    """The component receives refusal instead of empty success or a retry."""

    async def answer(message, response_type, params, ctx):
        return ElicitResult(action=action)

    async with Client(_server(), elicitation_handler=answer) as client:
        result = await client.call_tool(tool, arguments)

    data = json.loads(result.data) if tool == "get_prompt" else result.data
    assert data == _expected(tool, f"Input {action}")
    assert not result.is_error


@pytest.mark.parametrize("tool,arguments", _CALLS)
async def test_generated_tool_rejects_input_required_on_legacy_connection(
    tool: str, arguments: dict[str, Any]
):
    """An unsupported era must raise its usual error, not empty success."""
    async with Client(_server(), mode="legacy") as client:
        with pytest.raises(MCPError, match="2026-07-28"):
            await client.call_tool(tool, arguments)
