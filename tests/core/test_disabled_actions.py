"""Tests for WORKSPACE_MCP_DISABLED_ACTIONS."""

from types import SimpleNamespace

import pytest
from fastmcp.exceptions import ToolError

from core.disabled_actions import (
    DISABLED_ACTIONS_ENV,
    DisabledActionsConfigError,
    DisabledActionsMiddleware,
    load_disabled_actions,
    parse_disabled_actions,
)

HOUSE = "manage_event:delete, manage_task:delete ,manage_task:Clear_Completed,"


def test_absent_or_empty_blocks_nothing(monkeypatch):
    assert parse_disabled_actions(None) == {}
    assert parse_disabled_actions("  ") == {}
    monkeypatch.delenv(DISABLED_ACTIONS_ENV, raising=False)
    assert load_disabled_actions() == {}


def test_parse_groups_and_normalizes():
    assert parse_disabled_actions(HOUSE) == {
        "manage_event": frozenset({"delete"}),
        "manage_task": frozenset({"delete", "clear_completed"}),
    }


@pytest.mark.parametrize("raw", ["manage_event", "manage_event:", ":delete", "a:b:c"])
def test_malformed_value_is_a_config_error(raw):
    with pytest.raises(DisabledActionsConfigError):
        parse_disabled_actions(raw)


async def _call(tool, arguments):
    middleware = DisabledActionsMiddleware(parse_disabled_actions(HOUSE))
    reached = []

    async def call_next(context):
        reached.append(context.message.name)
        return "ran"

    context = SimpleNamespace(message=SimpleNamespace(name=tool, arguments=arguments))
    result = await middleware.on_call_tool(context, call_next)
    return result, reached


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "delete"},
        {"action": "DELETE"},
        {"action": "  Delete  "},
        {"Action": "delete"},
        {" action ": "delete"},
        {"action": 7},
        {"action": ["delete"]},
        {"action": "create", "Action": "delete"},
    ],
)
async def test_blocked_action_never_reaches_the_tool(arguments):
    with pytest.raises(ToolError, match="disabled"):
        await _call("manage_event", arguments)


@pytest.mark.asyncio
async def test_clear_completed_is_blocked_for_tasks():
    with pytest.raises(ToolError):
        await _call("manage_task", {"action": "clear_completed"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("manage_event", {"action": "create", "summary": "Dentist"}),
        ("manage_event", {"action": "update"}),
        ("manage_event", {}),
        ("manage_event", None),
        ("get_events", {"action": "delete"}),  # tool not listed
        ("manage_task", {"action": "create"}),
    ],
)
async def test_everything_else_runs(tool, arguments):
    result, reached = await _call(tool, arguments)
    assert result == "ran"
    assert reached == [tool]
