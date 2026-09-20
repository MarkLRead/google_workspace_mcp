"""
Block individual actions of compound tools.

``--disabled-tools`` works per tool. Some tools bundle several operations behind
an ``action`` argument (``manage_event``: create/update/delete/rsvp), so
disabling "delete" means disabling the whole tool. This middleware refuses named
actions server-side and leaves the rest of the tool usable.

Configured with ``WORKSPACE_MCP_DISABLED_ACTIONS``, comma-separated
``tool:action`` pairs, e.g. ``manage_event:delete,manage_task:delete``.
A malformed value stops the server at start-up.
"""

import logging
import os
from typing import Dict, FrozenSet, Optional

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext

logger = logging.getLogger(__name__)

DISABLED_ACTIONS_ENV = "WORKSPACE_MCP_DISABLED_ACTIONS"


class DisabledActionsConfigError(ValueError):
    """WORKSPACE_MCP_DISABLED_ACTIONS is set but malformed."""


def _normalize(value: str) -> str:
    return value.strip().casefold()


def parse_disabled_actions(raw: Optional[str]) -> Dict[str, FrozenSet[str]]:
    """Parse ``tool:action`` pairs. Absent or empty means nothing is blocked."""
    if raw is None or not raw.strip():
        return {}
    parsed: Dict[str, set] = {}
    for entry in raw.split(","):
        if not entry.strip():
            continue
        tool, sep, action = entry.partition(":")
        tool, action = tool.strip(), _normalize(action)
        if not sep or not tool or not action or ":" in action:
            raise DisabledActionsConfigError(
                f"{DISABLED_ACTIONS_ENV}: expected tool:action, got {entry.strip()!r}"
            )
        parsed.setdefault(tool, set()).add(action)
    return {tool: frozenset(actions) for tool, actions in parsed.items()}


def load_disabled_actions() -> Dict[str, FrozenSet[str]]:
    return parse_disabled_actions(os.environ.get(DISABLED_ACTIONS_ENV))


class DisabledActionsMiddleware(Middleware):
    """Refuse a configured ``action`` of a compound tool before it runs."""

    def __init__(self, disabled: Dict[str, FrozenSet[str]]) -> None:
        self._disabled = disabled

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext):
        blocked = self._disabled.get(context.message.name)
        if blocked:
            # Match the key loosely ("Action", " action") so the check does not
            # depend on where argument-name normalisation sits in the chain.
            actions = [
                value
                for key, value in (context.message.arguments or {}).items()
                if isinstance(key, str) and _normalize(key) == "action"
            ]
            # A listed tool called with a non-string action is refused too: the
            # tool would reject it anyway, and guessing its coercion is not safe.
            if any(
                value is not None
                and (not isinstance(value, str) or _normalize(value) in blocked)
                for value in actions
            ):
                logger.warning(
                    "Refused disabled action on tool '%s'", context.message.name
                )
                raise ToolError(
                    f"The requested action of '{context.message.name}' is disabled "
                    "on this server."
                )
        return await call_next(context)
