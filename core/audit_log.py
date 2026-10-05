"""
Audit log of tool calls.

The normal logs of a household deployment run at WARNING, because the INFO lines
carry search text, addresses and ids. That leaves no record of WHICH tools ran:
after the fact nobody can say whether a session read mail or moved it to Trash.

This middleware writes one JSON line per tool call to a separate file, for every
tool of every service (Gmail, Drive, Docs, Sheets, Slides, Calendar, Tasks,
Contacts): when, which tool, which action, which arguments were given (their
names only), every yes/no argument with its value (``trashed``, ``verify``, ...),
how many items, which Gmail system labels were added or removed, and how the call
ended.

It never writes argument content. No search text, subject, body, address or user
label name reaches the file. Ids are written only as fingerprints (the first 12
hex digits of their SHA-256), so "did any call touch THIS document or message?"
can be answered by someone who already holds its id, and by nobody else:

    python -c "from core.audit_log import fingerprint; print(fingerprint('<id>'))"

then search the audit log for the result.

Configured with ``WORKSPACE_MCP_AUDIT_LOG``, the path of the file. Unset or blank
means no audit log and nothing changes. A path that cannot be opened for append
stops the server at start-up: a deployment that asked for an audit log must not
run without one.

A write that fails later (disk full, file removed) does not fail the tool call.
It is reported once per failure at WARNING in the normal log.
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext

logger = logging.getLogger(__name__)

AUDIT_LOG_ENV = "WORKSPACE_MCP_AUDIT_LOG"

# Gmail system labels are fixed names, not user content, so they are safe to record.
_SYSTEM_LABELS = frozenset(
    {
        "INBOX",
        "TRASH",
        "SPAM",
        "UNREAD",
        "STARRED",
        "IMPORTANT",
        "SENT",
        "DRAFT",
        "CHAT",
    }
)
_ACTION_RE = re.compile(r"[a-z_]{1,40}")
_CATEGORY_RE = re.compile(r"CATEGORY_[A-Z]{1,20}")
_TOOL_RE = re.compile(r"[A-Za-z0-9_.-]{1,80}")
_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,39}")
_MAX_LABELS = 10
_MAX_NAMES = 40
_MAX_IDS = 1000


class AuditLogConfigError(ValueError):
    """WORKSPACE_MCP_AUDIT_LOG is set but the file cannot be opened."""


def _key(value: Any) -> str:
    return value.strip().casefold() if isinstance(value, str) else ""


def _normalized(arguments: Optional[Dict[Any, Any]]) -> Dict[str, Any]:
    """Arguments keyed loosely ("Action", " action "), like disabled_actions does."""
    out: Dict[str, Any] = {}
    for key, value in (arguments or {}).items():
        name = _key(key)
        if name and name not in out:
            out[name] = value
    return out


def _action(arguments: Dict[str, Any]) -> Optional[str]:
    """The ``action`` argument, only when it looks like an action name."""
    if "action" not in arguments:
        return None
    value = _key(arguments["action"])
    return value if _ACTION_RE.fullmatch(value) else "?"


def _labels(value: Any) -> Optional[List[str]]:
    """System label names as given; every other label collapses to "USER"."""
    if not isinstance(value, (list, tuple)) or not value:
        return None
    seen: List[str] = []
    for item in value:
        name = item.strip().upper() if isinstance(item, str) else ""
        label = (
            name if name in _SYSTEM_LABELS or _CATEGORY_RE.fullmatch(name) else "USER"
        )
        if label not in seen:
            seen.append(label)
    return seen[:_MAX_LABELS]


def _items(arguments: Dict[str, Any]) -> Optional[int]:
    """How many things the call names: a count, never the ids themselves."""
    for key, value in arguments.items():
        if "label" in key:
            continue
        if (key.endswith("_ids") or key == "updates") and isinstance(
            value, (list, tuple)
        ):
            return len(value)
    for key, value in arguments.items():
        if "label" in key:
            continue
        if key.endswith("_id") and isinstance(value, str) and value.strip():
            return 1
    return None


def fingerprint(value: str) -> str:
    """One-way tag of an id: enough to recognise it, not to recover it."""
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()[:12]


def _ids(arguments: Dict[str, Any]) -> Tuple[List[str], int]:
    """Fingerprints of the ids a call names, and how many were left out."""
    found: List[str] = []
    total = 0
    for key, value in arguments.items():
        if "label" in key:
            continue
        if key.endswith("_ids") and isinstance(value, (list, tuple)):
            candidates = [item for item in value if isinstance(item, str)]
        elif key.endswith("_id") and isinstance(value, str):
            candidates = [value]
        elif isinstance(value, (list, tuple)):
            # batch tools take a list of dicts, each naming its own id
            candidates = [
                inner
                for entry in value
                if isinstance(entry, dict)
                for name, inner in entry.items()
                if _key(name).endswith("_id")
                and "label" not in _key(name)
                and isinstance(inner, str)
            ]
        else:
            continue
        for item in candidates:
            if not item.strip():
                continue
            total += 1
            if len(found) < _MAX_IDS:
                found.append(fingerprint(item))
    return found, total - len(found)


def _argument_names(arguments: Dict[str, Any]) -> List[str]:
    """Names of the arguments that were given a value. Names only."""
    names = sorted(
        key
        for key, value in arguments.items()
        if value is not None and _NAME_RE.fullmatch(key)
    )
    return names[:_MAX_NAMES]


def _flags(arguments: Dict[str, Any]) -> Dict[str, bool]:
    """Every yes/no argument with its value; a bool carries no content."""
    flags = {
        key: value
        for key, value in arguments.items()
        if isinstance(value, bool) and _NAME_RE.fullmatch(key)
    }
    return {key: flags[key] for key in sorted(flags)[:_MAX_NAMES]}


def _session(context: MiddlewareContext) -> Optional[str]:
    """A short one-way tag that groups the calls of one client session."""
    try:
        session_id = getattr(context.fastmcp_context, "session_id", None)
    except Exception:
        return None
    if not isinstance(session_id, str) or not session_id:
        return None
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:8]


def build_record(
    tool: Any,
    arguments: Optional[Dict[Any, Any]],
    ok: bool,
    error: Optional[BaseException],
    elapsed_ms: int,
    session: Optional[str] = None,
) -> Dict[str, Any]:
    """The audit record for one call. Pure, so tests can check it directly."""
    args = _normalized(arguments)
    record: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": tool if isinstance(tool, str) and _TOOL_RE.fullmatch(tool) else "?",
        "ok": bool(ok),
        "ms": int(elapsed_ms),
    }
    action = _action(args)
    if action is not None:
        record["action"] = action
    items = _items(args)
    if items is not None:
        record["items"] = items
    added = _labels(args.get("add_label_ids"))
    removed = _labels(args.get("remove_label_ids"))
    if added is not None:
        record["add"] = added
    if removed is not None:
        record["remove"] = removed
    names = _argument_names(args)
    if names:
        record["args"] = names
    flags = _flags(args)
    if flags:
        record["flags"] = flags
    ids, ids_more = _ids(args)
    if ids:
        record["ids"] = ids
    if ids_more:
        record["ids_more"] = ids_more
    if added is not None or removed is not None or "trashed" in flags:
        record["trash"] = bool(
            (added and "TRASH" in added) or flags.get("trashed") is True
        )
    if error is not None:
        record["err"] = type(error).__name__
    if session:
        record["sess"] = session
    return record


class AuditLog:
    """An append-only file of JSON lines, opened once and kept open."""

    def __init__(self, path: str) -> None:
        self.path = path
        try:
            self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        except OSError as exc:
            raise AuditLogConfigError(
                f"{AUDIT_LOG_ENV}: cannot open the audit log for append ({type(exc).__name__})"
            ) from exc

    def write(self, record: Dict[str, Any]) -> bool:
        line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        try:
            os.write(self._fd, line.encode("utf-8"))
            return True
        except OSError as exc:
            logger.warning("Audit log write failed (%s)", type(exc).__name__)
            return False

    def close(self) -> None:
        try:
            os.close(self._fd)
        except OSError:
            pass


def load_audit_log() -> Optional[AuditLog]:
    """The configured audit log, or None when WORKSPACE_MCP_AUDIT_LOG is unset."""
    raw = os.environ.get(AUDIT_LOG_ENV)
    if raw is None or not raw.strip():
        return None
    return AuditLog(raw.strip())


class AuditLogMiddleware(Middleware):
    """Write one line per tool call, whatever the outcome."""

    def __init__(self, audit_log: AuditLog) -> None:
        self._audit_log = audit_log

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext):
        started = time.monotonic()
        error: Optional[BaseException] = None
        try:
            return await call_next(context)
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                self._audit_log.write(
                    build_record(
                        getattr(context.message, "name", None),
                        getattr(context.message, "arguments", None),
                        ok=error is None,
                        error=error,
                        elapsed_ms=int((time.monotonic() - started) * 1000),
                        session=_session(context),
                    )
                )
            except Exception as exc:  # never let the audit trail break a tool call
                logger.warning("Audit log record failed (%s)", type(exc).__name__)
