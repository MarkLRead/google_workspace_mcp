"""
Audit log of tool calls.

The normal logs of a household deployment run at WARNING, because the INFO lines
carry search text, addresses and ids. That leaves no record of WHICH tools ran:
after the fact nobody can say whether a session read mail or moved it to Trash.

This middleware writes one JSON line per tool call to a separate file, for every
tool of every service (Gmail, Drive, Docs, Sheets, Slides, Calendar, Tasks,
Contacts): when, which tool, which action, which declared arguments were given
(their names only), every yes/no argument with its value (``trashed``,
``verify``, ...), how many items, which Gmail system labels were added or
removed, and how the call ended.

It never writes argument content. No search text, subject, body, address or user
label name reaches the file. Only names the tool itself declares are written, so
a caller cannot smuggle text in as an argument name or a tool name. Ids are
written as fingerprints (the first 12 hex digits of their SHA-256), so "did any
call touch THIS document or message?" can be answered by someone who already
holds its id:

    python -c "from core.audit_log import fingerprint; print(fingerprint('<id>'))"

then search the audit log for the result. An id that is guessable (an email
address used as a calendar id, ``primary``, a very short value) is not
fingerprinted at all; the line says ``email``, ``primary`` or ``short`` instead.

The record is built from the arguments as the caller sent them, read the way the
tools read them: a list sent as a JSON string, or ``"true"`` for a yes/no
argument, is recorded as the tool will act on it.

Configured with ``WORKSPACE_MCP_AUDIT_LOG``, the path of the file, taken from the
process environment (a ``.env`` file is read too late to count). Unset or blank
means no audit log and nothing changes. A path that cannot be opened for append
stops the server at start-up: a deployment that asked for an audit log must not
run without one.

A write that fails later (disk full, descriptor gone) does not fail the tool
call; it is reported at WARNING in the normal log. A file that is deleted or
renamed while the server runs keeps receiving lines on the old descriptor, which
is what ``copytruncate`` rotation expects.

The file is created 0600 and an existing looser file is tightened. A file that is
already 0600 is opened as it is, so root may create it beforehand and mark it
append-only (``chattr +a``): the server can then add lines but can neither
rewrite nor remove what it has written.
"""

import hashlib
import json
import logging
import os
import stat
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext

from core.utils import _coerce_json_str_to_list

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
        "CATEGORY_PERSONAL",
        "CATEGORY_SOCIAL",
        "CATEGORY_PROMOTIONS",
        "CATEGORY_UPDATES",
        "CATEGORY_FORUMS",
    }
)
# The action words this server's tools act on. A tool that declares ``action`` as
# a plain string (no enum) is checked against this list instead: a word that is
# not here is caller text and is written as "?".
_KNOWN_ACTIONS = frozenset(
    {
        "clear_completed",
        "create",
        "delete",
        "grant",
        "grant_batch",
        "hide",
        "modify_members",
        "move",
        "populate_from_markdown",
        "rename",
        "reorder",
        "revoke",
        "transfer_owner",
        "unhide",
        "update",
    }
)
_TRUE = frozenset({"true", "t", "yes", "y", "on", "1"})
_FALSE = frozenset({"false", "f", "no", "n", "off", "0"})
_MAX_NAMES = 40
_MAX_IDS = 1000
_MIN_ID_LENGTH = 8


class AuditLogConfigError(ValueError):
    """WORKSPACE_MCP_AUDIT_LOG is set but the file cannot be used."""


def fingerprint(value: str) -> str:
    """One-way tag of an id: enough to recognise it, not to recover it."""
    # surrogatepass: JSON may carry a lone surrogate, which strict UTF-8 refuses;
    # one odd id must not cost the whole record.
    data = value.strip().encode("utf-8", "surrogatepass")
    return hashlib.sha256(data).hexdigest()[:12]


def _id_tag(value: str) -> str:
    """A fingerprint, unless the id could be guessed back from it."""
    text = value.strip()
    if text.casefold() in ("primary", "@default"):
        return "primary"
    if "@" in text:
        return "email"
    if len(text) < _MIN_ID_LENGTH:
        return "short"
    return fingerprint(text)


def _allows(schema: Any, json_type: str) -> bool:
    """Whether a parameter's JSON schema accepts ``json_type`` at its top level."""
    if not isinstance(schema, dict):
        return False
    declared = schema.get("type")
    if declared == json_type or (isinstance(declared, list) and json_type in declared):
        return True
    return any(
        _allows(option, json_type)
        for key in ("anyOf", "oneOf")
        for option in (schema.get(key) or [])
    )


def _as_bool(value: Any) -> Optional[bool]:
    """A yes/no argument the way pydantic will read it, or None if it will not."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):  # 1.0 and 0.0 too
        return bool(value)
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
    return None


def _as_list(value: Any) -> Optional[list]:
    """A list argument, accepting the JSON-string form the tools accept."""
    value = _coerce_json_str_to_list(value)
    return list(value) if isinstance(value, (list, tuple)) else None


def _action(value: Any, schema: Any) -> str:
    """The ``action`` argument, only when it is one the tool could mean."""
    text = value.strip().casefold() if isinstance(value, str) else ""
    allowed = schema.get("enum") if isinstance(schema, dict) else None
    if isinstance(allowed, list):
        known = {item.casefold() for item in allowed if isinstance(item, str)}
        return text if text in known else "?"
    return text if text in _KNOWN_ACTIONS else "?"


def _labels(value: Any) -> Optional[List[str]]:
    """System label names as given; every other label collapses to "USER"."""
    items = _as_list(value)
    if not items:
        return None
    seen: List[str] = []
    for item in items:
        name = item.strip().upper() if isinstance(item, str) else ""
        label = name if name in _SYSTEM_LABELS else "USER"
        if label not in seen:
            seen.append(label)
    # Never cut: the names come from a fixed list of 14 plus "USER", and a cut
    # would let a padded list push TRASH out of the record.
    return seen


def _ids(arguments: Dict[str, Any]) -> Tuple[List[str], int, int]:
    """Tags of the ids a call names, how many were left out, and the item count."""
    found: List[str] = []
    total = 0
    items: Optional[int] = None
    for key, value in arguments.items():
        if "label" in key or key == "action":
            continue
        listed = _as_list(value)
        if key.endswith("_ids") and listed is not None:
            candidates = [item for item in listed if isinstance(item, str)]
            items = len(listed) if items is None else items
        elif key.endswith("_id") and isinstance(value, str):
            candidates = [value]
        elif listed is not None and any(isinstance(entry, dict) for entry in listed):
            # batch tools take a list of dicts, each naming its own id
            candidates = [
                inner
                for entry in listed
                if isinstance(entry, dict)
                for name, inner in entry.items()
                if isinstance(name, str)
                and name.endswith("_id")
                and "label" not in name
                and isinstance(inner, str)
            ]
            items = len(listed) if items is None else items
        else:
            continue
        for item in candidates:
            if not item.strip():
                continue
            total += 1
            if len(found) < _MAX_IDS:
                found.append(_id_tag(item))
    if items is None and total:
        items = 1
    return found, total - len(found), items if items is not None else 0


def build_record(
    tool: Any,
    arguments: Optional[Dict[Any, Any]],
    declared: Optional[Dict[str, Any]],
    ok: bool,
    error: Optional[BaseException],
    elapsed_ms: int,
    session: Optional[str] = None,
) -> Dict[str, Any]:
    """The audit record for one call. Pure, so tests can check it directly.

    ``declared`` is the tool's own parameter schema (name -> JSON schema), or
    None when the tool could not be resolved. Nothing the caller named is
    written unless the tool declares it: without a schema the record holds
    neither the tool name nor anything taken from the arguments.
    """
    record: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": tool if declared is not None and isinstance(tool, str) else "?",
        "ok": bool(ok),
        "ms": int(elapsed_ms),
    }
    if error is not None:
        record["err"] = type(error).__name__
    if session:
        record["sess"] = session

    given = arguments if isinstance(arguments, dict) else {}
    if declared is None:
        if given:
            record["extra"] = len(given)
        return record

    args = {
        key: value
        for key, value in given.items()
        if isinstance(key, str) and key in declared and value is not None
    }
    extra = sum(1 for key in given if not (isinstance(key, str) and key in declared))
    if extra:
        record["extra"] = extra
    if args:
        record["args"] = sorted(args)[:_MAX_NAMES]
    if "action" in args:
        record["action"] = _action(args["action"], declared.get("action"))

    flags: Dict[str, bool] = {}
    for key in sorted(args):
        if _allows(declared.get(key), "boolean"):
            value = _as_bool(args[key])
            if value is not None and len(flags) < _MAX_NAMES:
                flags[key] = value
    if flags:
        record["flags"] = flags

    ids, ids_more, items = _ids(args)
    if items:
        record["items"] = items
    if ids:
        record["ids"] = ids
    if ids_more:
        record["ids_more"] = ids_more

    added = _labels(args.get("add_label_ids"))
    removed = _labels(args.get("remove_label_ids"))
    if added is not None:
        record["add"] = added
    if removed is not None:
        record["remove"] = removed
    if added is not None or removed is not None or "trashed" in flags:
        record["trash"] = bool(
            (added and "TRASH" in added) or flags.get("trashed") is True
        )
    return record


class AuditLog:
    """An append-only file of JSON lines, opened once and kept open."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._torn = False  # the last write left a line without its newline
        fd = -1
        try:
            # O_NONBLOCK: a FIFO at the path must fail here, not hang the start.
            fd = os.open(
                path,
                os.O_RDWR  # read: only to look at the last byte, below
                | os.O_CREAT
                | os.O_APPEND
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                0o600,
            )
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                # a device or a pipe would take the lines and keep none
                raise OSError("not a regular file")
            # Only when it is needed: a file root has marked append-only
            # (chattr +a) refuses every chmod, even one that changes nothing.
            if stat.S_IMODE(info.st_mode) != 0o600 and hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            if hasattr(os, "O_NONBLOCK"):
                os.set_blocking(fd, True)  # writes must block, never EAGAIN
            # A file left without its final newline (a write cut short before a
            # restart): the first record must start a line of its own. O_APPEND
            # puts every write at the end wherever this read leaves the offset.
            if info.st_size > 0:
                os.lseek(fd, -1, os.SEEK_END)
                self._torn = os.read(fd, 1) != b"\n"
        except OSError as exc:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise AuditLogConfigError(
                f"{AUDIT_LOG_ENV}: cannot open the audit log for append "
                f"({type(exc).__name__})"
            ) from exc
        self._fd = fd

    def write(self, record: Dict[str, Any]) -> bool:
        data = (
            json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")
        if self._torn:
            # An earlier write stopped part-way: end that broken line first, so
            # this record starts a line of its own.
            data = b"\n" + data
        view = memoryview(data)
        try:
            # os.write may take only part of the buffer: go on until it is all out.
            while view:
                written = os.write(self._fd, view)
                if written <= 0:
                    raise OSError("no progress")
                self._torn = True
                view = view[written:]
        except OSError as exc:
            logger.warning("Audit log write failed (%s)", type(exc).__name__)
            return False
        self._torn = False
        return True

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


def _session(context: MiddlewareContext) -> Optional[str]:
    """A short one-way tag that groups the calls of one client session."""
    try:
        session_id = getattr(context.fastmcp_context, "session_id", None)
    except Exception:
        return None
    if not isinstance(session_id, str) or not session_id:
        return None
    data = session_id.encode("utf-8", "surrogatepass")
    return hashlib.sha256(data).hexdigest()[:8]


async def _declared(context: MiddlewareContext) -> Optional[Dict[str, Any]]:
    """The called tool's parameter schema, or None when it cannot be resolved."""
    try:
        tool = await context.fastmcp_context.fastmcp.get_tool(context.message.name)
    except Exception:
        return None
    if tool is None:
        return None
    properties = (getattr(tool, "parameters", None) or {}).get("properties")
    return properties if isinstance(properties, dict) else {}


class AuditLogMiddleware(Middleware):
    """Write one line per tool call, whatever the outcome."""

    def __init__(self, audit_log: AuditLog) -> None:
        self._audit_log = audit_log

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext):
        started = time.monotonic()
        declared: Optional[Dict[str, Any]] = None
        error: Optional[BaseException] = None
        try:
            # Inside the try: a call cancelled while the tool is looked up must
            # leave its line too.
            declared = await _declared(context)
            return await call_next(context)
        except BaseException as exc:
            error = exc
            raise
        finally:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            try:
                record = build_record(
                    getattr(context.message, "name", None),
                    getattr(context.message, "arguments", None),
                    declared,
                    ok=error is None,
                    error=error,
                    elapsed_ms=elapsed_ms,
                    session=_session(context),
                )
            except Exception as exc:
                # The record could not be built from these arguments: the call
                # still leaves a line, with nothing taken from them.
                logger.warning("Audit log record failed (%s)", type(exc).__name__)
                record = build_record(
                    None, None, None, error is None, error, elapsed_ms
                )
                record["audit_err"] = type(exc).__name__
            try:
                self._audit_log.write(record)
            except Exception as exc:  # never let the audit trail break a tool call
                logger.warning("Audit log write failed (%s)", type(exc).__name__)
