"""Tests for WORKSPACE_MCP_AUDIT_LOG."""

import asyncio
import json
import os
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import Optional

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

import core.audit_log as audit_log_module
from core.audit_log import (
    AUDIT_LOG_ENV,
    AuditLog,
    AuditLogConfigError,
    AuditLogMiddleware,
    build_record,
    fingerprint,
    load_audit_log,
)
from core.disabled_actions import DisabledActionsMiddleware, parse_disabled_actions
from core.utils import StringList

SECRETS = [
    "from:boss@example.com budget",
    "18c0ffee0000beef",
    "Quarterly numbers",
    "Label_4242",
    "amy@example.com",
]

_STR = {"type": "string"}
_OPT_STR = {"anyOf": [{"type": "string"}, {"type": "null"}]}
_OPT_BOOL = {"anyOf": [{"type": "boolean"}, {"type": "null"}]}
_STR_LIST = {"type": "array", "items": {"type": "string"}}
_OPT_STR_LIST = {"anyOf": [_STR_LIST, {"type": "null"}]}

SCHEMAS = {
    "search_gmail_messages": {
        "query": _STR,
        "page_size": {"type": "integer"},
        "user_google_email": _STR,
    },
    "draft_gmail_message": {"subject": _STR, "to": _OPT_STR, "thread_id": _OPT_STR},
    "batch_modify_gmail_message_labels": {
        "message_ids": _STR_LIST,
        "add_label_ids": _OPT_STR_LIST,
        "remove_label_ids": _OPT_STR_LIST,
        "verify": {"type": "boolean"},
    },
    "modify_gmail_message_labels": {
        "message_id": _STR,
        "add_label_ids": _OPT_STR_LIST,
        "remove_label_ids": _OPT_STR_LIST,
    },
    "update_drive_file": {"file_id": _STR, "name": _OPT_STR, "trashed": _OPT_BOOL},
    "modify_doc_text": {
        "document_id": _STR,
        "text": _OPT_STR,
        "start_index": {"type": "integer"},
        "bold": _OPT_BOOL,
    },
    "manage_event": {
        "action": _STR,
        "event_id": _OPT_STR,
        "calendar_id": _STR,
        "summary": _OPT_STR,
    },
    "manage_contacts_batch": {
        "action": {"type": "string", "enum": ["create", "update", "delete"]},
        "updates": {"type": "array", "items": {"type": "object"}},
    },
    "list_calendars": {},
}


def _record(tool, arguments, ok=True, error=None):
    return build_record(tool, arguments, SCHEMAS.get(tool), ok, error, 3)


def _lines(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle.read().splitlines()]


async def _call(audit_log, tool, arguments, raises=None, session_id=None, lookup=None):
    middleware = AuditLogMiddleware(audit_log)

    async def call_next(context):
        if raises is not None:
            raise raises
        return "ran"

    async def get_tool(name):
        if lookup is not None:
            return lookup(name)
        schema = SCHEMAS.get(name)
        if schema is None:
            return None
        return SimpleNamespace(parameters={"properties": schema})

    context = SimpleNamespace(
        message=SimpleNamespace(name=tool, arguments=arguments),
        fastmcp_context=SimpleNamespace(
            session_id=session_id, fastmcp=SimpleNamespace(get_tool=get_tool)
        ),
    )
    return await middleware.on_call_tool(context, call_next)


# --- configuration and the file -------------------------------------------------


def test_unset_or_blank_means_no_audit_log(monkeypatch):
    monkeypatch.delenv(AUDIT_LOG_ENV, raising=False)
    assert load_audit_log() is None
    monkeypatch.setenv(AUDIT_LOG_ENV, "   ")
    assert load_audit_log() is None


def test_unusable_path_is_a_config_error(monkeypatch, tmp_path):
    monkeypatch.setenv(AUDIT_LOG_ENV, str(tmp_path / "missing-dir" / "audit.log"))
    with pytest.raises(AuditLogConfigError) as excinfo:
        load_audit_log()
    assert "missing-dir" not in str(excinfo.value)  # the message does not echo the path


def test_a_symlink_is_refused(tmp_path):
    target = tmp_path / "elsewhere.log"
    target.write_text("", encoding="utf-8")
    link = tmp_path / "audit.log"
    link.symlink_to(target)
    with pytest.raises(AuditLogConfigError):
        AuditLog(str(link))


def test_file_is_private_and_appended_to(monkeypatch, tmp_path):
    path = tmp_path / "audit.log"
    path.write_text('{"earlier":true}\n', encoding="utf-8")
    os.chmod(path, 0o644)
    monkeypatch.setenv(AUDIT_LOG_ENV, f"  {path}  ")
    audit_log = load_audit_log()
    assert audit_log.write({"tool": "x"}) is True
    audit_log.close()
    assert [sorted(line) for line in _lines(path)] == [["earlier"], ["tool"]]
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600  # an existing file is tightened

    fresh = tmp_path / "fresh.log"
    AuditLog(str(fresh)).close()
    assert stat.S_IMODE(os.stat(fresh).st_mode) == 0o600


def test_a_file_that_is_already_private_is_not_chmodded(monkeypatch, tmp_path):
    # root may mark the file append-only (chattr +a); every chmod is then refused,
    # even one that changes nothing, and the server must still start.
    path = tmp_path / "audit.log"
    path.write_text("", encoding="utf-8")
    os.chmod(path, 0o600)

    def refuse(*args, **kwargs):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(audit_log_module.os, "fchmod", refuse)
    audit_log = AuditLog(str(path))
    assert audit_log.write({"tool": "x"}) is True
    audit_log.close()

    loose = tmp_path / "loose.log"
    loose.write_text("", encoding="utf-8")
    os.chmod(loose, 0o644)
    with pytest.raises(AuditLogConfigError):  # a loose file that cannot be tightened
        AuditLog(str(loose))


# --- what a line holds, and what it never holds ---------------------------------


@pytest.mark.asyncio
async def test_one_line_per_call_and_no_argument_content(tmp_path):
    path = tmp_path / "audit.log"
    audit_log = AuditLog(str(path))
    assert (
        await _call(
            audit_log,
            "search_gmail_messages",
            {"query": SECRETS[0], "page_size": 25, "user_google_email": SECRETS[4]},
            session_id="session-abc",
        )
        == "ran"
    )
    await _call(
        audit_log,
        "draft_gmail_message",
        {"subject": SECRETS[2], "to": SECRETS[4], "thread_id": SECRETS[1]},
    )
    audit_log.close()

    first, second = _lines(path)
    assert first["tool"] == "search_gmail_messages"
    assert first["ok"] is True and "err" not in first and "items" not in first
    assert isinstance(first["ms"], int) and first["ts"].endswith("Z")
    assert first["sess"] != "session-abc" and len(first["sess"]) == 8
    assert first["args"] == ["page_size", "query", "user_google_email"]
    assert "ids" not in first and "flags" not in first and "extra" not in first
    assert second["tool"] == "draft_gmail_message" and second["items"] == 1
    assert second["args"] == ["subject", "thread_id", "to"]
    assert second["ids"] == [fingerprint(SECRETS[1])]
    assert "sess" not in second

    text = path.read_text(encoding="utf-8")
    for secret in SECRETS + ["session-abc"]:
        assert secret not in text


def test_trash_is_visible_and_user_labels_are_not_named():
    batch = _record(
        "batch_modify_gmail_message_labels",
        {
            "message_ids": [SECRETS[1], "18c0ffee0000beee", "18c0ffee0000beed"],
            "add_label_ids": ["trash", SECRETS[3]],
            "remove_label_ids": ["INBOX", "CATEGORY_PROMOTIONS", "category_secretname"],
        },
    )
    single = _record(
        "modify_gmail_message_labels",
        {"message_id": SECRETS[1], "remove_label_ids": ["INBOX"]},
    )
    assert batch["items"] == 3 and len(batch["ids"]) == 3
    assert batch["add"] == ["TRASH", "USER"]
    assert batch["remove"] == ["INBOX", "CATEGORY_PROMOTIONS", "USER"]
    assert batch["trash"] is True
    assert single["items"] == 1 and single["remove"] == ["INBOX"]
    assert single["trash"] is False and "add" not in single
    text = json.dumps([batch, single])
    assert SECRETS[1] not in text and SECRETS[3] not in text
    assert "secretname" not in text.casefold()


def test_a_trash_sent_in_the_loose_forms_the_tools_accept_is_still_seen():
    """Some clients send lists as JSON strings and booleans as words; the tools
    act on those, so the audit line must show the same thing."""
    as_strings = _record(
        "batch_modify_gmail_message_labels",
        {
            "message_ids": '["18c0ffee0000aaaa","18c0ffee0000bbbb"]',
            "add_label_ids": '["TRASH"]',
        },
    )
    assert as_strings["items"] == 2 and as_strings["add"] == ["TRASH"]
    assert as_strings["trash"] is True
    assert as_strings["ids"] == [
        fingerprint("18c0ffee0000aaaa"),
        fingerprint("18c0ffee0000bbbb"),
    ]
    # a padded list must not push TRASH out of the record (list or JSON string)
    padded = [
        SECRETS[3],
        "INBOX",
        "UNREAD",
        "STARRED",
        "IMPORTANT",
        "CATEGORY_PERSONAL",
        "CATEGORY_SOCIAL",
        "CATEGORY_PROMOTIONS",
        "CATEGORY_UPDATES",
        "CATEGORY_FORUMS",
        "SENT",
        "TRASH",
    ]
    for form in (padded, json.dumps(padded)):
        record = _record(
            "modify_gmail_message_labels",
            {"message_id": SECRETS[1], "add_label_ids": form},
        )
        assert record["trash"] is True and "TRASH" in record["add"]
        assert SECRETS[3] not in json.dumps(record)
    for loose in ("true", " True ", "yes", 1, 1.0):
        record = _record(
            "update_drive_file", {"file_id": "1AbCdEfDriveFile", "trashed": loose}
        )
        assert record["flags"] == {"trashed": True} and record["trash"] is True
    for loose in ("false", "0", 0, 0.0):
        record = _record(
            "update_drive_file", {"file_id": "1AbCdEfDriveFile", "trashed": loose}
        )
        assert record["flags"] == {"trashed": False} and record["trash"] is False
    odd = _record(
        "update_drive_file", {"file_id": "1AbCdEfDriveFile", "trashed": SECRETS[2]}
    )
    assert "flags" not in odd and SECRETS[2] not in json.dumps(odd)


def test_every_service_is_recorded_not_only_mail():
    drive = _record(
        "update_drive_file",
        {"file_id": "1AbCdEfDriveFile", "trashed": True, "name": None},
    )
    doc = _record(
        "modify_doc_text",
        {
            "document_id": "1AbCdEfDocument",
            "text": SECRETS[2],
            "start_index": 1,
            "bold": False,
        },
    )
    event = _record(
        "manage_event",
        {
            "action": "update",
            "event_id": "4vq8s54q711eu19b",
            "calendar_id": "primary",
            "summary": SECRETS[2],
        },
    )
    contacts = _record(
        "manage_contacts_batch",
        {
            "action": "update",
            "updates": [
                {"contact_id": "c4943545581014634986", "family_name": SECRETS[2]},
                {"contact_id": "c7478952745262746950"},
            ],
        },
    )
    assert drive["args"] == [
        "file_id",
        "trashed",
    ]  # an argument left at None is not listed
    assert drive["flags"] == {"trashed": True} and drive["trash"] is True
    assert drive["ids"] == [fingerprint("1AbCdEfDriveFile")]
    assert doc["args"] == ["bold", "document_id", "start_index", "text"]
    assert doc["flags"] == {"bold": False} and "trash" not in doc
    assert doc["ids"] == [fingerprint("1AbCdEfDocument")]
    assert event["action"] == "update"
    assert sorted(event["ids"]) == sorted([fingerprint("4vq8s54q711eu19b"), "primary"])
    assert contacts["items"] == 2
    assert contacts["ids"] == [
        fingerprint("c4943545581014634986"),
        fingerprint("c7478952745262746950"),
    ]
    text = json.dumps([drive, doc, event, contacts])
    for raw in [
        "1AbCdEfDriveFile",
        "1AbCdEfDocument",
        "c4943545581014634986",
        SECRETS[2],
    ]:
        assert raw not in text


def test_a_guessable_id_is_never_fingerprinted():
    record = _record(
        "manage_event",
        {"action": "create", "calendar_id": SECRETS[4], "event_id": "ev1"},
    )
    assert sorted(record["ids"]) == ["email", "short"]
    assert fingerprint(SECRETS[4]) not in json.dumps(record)
    assert fingerprint("ev1") not in json.dumps(record)
    assert _record("manage_event", {"action": "create", "calendar_id": "@default"})[
        "ids"
    ] == ["primary"]


def test_names_the_tool_does_not_declare_are_never_written():
    record = _record(
        "update_drive_file",
        {
            "file_id": "1AbCdEfDriveFile",
            "affair_with_bob_tuesday": True,
            "MyAffairWithBob": "x",
            5: "y",
        },
    )
    assert record["args"] == ["file_id"] and record["extra"] == 3
    assert "flags" not in record
    assert "affair" not in json.dumps(record).casefold()


@pytest.mark.asyncio
@pytest.mark.parametrize("lookup", ["missing", "raises"])
async def test_an_unresolved_tool_leaves_no_name_and_nothing_from_its_arguments(
    tmp_path, lookup
):
    def find(name):
        if lookup == "raises":
            raise LookupError(name)
        return None

    path = tmp_path / "audit.log"
    audit_log = AuditLog(str(path))
    assert (
        await _call(
            audit_log,
            "amy_password_is_hunter2",
            {"file_id": "1AbCdEfDriveFile", SECRETS[2]: True},
            lookup=find,
        )
        == "ran"
    )
    audit_log.close()
    (record,) = _lines(path)
    assert record["tool"] == "?" and record["extra"] == 2
    assert set(record) == {"ts", "tool", "ok", "ms", "extra"}
    text = path.read_text(encoding="utf-8")
    assert "hunter2" not in text and SECRETS[2] not in text


@pytest.mark.parametrize(
    "tool,value,expected",
    [
        ("manage_contacts_batch", "update", "update"),
        ("manage_contacts_batch", " DELETE ", "delete"),
        ("manage_contacts_batch", "purge", "?"),  # not one of the declared actions
        ("manage_event", "Clear_Completed", "clear_completed"),
        ("manage_event", "delete everything; " + SECRETS[2], "?"),
        ("manage_event", "this_is_far_too_long_to_be_an_action", "?"),
        # a plain-string action is checked against the server's own action words:
        # a word that merely looks like one is caller text
        ("manage_event", "secret_project_orchid", "?"),
        ("manage_event", "budget", "?"),
        ("manage_event", "move", "move"),
        ("manage_event", 7, "?"),
        ("manage_event", ["delete"], "?"),
    ],
)
def test_action_is_recorded_only_when_it_looks_like_one(tool, value, expected):
    record = _record(tool, {"action": value})
    assert record["action"] == expected
    assert SECRETS[2] not in json.dumps(record)
    assert "action" not in _record(tool, {})


def test_ids_are_fingerprints_and_a_huge_batch_is_capped():
    ids = [f"message-{n:05d}" for n in range(1005)]
    record = _record(
        "batch_modify_gmail_message_labels",
        {"message_ids": ids, "add_label_ids": ["TRASH"], "verify": False},
    )
    assert record["items"] == 1005 and len(record["ids"]) == 1000
    assert record["ids_more"] == 5 and record["ids"][0] == fingerprint("message-00000")
    assert record["flags"] == {"verify": False} and record["trash"] is True
    assert len(fingerprint(" some-id ")) == 12
    assert fingerprint(" some-id ") == fingerprint("some-id")
    assert "message-00000" not in json.dumps(record)


def test_hostile_arguments_do_not_break_the_record():
    assert build_record(None, None, None, True, None, 0)["tool"] == "?"
    assert build_record("x", "not a dict", {}, True, None, 0)["tool"] == "x"
    record = _record(
        "batch_modify_gmail_message_labels",
        {
            "message_ids": {"nested": [1, 2]},
            "add_label_ids": [None, 5, {"a": 1}],
            "remove_label_ids": "not json [",
            "verify": {"deep": True},
        },
    )
    assert record["add"] == ["USER"] and "remove" not in record
    assert "flags" not in record and "ids" not in record


# --- the middleware ------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_failed_call_is_recorded_and_still_raises(tmp_path):
    path = tmp_path / "audit.log"
    audit_log = AuditLog(str(path))
    with pytest.raises(RuntimeError, match=SECRETS[2]):
        await _call(
            audit_log,
            "manage_event",
            {"action": " Delete ", "event_id": "4vq8s54q711eu19b"},
            raises=RuntimeError(SECRETS[2]),
        )
    audit_log.close()

    (record,) = _lines(path)
    assert record["ok"] is False and record["err"] == "RuntimeError"
    assert record["action"] == "delete" and record["items"] == 1
    assert SECRETS[2] not in path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_cancelled_call_is_recorded_and_still_cancels(tmp_path):
    path = tmp_path / "audit.log"
    audit_log = AuditLog(str(path))
    with pytest.raises(asyncio.CancelledError):
        await _call(audit_log, "list_calendars", {}, raises=asyncio.CancelledError())
    audit_log.close()
    (record,) = _lines(path)
    assert record["ok"] is False and record["err"] == "CancelledError"


@pytest.mark.asyncio
async def test_a_failed_write_never_fails_the_tool(tmp_path, caplog, monkeypatch):
    path = tmp_path / "audit.log"
    audit_log = AuditLog(str(path))
    real_write = os.write
    sizes = []

    def short_then_whole(fd, data):
        # the first call takes one byte, as a nearly full disk would
        sizes.append(len(data))
        return real_write(fd, bytes(data[:1]) if len(sizes) == 1 else bytes(data))

    monkeypatch.setattr(audit_log_module.os, "write", short_then_whole)
    assert await _call(audit_log, "list_calendars", {}) == "ran"
    monkeypatch.undo()
    assert len(sizes) == 2  # the rest of the line followed
    assert [line["tool"] for line in _lines(path)] == ["list_calendars"]

    state = {"calls": 0}

    def prefix_then_fail(fd, data):
        # three bytes reach the file, then the disk is full
        state["calls"] += 1
        if state["calls"] == 1:
            return real_write(fd, bytes(data[:3]))
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(audit_log_module.os, "write", prefix_then_fail)
    with caplog.at_level("WARNING"):
        assert await _call(audit_log, "list_calendars", {}) == "ran"
    assert "Audit log write failed (OSError)" in caplog.text
    monkeypatch.undo()
    # the next record must not be glued to the torn one
    assert await _call(audit_log, "list_calendars", {}) == "ran"
    raw = path.read_text(encoding="utf-8").splitlines()
    assert len(raw) == 3 and len(raw[1]) == 3
    assert json.loads(raw[2])["tool"] == "list_calendars"

    audit_log.close()  # the next write hits a closed descriptor
    with caplog.at_level("WARNING"):
        assert await _call(audit_log, "list_calendars", {}) == "ran"
    assert "Audit log write failed" in caplog.text


@pytest.mark.asyncio
async def test_a_broken_record_never_fails_the_tool(tmp_path, caplog, monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError(SECRETS[2])

    path = tmp_path / "audit.log"
    audit_log = AuditLog(str(path))
    real_build = build_record

    def boom_once(tool, *args, **kwargs):
        if tool is not None:  # the fallback call names no tool
            boom()
        return real_build(tool, *args, **kwargs)

    monkeypatch.setattr(audit_log_module, "build_record", boom_once)
    with caplog.at_level("WARNING"):
        assert await _call(audit_log, "list_calendars", {"x": SECRETS[2]}) == "ran"
    assert "Audit log record failed (ValueError)" in caplog.text
    assert SECRETS[2] not in caplog.text
    monkeypatch.undo()
    audit_log.close()
    # the call still left its line, with nothing taken from the arguments
    (record,) = _lines(path)
    assert record["tool"] == "?" and record["ok"] is True
    assert record["audit_err"] == "ValueError"
    assert SECRETS[2] not in path.read_text(encoding="utf-8")


def test_an_id_with_a_lone_surrogate_does_not_cost_the_record():
    record = _record("modify_gmail_message_labels", {"message_id": "\ud800abcdefgh"})
    assert record["items"] == 1 and len(record["ids"][0]) == 12
    json.dumps(record)


@pytest.mark.asyncio
async def test_a_call_cancelled_during_the_tool_lookup_is_recorded(tmp_path):
    path = tmp_path / "audit.log"
    audit_log = AuditLog(str(path))

    def cancelled(name):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await _call(audit_log, "list_calendars", {}, lookup=cancelled)
    audit_log.close()
    (record,) = _lines(path)
    assert record["ok"] is False and record["err"] == "CancelledError"
    assert record["tool"] == "?"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a FIFO")
def test_a_path_that_is_not_a_regular_file_is_refused(tmp_path):
    fifo = tmp_path / "audit.log"
    os.mkfifo(fifo)
    with pytest.raises(AuditLogConfigError):  # and it must not hang
        AuditLog(str(fifo))
    with pytest.raises(AuditLogConfigError):
        AuditLog("/dev/null")


# --- through a real server ------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_with_real_argument_coercion(tmp_path):
    """Real FastMCP, real pydantic coercion: what the tool acts on is what the
    line shows, a refused action is recorded, and nothing a caller invents
    (argument name, tool name) reaches the file."""
    path = tmp_path / "audit.log"
    audit_log = AuditLog(str(path))
    server = FastMCP("audit-test")

    @server.tool
    def batch_modify(message_ids: StringList, add_label_ids: StringList) -> str:
        return f"added {add_label_ids} to {len(message_ids)}"

    @server.tool
    def update_file(file_id: str, trashed: Optional[bool] = None) -> str:
        return f"trashed={trashed}"

    @server.tool
    def manage_event(action: str, event_id: str = "") -> str:
        return f"did {action}"

    server.add_middleware(AuditLogMiddleware(audit_log))
    server.add_middleware(
        DisabledActionsMiddleware(parse_disabled_actions("manage_event:delete"))
    )

    async with Client(server) as client:
        result = await client.call_tool(
            "batch_modify",
            {
                "message_ids": '["18c0ffee0000aaaa","18c0ffee0000bbbb"]',
                "add_label_ids": '["TRASH"]',
            },
        )
        assert "added ['TRASH'] to 2" in str(result)
        result = await client.call_tool(
            "update_file", {"file_id": "1AbCdEfDriveFile", "trashed": "true"}
        )
        assert "trashed=True" in str(result)
        with pytest.raises(ToolError, match="disabled"):
            await client.call_tool(
                "manage_event", {"action": "delete", "event_id": SECRETS[1]}
            )
        with pytest.raises(ToolError):
            await client.call_tool(
                "update_file", {"file_id": "1AbCdEfDriveFile", "affair_with_bob": True}
            )
        with pytest.raises(Exception):
            await client.call_tool("amy_password_is_hunter2", {"x": SECRETS[2]})
    audit_log.close()

    lines = _lines(path)
    assert len(lines) == 5  # the call to a tool that does not exist leaves one too
    batch, update, refused, invented, unknown = lines
    assert unknown["tool"] == "?" and unknown["ok"] is False
    assert "args" not in unknown and "ids" not in unknown
    assert batch["tool"] == "batch_modify" and batch["items"] == 2
    assert batch["add"] == ["TRASH"] and batch["trash"] is True and batch["ok"] is True
    assert update["flags"] == {"trashed": True} and update["trash"] is True
    assert (refused["action"], refused["ok"], refused["err"]) == (
        "delete",
        False,
        "ToolError",
    )
    assert invented["ok"] is False and invented["args"] == ["file_id"]
    assert invented["extra"] == 1
    text = path.read_text(encoding="utf-8")
    for raw in ["affair", "hunter2", SECRETS[1], SECRETS[2], "1AbCdEfDriveFile"]:
        assert raw not in text
    assert all(
        line["tool"] in {"batch_modify", "update_file", "manage_event", "?"}
        for line in lines
    )


def test_the_real_server_registers_it_outside_the_disabled_actions_check(tmp_path):
    """core/server.py itself: with the variable set the audit middleware sits
    after the camelCase rename and before the disabled-actions refusal; with it
    unset it is absent."""
    code = (
        "import core.server as s;"
        "print(','.join(type(m).__name__ for m in s.server.middleware))"
    )
    env = {k: v for k, v in os.environ.items() if k != AUDIT_LOG_ENV}
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def order(extra):
        out = subprocess.run(
            [sys.executable, "-c", code],
            env={**env, **extra},
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert out.returncode == 0, out.stderr[-400:]
        return out.stdout.strip().splitlines()[-1].split(",")

    without = order({})
    assert "AuditLogMiddleware" not in without
    with_log = order({AUDIT_LOG_ENV: str(tmp_path / "audit.log")})
    assert (
        with_log.index("CamelCaseArgumentsMiddleware")
        < with_log.index("AuditLogMiddleware")
        < with_log.index("DisabledActionsMiddleware")
    )
