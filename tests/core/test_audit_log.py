"""Tests for WORKSPACE_MCP_AUDIT_LOG."""

import json
import os
import stat
from types import SimpleNamespace

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

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

SECRETS = [
    "from:boss@example.com budget",
    "18c0ffee0000beef",
    "Quarterly numbers",
    "Label_4242",
    "amy@example.com",
]


def _lines(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle.read().splitlines()]


async def _call(audit_log, tool, arguments, result="ran", raises=None, session_id=None):
    middleware = AuditLogMiddleware(audit_log)

    async def call_next(context):
        if raises is not None:
            raise raises
        return result

    context = SimpleNamespace(
        message=SimpleNamespace(name=tool, arguments=arguments),
        fastmcp_context=SimpleNamespace(session_id=session_id),
    )
    return await middleware.on_call_tool(context, call_next)


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


def test_file_is_created_private_and_appended_to(monkeypatch, tmp_path):
    path = tmp_path / "audit.log"
    path.write_text('{"earlier":true}\n', encoding="utf-8")
    monkeypatch.setenv(AUDIT_LOG_ENV, f"  {path}  ")
    audit_log = load_audit_log()
    assert audit_log.write({"tool": "x"}) is True
    audit_log.close()
    assert [sorted(line) for line in _lines(path)] == [["earlier"], ["tool"]]

    fresh = tmp_path / "fresh.log"
    AuditLog(str(fresh)).close()
    assert stat.S_IMODE(os.stat(fresh).st_mode) == 0o600


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
    assert "ids" not in first and "flags" not in first
    assert second["tool"] == "draft_gmail_message" and second["items"] == 1
    assert second["args"] == ["subject", "thread_id", "to"]
    assert second["ids"] == [fingerprint(SECRETS[1])]
    assert "sess" not in second

    text = path.read_text(encoding="utf-8")
    for secret in SECRETS + ["session-abc"]:
        assert secret not in text


@pytest.mark.asyncio
async def test_trash_is_visible_and_user_labels_are_not_named(tmp_path):
    path = tmp_path / "audit.log"
    audit_log = AuditLog(str(path))
    await _call(
        audit_log,
        "batch_modify_gmail_message_labels",
        {
            "message_ids": [SECRETS[1], "18c0ffee0000beee", "18c0ffee0000beed"],
            "add_label_ids": ["trash", SECRETS[3]],
            "remove_label_ids": ["INBOX", "CATEGORY_PROMOTIONS", "CATEGORY_x y"],
        },
    )
    await _call(
        audit_log,
        "modify_gmail_message_labels",
        {"message_id": SECRETS[1], "remove_label_ids": ["INBOX"]},
    )
    audit_log.close()

    batch, single = _lines(path)
    assert batch["items"] == 3
    assert batch["add"] == ["TRASH", "USER"]
    assert batch["remove"] == ["INBOX", "CATEGORY_PROMOTIONS", "USER"]
    assert batch["trash"] is True
    assert single["items"] == 1 and single["remove"] == ["INBOX"]
    assert single["trash"] is False and "add" not in single
    text = path.read_text(encoding="utf-8")
    assert SECRETS[1] not in text and SECRETS[3] not in text and "x y" not in text


@pytest.mark.asyncio
async def test_every_service_is_recorded_not_only_mail(tmp_path):
    path = tmp_path / "audit.log"
    audit_log = AuditLog(str(path))
    await _call(
        audit_log,
        "update_drive_file",
        {"file_id": "1AbCdEfDriveFile", "trashed": True, "name": None},
    )
    await _call(
        audit_log,
        "modify_doc_text",
        {
            "document_id": "1AbCdEfDocument",
            "text": SECRETS[2],
            "start_index": 1,
            "bold": False,
        },
    )
    await _call(
        audit_log,
        "manage_event",
        {"action": "update", "event_id": "ev1", "summary": SECRETS[2]},
    )
    await _call(
        audit_log,
        "manage_contacts_batch",
        {
            "action": "update",
            "updates": [
                {"contact_id": "c1", "family_name": SECRETS[2]},
                {"contact_id": "c2"},
            ],
        },
    )
    audit_log.close()

    drive, doc, event, contacts = _lines(path)
    assert drive["args"] == [
        "file_id",
        "trashed",
    ]  # an argument left at None is not listed
    assert drive["flags"] == {"trashed": True} and drive["trash"] is True
    assert drive["ids"] == [fingerprint("1AbCdEfDriveFile")]
    assert doc["args"] == ["bold", "document_id", "start_index", "text"]
    assert doc["flags"] == {"bold": False} and "trash" not in doc
    assert doc["ids"] == [fingerprint("1AbCdEfDocument")]
    assert event["action"] == "update" and event["ids"] == [fingerprint("ev1")]
    assert contacts["items"] == 2
    assert contacts["ids"] == [fingerprint("c1"), fingerprint("c2")]
    text = path.read_text(encoding="utf-8")
    for raw in ["1AbCdEfDriveFile", "1AbCdEfDocument", SECRETS[2]]:
        assert raw not in text


def test_ids_are_fingerprints_and_a_huge_batch_is_capped():
    ids = [f"msg{n:05d}" for n in range(1005)]
    record = build_record(
        "batch_modify_gmail_message_labels",
        {"message_ids": ids, "add_label_ids": ["TRASH"], "verify": False},
        True,
        None,
        9,
    )
    assert record["items"] == 1005 and len(record["ids"]) == 1000
    assert record["ids_more"] == 5 and record["ids"][0] == fingerprint("msg00000")
    assert record["flags"] == {"verify": False} and record["trash"] is True
    assert len(fingerprint(" some-id ")) == 12 and fingerprint(
        " some-id "
    ) == fingerprint("some-id")
    assert "msg00000" not in json.dumps(record)


@pytest.mark.asyncio
async def test_a_failed_call_is_recorded_and_still_raises(tmp_path):
    path = tmp_path / "audit.log"
    audit_log = AuditLog(str(path))
    with pytest.raises(RuntimeError, match=SECRETS[2]):
        await _call(
            audit_log,
            "manage_event",
            {"Action": " Delete ", "event_id": "abc"},
            raises=RuntimeError(SECRETS[2]),
        )
    audit_log.close()

    (record,) = _lines(path)
    assert record["ok"] is False and record["err"] == "RuntimeError"
    assert record["action"] == "delete" and record["items"] == 1
    assert SECRETS[2] not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ({"action": "create"}, "create"),
        ({"action": "Clear_Completed"}, "clear_completed"),
        ({"action": "delete everything; " + SECRETS[2]}, "?"),
        ({"action": 7}, "?"),
        ({"action": ["delete"]}, "?"),
        ({}, None),
    ],
)
def test_action_is_recorded_only_when_it_looks_like_one(arguments, expected):
    record = build_record("manage_task", arguments, True, None, 3)
    assert record.get("action") == expected
    assert SECRETS[2] not in json.dumps(record)


def test_odd_tool_names_and_arguments_do_not_break_the_record():
    assert build_record(None, None, True, None, 0)["tool"] == "?"
    assert build_record("bad name\n" + SECRETS[2], {}, True, None, 0)["tool"] == "?"
    record = build_record(
        "manage_contacts_batch",
        {
            "updates": [{"contact_id": "c1"}, {"contact_id": "c2"}],
            5: "x",
            "label_id": "L",
        },
        True,
        None,
        1,
    )
    assert record["items"] == 2


@pytest.mark.asyncio
async def test_a_failed_write_never_fails_the_tool(tmp_path, caplog):
    audit_log = AuditLog(str(tmp_path / "audit.log"))
    audit_log.close()  # the next write hits a closed descriptor
    with caplog.at_level("WARNING"):
        assert await _call(audit_log, "list_calendars", {}) == "ran"
    assert "Audit log write failed" in caplog.text


@pytest.mark.asyncio
async def test_end_to_end_a_refused_action_is_recorded(tmp_path):
    """Through a real server: registered before the disabled-actions middleware,
    the audit log sees the calls that middleware refuses as well."""
    path = tmp_path / "audit.log"
    audit_log = AuditLog(str(path))
    server = FastMCP("audit-test")

    @server.tool
    def manage_event(action: str, event_id: str = "") -> str:
        return f"did {action}"

    server.add_middleware(AuditLogMiddleware(audit_log))
    server.add_middleware(
        DisabledActionsMiddleware(parse_disabled_actions("manage_event:delete"))
    )

    async with Client(server) as client:
        result = await client.call_tool(
            "manage_event", {"action": "create", "event_id": SECRETS[1]}
        )
        assert "did create" in str(result)
        with pytest.raises(ToolError, match="disabled"):
            await client.call_tool(
                "manage_event", {"action": "delete", "event_id": SECRETS[1]}
            )
    audit_log.close()

    created, refused = _lines(path)
    assert (created["tool"], created["action"], created["ok"]) == (
        "manage_event",
        "create",
        True,
    )
    assert (refused["action"], refused["ok"], refused["err"]) == (
        "delete",
        False,
        "ToolError",
    )
    assert created["items"] == 1 and SECRETS[1] not in path.read_text(encoding="utf-8")
