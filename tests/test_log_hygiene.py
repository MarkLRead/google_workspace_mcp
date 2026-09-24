"""User content stays out of INFO logs.

Server logs routinely ship to aggregators whose access rules are broader than
the user's own data (operators, retention pipelines), so free-text the user
typed — search queries, find/replace text, subjects, titles — must not appear
at INFO. Operational metadata (who, which document, how much) is what INFO is
for; the full text may go to DEBUG, which production does not run at.

These tests pin the principle on the two historically worst offenders rather
than enumerating every tool: a regression elsewhere should be caught in review
by pattern-matching against these.
"""

import logging
import os
import subprocess
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gdocs.docs_tools import find_and_replace_doc, search_docs  # noqa: E402


def _unwrap(tool):
    fn = tool.fn if hasattr(tool, "fn") else tool
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


SECRET = "acquisition of ExampleCorp"


def _info_text(caplog) -> str:
    return " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.INFO)


@pytest.mark.asyncio
async def test_find_and_replace_logs_lengths_not_text(caplog):
    service = Mock()
    service.documents().batchUpdate().execute.return_value = {
        "replies": [{"replaceAllText": {"occurrencesChanged": 2}}]
    }

    with caplog.at_level(logging.DEBUG):
        await _unwrap(find_and_replace_doc)(
            service=service,
            user_google_email="user@example.com",
            document_id="doc-1",
            find_text=SECRET,
            replace_text=SECRET + " (final)",
        )

    assert SECRET not in _info_text(caplog)
    # The text is still available for debugging — at DEBUG, not silently gone.
    debug_text = " ".join(
        r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG
    )
    assert SECRET in debug_text


@pytest.mark.asyncio
async def test_search_docs_logs_query_length_not_query(caplog):
    service = Mock()
    service.files().list().execute.return_value = {"files": []}

    with caplog.at_level(logging.DEBUG):
        await _unwrap(search_docs)(
            service=service,
            user_google_email="user@example.com",
            query=SECRET,
        )

    info_text = _info_text(caplog)
    # Positive anchor first: prove capture is working at all, so the
    # exclusion below cannot pass vacuously against an empty log.
    assert "query_len=" in info_text
    assert SECRET not in info_text
    debug_text = " ".join(
        r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG
    )
    assert SECRET in debug_text


def test_scrub_url_queries_strips_embedded_request_uris():
    """HttpError messages embed the request URI; its query string carries the
    user's search terms, so the ERROR-level log must shed it while keeping
    the endpoint path that identifies what failed."""
    from core.utils import _scrub_url_queries

    msg = (
        "<HttpError 400 when requesting "
        f"https://gmail.googleapis.com/gmail/v1/users/me/messages?q={SECRET}"
        "&maxResults=25 returned bad request>"
    )
    scrubbed = _scrub_url_queries(msg)
    assert SECRET not in scrubbed
    assert "ExampleCorp" not in scrubbed
    assert "gmail/v1/users/me/messages" in scrubbed
    assert "<query-redacted>" in scrubbed


def test_scrub_url_queries_redacts_people_api_path_ids_and_survives_newlines():
    """2026-09-24 (mp-reviewer on the #303 leak fix): People API resource ids
    sit in the URL PATH, and a caller can put a person's name there
    (``get_contact("John Smith")`` → ``people/John%20Smith``); a query followed
    by a raw newline must not survive either. ``people/me`` stays readable.
    The path pattern needs the leading slash of an absolute URL (`/people/<id>`);
    a relative `people/<id>` on its own is not a request URL shape."""
    from core.utils import _scrub_url_queries

    token = "zzqtestperson"
    for msg in (
        f"<HttpError 404 when requesting https://people.googleapis.com/v1/people/{token}?personFields=names returned \"Not Found\">",
        f"https://people.googleapis.com/v1/people/{token}",
        f"https://people.googleapis.com/v1/otherContacts/{token}?readMask=names",
        f"https://people.googleapis.com/v1/contactGroups/{token}",
        f"https://people.googleapis.com/v1/people/John%20Smith?personFields=names",
    ):
        out = _scrub_url_queries(msg)
        assert token not in out and "John" not in out, (msg, out)
        assert "<id-redacted>" in out
        assert "people.googleapis.com/v1/" in out
    # The self-reference and the search endpoints keep their shape.
    assert _scrub_url_queries("https://people.googleapis.com/v1/people/me/connections?personFields=names") == (
        "https://people.googleapis.com/v1/people/me/connections?<query-redacted>"
    )
    assert _scrub_url_queries(f"https://people.googleapis.com/v1/people:searchContacts?query={SECRET}") == (
        "https://people.googleapis.com/v1/people:searchContacts?<query-redacted>"
    )
    # A query followed by a raw newline.
    out = _scrub_url_queries(f"https://x.googleapis.com/v1/a?q={SECRET}\nnext line")
    assert SECRET not in out and out.endswith("?<query-redacted>\nnext line")


def test_log_level_override_survives_import_time_basic_config(tmp_path):
    """Earlier imports configure logging before main reads the override."""
    env = os.environ.copy()
    env["WORKSPACE_MCP_LOG_LEVEL"] = "DEBUG"
    env["WORKSPACE_MCP_LOG_DIR"] = str(tmp_path)
    code = (
        "import logging, sys; import main; "
        "sys.stderr.write(str(logging.getLogger().getEffectiveLevel()))"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr.endswith(str(logging.DEBUG))


@pytest.mark.asyncio
async def test_search_gmail_messages_hides_query_from_info(caplog):
    """The flagship search tool: Gmail queries carry names and sensitive
    terms, so the value lives at DEBUG and INFO carries only its length."""
    from gmail.gmail_tools import search_gmail_messages

    service = Mock()
    service.users().messages().list().execute.return_value = {"messages": []}

    with caplog.at_level(logging.DEBUG):
        await _unwrap(search_gmail_messages)(
            service=service,
            user_google_email="user@example.com",
            query=SECRET,
        )

    info_text = _info_text(caplog)
    assert "query_len=" in info_text
    assert SECRET not in info_text
    assert SECRET in " ".join(
        r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG
    )


@pytest.mark.asyncio
async def test_search_drive_files_hides_query_from_info(caplog):
    """Drive logged the query in up to three lines (invoked + the two
    reformat branches); all value-bearing lines are DEBUG now."""
    from gdrive.drive_tools import search_drive_files

    service = Mock()
    service.files().list().execute.return_value = {"files": []}

    with caplog.at_level(logging.DEBUG):
        await _unwrap(search_drive_files)(
            service=service,
            user_google_email="user@example.com",
            query=SECRET,
        )

    info_text = _info_text(caplog)
    assert "query_len=" in info_text
    assert SECRET not in info_text


@pytest.mark.asyncio
async def test_chat_search_accepts_none_query():
    """Regression: search_messages allows query=None (time-filter-only), and
    the first draft of the hygiene sweep crashed on len(None) at the log
    line — before this call could do any work at all."""
    from gchat.chat_tools import search_messages

    chat_service = Mock()
    chat_service.spaces().list().execute.return_value = {"spaces": []}

    result = await _unwrap(search_messages)(
        chat_service=chat_service,
        people_service=Mock(),
        user_google_email="user@example.com",
        query=None,
        time_filter='createTime > "2026-01-01T00:00:00Z"',
    )
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_handle_http_errors_scrubs_request_uri_at_error(caplog):
    """Integration for the ERROR-path re-leak: HttpError embeds the request
    URI, so the decorator must log a scrubbed message at ERROR (no traceback,
    since the traceback's exception repr re-embeds the URI) and keep the full
    exc_info at DEBUG."""
    from googleapiclient.errors import HttpError

    from core.utils import handle_http_errors

    class _Resp:
        status = 400
        reason = "Bad Request"

    uri = f"https://gmail.googleapis.com/gmail/v1/users/me/messages?q={SECRET}"

    @handle_http_errors("dummy_tool")
    async def dummy():
        content = f'{{"error": {{"message": "Invalid query: {SECRET}"}}}}'.encode()
        raise HttpError(_Resp(), content, uri=uri)

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception):
            await dummy()

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error_records, "the decorator must still log the failure at ERROR"
    error_text = " ".join(r.getMessage() for r in error_records)
    assert SECRET not in error_text
    assert "ExampleCorp" not in error_text
    assert "<query-redacted>" in error_text
    assert all(not r.exc_info for r in error_records)
    assert any(r.exc_info for r in caplog.records if r.levelno == logging.DEBUG)


@pytest.mark.asyncio
async def test_handle_http_errors_raised_message_and_cause_are_scrubbed():
    """The RAISED exception travels upstream to re-loggers this module does not
    control (the auth middleware, FastMCP's "Error calling tool" traceback), so
    its text must already be clean: no request query string (a search query is
    a person's name), and the chained cause must be clean too (a traceback
    prints `__cause__` verbatim, and HttpError's own str embeds its `uri`).
    Google's reason and error body stay, and the HttpError object stays the
    cause: the client and the gchat edit tests need them.
    Found live on 2026-09-24: a synthetic token in a People API request reached
    PM2's error log through exactly these two paths."""
    from googleapiclient.errors import HttpError

    from core.utils import handle_http_errors

    class _Resp:
        status = 404
        reason = "Not Found"

    uri = f"https://people.googleapis.com/v1/people:searchContacts?query={SECRET}&readMask=names"

    # Astra a-20260924-g9e8: the body must survive UNCHANGED even when it holds
    # a "?" — scrubbing the whole text would have clipped it.
    body = "Invalid query? Requested entity was not found."

    @handle_http_errors("search_contacts", service_type="people")
    async def dummy():
        content = f'{{"error": {{"message": "{body}"}}}}'.encode()
        raise HttpError(_Resp(), content, uri=uri)

    with pytest.raises(Exception) as excinfo:
        await dummy()

    text = str(excinfo.value)
    assert SECRET not in text
    assert "ExampleCorp" not in text
    assert "<query-redacted>" in text
    assert body in text
    cause = excinfo.value.__cause__
    assert isinstance(cause, HttpError)
    assert SECRET not in str(cause) and "<query-redacted>" in str(cause)
    assert SECRET not in (cause.uri or "")


@pytest.mark.asyncio
async def test_auth_middleware_logs_scrubbed_error_without_traceback_at_error(
    caplog, monkeypatch
):
    """The middleware re-logs any tool error; a message carrying a request URL
    must be scrubbed at ERROR, with the traceback at DEBUG only."""
    from types import SimpleNamespace

    from auth.auth_info_middleware import AuthInfoMiddleware

    middleware = AuthInfoMiddleware()

    async def _noop(context):
        return None

    monkeypatch.setattr(middleware, "_process_request_for_auth", _noop)
    context = SimpleNamespace(fastmcp_context=None)
    url = f"https://people.googleapis.com/v1/otherContacts:search?query={SECRET}"

    async def call_next(ctx):
        raise RuntimeError(f"<HttpError 400 when requesting {url} returned 'Bad Request'>")

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RuntimeError):
            await middleware.on_call_tool(context, call_next)

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error_records
    error_text = " ".join(r.getMessage() for r in error_records)
    assert SECRET not in error_text
    assert "<query-redacted>" in error_text
    assert all(not r.exc_info for r in error_records)
    assert any(r.exc_info for r in caplog.records if r.levelno == logging.DEBUG)


@pytest.mark.asyncio
async def test_create_drive_file_hides_url_and_local_path(caplog, monkeypatch):
    """Caller-controlled URLs and local paths never enter log records."""
    import gdrive.drive_tools as drive_tools

    secret_path = f"/private/tmp/{SECRET}.txt"
    file_url = f"file://{secret_path}"
    path_obj = Mock()
    path_obj.exists.return_value = True
    path_obj.is_file.return_value = True
    path_obj.read_bytes.return_value = b"secret payload"
    monkeypatch.setattr(drive_tools, "validate_file_path", lambda _path: path_obj)

    service = Mock()
    service.files().get().execute.return_value = {
        "id": "root",
        "mimeType": "application/vnd.google-apps.folder",
    }
    service.files().create().execute.return_value = {
        "id": "F1",
        "name": "safe-output.txt",
        "webViewLink": "https://drive.google.com/safe-link",
    }

    with caplog.at_level(logging.DEBUG):
        await _unwrap(drive_tools.create_drive_file)(
            service=service,
            user_google_email="user@example.com",
            file_name="safe-output.txt",
            fileUrl=file_url,
        )

    log_text = " ".join(record.getMessage() for record in caplog.records)
    assert "has_fileUrl=True" in log_text
    assert SECRET not in log_text
    assert secret_path not in log_text
    assert file_url not in log_text


@pytest.mark.asyncio
async def test_create_drive_file_failure_hides_local_path_at_error(caplog, monkeypatch):
    """Path validation failures stay redacted through the error decorator."""
    import gdrive.drive_tools as drive_tools
    from core.utils import handle_http_errors

    secret_path = f"/private/tmp/{SECRET}.txt"
    file_url = f"file://{secret_path}"

    def fail_validation(_path):
        raise FileNotFoundError(f"Path does not exist: {secret_path}")

    monkeypatch.setattr(drive_tools, "validate_file_path", fail_validation)
    service = Mock()
    service.files().get().execute.return_value = {
        "id": "root",
        "mimeType": "application/vnd.google-apps.folder",
    }
    wrapped = handle_http_errors("create_drive_file")(
        _unwrap(drive_tools.create_drive_file)
    )

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception):
            await wrapped(
                service=service,
                user_google_email="user@example.com",
                file_name="safe-output.txt",
                fileUrl=file_url,
            )

    error_text = " ".join(
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.INFO
    )
    assert "Local file could not be accessed" in error_text
    assert SECRET not in error_text
    assert secret_path not in error_text
    assert file_url not in error_text


@pytest.mark.asyncio
async def test_import_to_google_doc_hides_file_name_from_info(caplog):
    """File names are user content too ("Termination letter — <name>.docx").
    Found live 2026-08-21: the import tools logged File Name: '<name>' at INFO.
    INFO carries file_name_len; the name itself is DEBUG."""
    from gdrive.drive_tools import import_to_google_doc

    secret_name = f"{SECRET}.md"
    service = Mock()
    # Folder resolution shortcut-checks the target; answer as a real folder.
    service.files().get().execute.return_value = {
        "id": "root",
        "mimeType": "application/vnd.google-apps.folder",
    }
    service.files().create().execute.return_value = {
        "id": "F1",
        "name": SECRET,
        "webViewLink": "https://docs.google.com/x",
        "mimeType": "application/vnd.google-apps.document",
    }

    with caplog.at_level(logging.DEBUG):
        await _unwrap(import_to_google_doc)(
            service=service,
            user_google_email="user@example.com",
            file_name=secret_name,
            content="# hi",
        )

    info_text = _info_text(caplog)
    assert "file_name_len=" in info_text
    assert SECRET not in info_text
    assert secret_name not in info_text


def test_gmail_attach_logs_length_not_filename(caplog):
    """Attachment success and failure logs contain only safe metadata."""
    import base64

    from gmail.gmail_tools import _prepare_gmail_message

    secret_name = f"{SECRET}.pdf"
    payload = base64.b64encode(b"%PDF-fake").decode()
    file_path = f"/definitely-missing/{secret_name}"

    with caplog.at_level(logging.DEBUG):
        _prepare_gmail_message(
            to="user@example.com",
            subject="s",
            body='<img src="cid:secret-inline">',
            body_format="html",
            attachments=[
                {"content": payload, "filename": secret_name},
                {
                    "content": payload,
                    "filename": secret_name,
                    "content_id": "secret-inline",
                    "mime_type": "image/png",
                },
                {"content": "a", "filename": secret_name},
                {"path": file_path, "filename": secret_name},
            ],
        )

    info_text = _info_text(caplog)
    assert "filename_len=" in info_text
    assert "content_id_len=" in info_text
    assert "path_len=" in info_text
    assert "error_type=" in info_text
    assert SECRET not in info_text
    assert secret_name not in info_text
    assert file_path not in info_text


# --- 2026-09-24 (R85): the D0-6 follow-ups — idempotent scrub + the raw-HttpError spots ---


class _HttpResp:
    def __init__(self, status: int, reason: str = "Not Found"):
        self.status = status
        self.reason = reason


def _http_error(uri: str, status: int = 404):
    from googleapiclient.errors import HttpError

    return HttpError(_HttpResp(status), b'{"error": {"message": "nope"}}', uri=uri)


def test_scrub_url_queries_is_idempotent():
    """The raised message's uri is scrubbed once in ``handle_http_errors`` and
    again by every re-logger upstream (the auth middleware scrubs ``str(e)``).
    Seen live 2026-09-24: the second pass left ``<id-redacted>>?<query-redacted>>``.
    A second pass must be a no-op, for both the query and the path-id patterns."""
    from core.utils import _scrub_url_queries

    for raw in (
        f"<HttpError 404 when requesting https://people.googleapis.com/v1/people/c123?personFields=names returned \"Not Found\">",
        f"https://people.googleapis.com/v1/otherContacts/c9?readMask=names",
        f"https://people.googleapis.com/v1/people:searchContacts?query={SECRET}",
        f"https://gmail.googleapis.com/gmail/v1/users/me/messages?q={SECRET}\nnext line",
        "https://people.googleapis.com/v1/people/me/connections?personFields=names",
    ):
        once = _scrub_url_queries(raw)
        twice = _scrub_url_queries(once)
        assert once == twice, (raw, once, twice)
        assert ">>" not in once
        assert SECRET not in once


def test_scrub_url_queries_redacts_path_id_with_apostrophe():
    """people.get expands ``{+resourceName}`` reserved, so an apostrophe in a
    name goes out unencoded; the id must be consumed whole (mp-reviewer,
    2026-09-24: ``people/<id-redacted>'Brien`` leaked half the name)."""
    from core.utils import _scrub_url_queries

    # Reserved expansion percent-encodes the space but not the apostrophe.
    raw = "<HttpError 404 when requesting https://people.googleapis.com/v1/people/John%20O'Brien?personFields=names returned \"Not Found\">"
    out = _scrub_url_queries(raw)
    assert "Brien" not in out and "John" not in out, out
    assert "/people/<id-redacted>?<query-redacted>" in out
    assert _scrub_url_queries(out) == out


def test_scrub_http_error_uri_cleans_object_and_text():
    from core.utils import scrub_http_error_uri

    err = _http_error(f"https://people.googleapis.com/v1/people/{'zzqtestperson'}?personFields=names")
    text = scrub_http_error_uri(err)
    assert "zzqtestperson" not in text
    assert "zzqtestperson" not in str(err)
    assert "zzqtestperson" not in (err.uri or "")
    assert "<id-redacted>" in text and "<query-redacted>" in text
    # Twice is harmless; a non-HttpError passes through as its str.
    assert scrub_http_error_uri(err) == text
    assert scrub_http_error_uri(ValueError("plain")) == "plain"


def test_gtasks_reauth_message_scrubs_request_uri():
    """``_format_reauth_message`` is both logged at ERROR (with exc_info) and
    raised; the raw HttpError text carried the request URL's query string."""
    from gtasks.tasks_tools import _format_reauth_message

    err = _http_error(
        f"https://tasks.googleapis.com/tasks/v1/lists/L1/tasks?dueMin={SECRET}&showCompleted=true",
        status=403,
    )
    message = _format_reauth_message(err, "user@example.com")
    assert SECRET not in message
    assert "<query-redacted>" in message
    assert "tasks/v1/lists/L1/tasks" in message
    assert "re-authenticate" in message
    # The object itself (the future chained cause / traceback text) is clean.
    assert SECRET not in str(err)


@pytest.mark.asyncio
async def test_contacts_warmup_warning_is_scrubbed(caplog):
    """The saved-contacts warm-up logged the raw HttpError at WARNING (the
    other-contacts warm-up already logged the scrubbed form)."""
    import gcontacts.contacts_tools as ct

    ct._search_cache_warmed_up.pop("u@example.com", None)
    service = Mock()
    service.people().searchContacts().execute.side_effect = _http_error(
        f"https://people.googleapis.com/v1/people:searchContacts?query=&readMask=names&x={SECRET}",
        status=429,
    )
    with caplog.at_level(logging.DEBUG):
        await ct._warmup_search_cache(service, "u@example.com")
    warn_text = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "warmup failed" in warn_text
    assert SECRET not in warn_text
    assert "status=429" in warn_text


@pytest.mark.asyncio
async def test_auth_middleware_prompt_path_logs_scrubbed_error_without_traceback_at_error(
    caplog, monkeypatch
):
    """``on_get_prompt`` re-logged the raw error with exc_info at ERROR; it now
    follows the same rule as ``on_call_tool``."""
    from types import SimpleNamespace

    from auth.auth_info_middleware import AuthInfoMiddleware

    middleware = AuthInfoMiddleware()

    async def _noop(context):
        return None

    monkeypatch.setattr(middleware, "_process_request_for_auth", _noop)
    context = SimpleNamespace(fastmcp_context=None)
    url = f"https://people.googleapis.com/v1/people:searchContacts?query={SECRET}"

    async def call_next(ctx):
        raise RuntimeError(f"<HttpError 400 when requesting {url} returned 'Bad Request'>")

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RuntimeError):
            await middleware.on_get_prompt(context, call_next)

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error_records
    error_text = " ".join(r.getMessage() for r in error_records)
    assert SECRET not in error_text
    assert "<query-redacted>" in error_text
    assert all(not r.exc_info for r in error_records)
    assert any(r.exc_info for r in caplog.records if r.levelno == logging.DEBUG)


def test_no_raw_http_error_interpolation_in_calendar_precheck_or_tasks():
    """Source scan: the four spots mp-reviewer listed on 2026-09-24 (plus the
    identical delete_event block) must not format a raw HttpError variable."""
    import re

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cal = open(os.path.join(root, "gcalendar", "calendar_tools.py"), encoding="utf-8").read()
    assert not re.search(r"verification[^\n]*\{get_error\}", cal), "calendar pre-check logs a raw HttpError"
    tasks = open(os.path.join(root, "gtasks", "tasks_tools.py"), encoding="utf-8").read()
    assert 'f"API error: {error}"' not in tasks
    contacts = open(os.path.join(root, "gcontacts", "contacts_tools.py"), encoding="utf-8").read()
    assert not re.search(r"warmup failed: \{e\}", contacts)
    mw = open(os.path.join(root, "auth", "auth_info_middleware.py"), encoding="utf-8").read()
    assert not re.search(r"middleware: \{e\}\", exc_info=True", mw)


# --- Astra a-20260924-3wee (NO-GO) — the certain findings, each pinned ---


def test_scrub_url_queries_me_boundary_and_placeholder_prefixes():
    """(1) `people/me'Jane` went out unchanged: the `me` exclusion treated the
    apostrophe as a boundary while the id class did not. (3) A placeholder is
    skipped only as a complete component: `?<query-redacted>Alice` and
    `/people/<id-redacted>Alice` are not scrubbed text."""
    from core.utils import _scrub_url_queries

    for raw, must_not in (
        ("https://people.googleapis.com/v1/people/me'Jane?personFields=names", "Jane"),
        ("https://people.googleapis.com/v1/people/me'O'Brien", "Brien"),
        ("https://people.googleapis.com/v1/people/<id-redacted>Alice?x=1", "Alice"),
        ("https://x.googleapis.com/v1/a?<query-redacted>Alice", "Alice"),
        ("https://x.googleapis.com/v1/a?<query-redacted>&q=Alice", "Alice"),
    ):
        out = _scrub_url_queries(raw)
        assert must_not not in out, (raw, out)
        assert _scrub_url_queries(out) == out, (raw, out)
    # The genuine self-references keep their shape.
    assert _scrub_url_queries("https://people.googleapis.com/v1/people/me") == (
        "https://people.googleapis.com/v1/people/me"
    )
    assert _scrub_url_queries("https://people.googleapis.com/v1/people/me/connections") == (
        "https://people.googleapis.com/v1/people/me/connections"
    )


def test_scrub_url_queries_redacts_calendar_ids_in_path():
    """A calendar id is usually an e-mail address (`%40`-encoded); `primary`
    is the fixed self-reference (Astra a-20260924-3wee, judgment item)."""
    from core.utils import _scrub_url_queries

    raw = (
        "<HttpError 404 when requesting https://www.googleapis.com/calendar/v3/calendars/"
        "jane.doe%40example.com/events/abc123?alt=json returned \"Not Found\">"
    )
    out = _scrub_url_queries(raw)
    assert "jane" not in out and "example.com" not in out, out
    assert "/calendars/<id-redacted>/events/abc123?<query-redacted>" in out
    assert _scrub_url_queries(out) == out
    assert _scrub_url_queries("https://www.googleapis.com/calendar/v3/calendars/primary/events") == (
        "https://www.googleapis.com/calendar/v3/calendars/primary/events"
    )
    assert "<id-redacted>" in _scrub_url_queries(
        "https://www.googleapis.com/calendar/v3/users/me/calendarList/jane%40example.com"
    )


def test_contacts_resource_name_refuses_slash_without_echo():
    """(2) `get_contact("John/Smith")` used to reach googleapiclient's
    `^people/[^/]+$` check, whose TypeError echoed the value into the
    catch-all ERROR log. The validator refuses it first, naming nothing."""
    from core.utils import UserInputError
    from gcontacts.contacts_tools import _resource_name

    assert _resource_name("c123", "people", "contact") == "people/c123"
    assert _resource_name("people/c123", "people", "contact") == "people/c123"
    assert _resource_name("me", "people", "contact") == "people/me"
    assert _resource_name("contactGroups/g1", "contactGroups", "contact group") == "contactGroups/g1"
    for bad in ("John/Smith", "people/John/Smith", "people/", "", "John Smith", None):
        with pytest.raises(UserInputError) as excinfo:
            _resource_name(bad, "people", "contact")
        assert "John" not in str(excinfo.value) and "Smith" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_contact_with_slash_id_logs_nothing_of_the_value(caplog, monkeypatch):
    """End to end through the decorators: the rejected value never reaches a
    log record at any level, and no TypeError from googleapiclient is raised."""
    import gcontacts.contacts_tools as ct

    fn = _unwrap(ct.get_contact)
    service = Mock()
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception) as excinfo:
            await fn(service=service, user_google_email="u@example.com", contact_id="John/Smith")
    assert "TypeError" not in repr(excinfo.value)
    everything = " ".join(r.getMessage() for r in caplog.records)
    assert "John" not in everything and "Smith" not in everything
    service.people().get.assert_not_called()


# --- Astra a-20260924-8ao3 (NO-GO): `?` through reserved expansion ---


def test_contacts_resource_name_refuses_query_and_fragment_chars():
    from core.utils import UserInputError
    from gcontacts.contacts_tools import _resource_name

    for bad in ("abc?x=Alice'zzqBob", "abc#frag", "people/abc?x=1"):
        with pytest.raises(UserInputError) as excinfo:
            _resource_name(bad, "people", "contact")
        assert "zzqBob" not in str(excinfo.value) and "abc" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_contact_with_query_in_id_logs_nothing_of_the_value(caplog):
    import gcontacts.contacts_tools as ct

    fn = _unwrap(ct.get_contact)
    service = Mock()
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception):
            await fn(service=service, user_google_email="u@example.com", contact_id="abc?x=Alice'zzqBob")
    everything = " ".join(r.getMessage() for r in caplog.records)
    assert "zzqBob" not in everything and "Alice" not in everything
    service.people().get.assert_not_called()


def test_scrub_http_error_uri_takes_whole_query_of_a_bare_uri():
    """Defence in depth: a bare `HttpError.uri` has no wrapper terminators, so
    a raw `'` or `>` inside its query (reachable through reserved expansion)
    must not end the scrub early."""
    from core.utils import _scrub_bare_uri, scrub_http_error_uri

    uri = "https://people.googleapis.com/v1/people/abc?x=Alice'zzqBob?personFields=names"
    err = _http_error(uri)
    text = scrub_http_error_uri(err)
    for leak in ("zzqBob", "Alice", "personFields"):
        assert leak not in text and leak not in (err.uri or ""), text
    assert err.uri == "https://people.googleapis.com/v1/people/<id-redacted>?<query-redacted>"
    assert _scrub_bare_uri(err.uri) == err.uri
    assert _scrub_bare_uri("https://x.googleapis.com/v1/a?q=1>b\"c\nd") == (
        "https://x.googleapis.com/v1/a?<query-redacted>"
    )
    assert _scrub_bare_uri("https://x.googleapis.com/v1/a") == "https://x.googleapis.com/v1/a"
