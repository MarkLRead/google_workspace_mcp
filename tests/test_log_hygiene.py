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
