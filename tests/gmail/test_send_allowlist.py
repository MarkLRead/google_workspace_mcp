"""
Fork (market-pulse ticket #60): WORKSPACE_MCP_SEND_ALLOWLIST limits send_gmail_message
to a fixed list of recipients. Every To, Cc and Bcc must be on it or nothing is sent.
"""

import os
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.utils import UserInputError
from gmail.gmail_tools import (
    SEND_ALLOWLIST_ENV,
    _forward_gmail_message_impl,
    send_gmail_message,
)

ALLOWED = (
    "aevans@lakeviewvillage.org,mgreen@lakeviewvillage.org,kayboxread@gmail.com,"
    "mark@markread.org,amy.parker.read@gmail.com"
)


def _unwrap(tool):
    fn = tool.fn if hasattr(tool, "fn") else tool
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _service():
    service = Mock()
    service.users().messages().send().execute.return_value = {"id": "sent-1"}
    service.users().messages().send.reset_mock()
    return service


async def _send(service, **kwargs):
    args = dict(
        service=service,
        user_google_email="kayboxread@gmail.com",
        subject="Conflicts next week",
        body="Hello",
        include_signature=False,
    )
    args.update(kwargs)
    return await _unwrap(send_gmail_message)(**args)


@pytest.fixture
def allowlist(monkeypatch):
    monkeypatch.setenv(SEND_ALLOWLIST_ENV, ALLOWED)


@pytest.mark.asyncio
async def test_allowed_send_succeeds(allowlist):
    service = _service()
    result = await _send(
        service, to="aevans@lakeviewvillage.org", cc="mgreen@lakeviewvillage.org"
    )
    assert "sent-1" in result
    service.users().messages().send.assert_called_once()


@pytest.mark.asyncio
async def test_unlisted_cc_refuses_whole_send(allowlist):
    service = _service()
    with pytest.raises(UserInputError, match="stranger@example.com"):
        await _send(service, to="aevans@lakeviewvillage.org", cc="stranger@example.com")
    service.users().messages().send.assert_not_called()


@pytest.mark.asyncio
async def test_unlisted_bcc_refuses_whole_send(allowlist):
    service = _service()
    with pytest.raises(UserInputError, match="hidden@example.com"):
        await _send(service, to="mark@markread.org", bcc="hidden@example.com")
    service.users().messages().send.assert_not_called()


@pytest.mark.asyncio
async def test_one_bad_address_in_a_list_refuses(allowlist):
    service = _service()
    with pytest.raises(UserInputError, match="other@example.com"):
        await _send(service, to="mark@markread.org, other@example.com")
    service.users().messages().send.assert_not_called()


@pytest.mark.asyncio
async def test_mixed_case_address_accepted(allowlist):
    service = _service()
    await _send(service, to="AEvans@LakeviewVillage.ORG")
    service.users().messages().send.assert_called_once()


@pytest.mark.asyncio
async def test_display_name_form_accepted(allowlist):
    service = _service()
    await _send(service, to="Angela Evans <aevans@lakeviewvillage.org>")
    service.users().messages().send.assert_called_once()


@pytest.mark.asyncio
async def test_display_name_cannot_smuggle_an_address(allowlist):
    service = _service()
    with pytest.raises(UserInputError, match="evil@example.com"):
        await _send(service, to='"aevans@lakeviewvillage.org" <evil@example.com>')
    service.users().messages().send.assert_not_called()


@pytest.mark.asyncio
async def test_empty_allowlist_refuses_everything(monkeypatch):
    monkeypatch.setenv(SEND_ALLOWLIST_ENV, "")
    service = _service()
    with pytest.raises(UserInputError):
        await _send(service, to="mark@markread.org")
    service.users().messages().send.assert_not_called()


@pytest.mark.asyncio
async def test_unset_allowlist_keeps_upstream_behaviour(monkeypatch):
    monkeypatch.delenv(SEND_ALLOWLIST_ENV, raising=False)
    service = _service()
    await _send(service, to="anyone@example.com")
    service.users().messages().send.assert_called_once()


@pytest.mark.asyncio
async def test_forward_to_unlisted_address_refused(allowlist):
    service = Mock()
    service.users().messages().get().execute.return_value = {
        "id": "orig",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "Subject", "value": "Hi"},
                {"name": "From", "value": "a@example.com"},
                {"name": "To", "value": "kayboxread@gmail.com"},
                {"name": "Date", "value": "Mon, 1 Jan 2024 10:00:00 -0000"},
            ],
            "body": {"data": "SGVsbG8="},
        },
    }
    service.users().messages().send.reset_mock()
    with pytest.raises(UserInputError, match="outside@example.com"):
        await _forward_gmail_message_impl(
            service=service,
            message_id="orig",
            to="outside@example.com",
            user_google_email="kayboxread@gmail.com",
        )
    service.users().messages().send.assert_not_called()
