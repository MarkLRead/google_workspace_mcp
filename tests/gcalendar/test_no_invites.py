"""
Fork (market-pulse ticket #60): WORKSPACE_MCP_CALENDAR_NO_INVITES=1 forces
sendUpdates="none" on every calendar write, so Google emails no guest.
"""

import os
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from gcalendar.calendar_tools import (
    NO_INVITES_ENV,
    _create_event_impl,
    _effective_send_updates,
    _rsvp_event_impl,
)


def _service(event=None):
    service = Mock()
    service.events().insert().execute = Mock(
        return_value={"id": "evt1", "htmlLink": "https://x", "summary": "S"}
    )
    service.events().get().execute = Mock(return_value=event or {})
    service.events().patch().execute = Mock(return_value=event or {})
    return service


def test_switch_off_keeps_caller_value(monkeypatch):
    monkeypatch.delenv(NO_INVITES_ENV, raising=False)
    assert _effective_send_updates("all") == "all"
    assert _effective_send_updates("externalOnly") == "externalOnly"


@pytest.mark.parametrize("value", ["all", "externalOnly", "none"])
def test_switch_on_forces_none(monkeypatch, value):
    monkeypatch.setenv(NO_INVITES_ENV, "1")
    assert _effective_send_updates(value) == "none"


@pytest.mark.asyncio
async def test_create_with_guest_sends_no_invite(monkeypatch):
    monkeypatch.setenv(NO_INVITES_ENV, "1")
    service = _service()
    await _create_event_impl(
        service=service,
        user_google_email="kayboxread@gmail.com",
        summary="Therapy",
        start_time="2026-10-14T09:00:00Z",
        end_time="2026-10-14T10:00:00Z",
        attendees=["stranger@example.com"],
        send_updates="all",
    )
    assert service.events().insert.call_args.kwargs["sendUpdates"] == "none"


@pytest.mark.asyncio
async def test_rsvp_sends_no_update(monkeypatch):
    monkeypatch.setenv(NO_INVITES_ENV, "1")
    event = {
        "id": "evt123",
        "attendees": [{"email": "kayboxread@gmail.com", "self": True}],
        "organizer": {"self": False, "email": "org@example.com"},
    }
    service = _service(event)
    await _rsvp_event_impl(
        service=service,
        user_google_email="kayboxread@gmail.com",
        event_id="evt123",
        response="accepted",
    )
    assert service.events().patch.call_args.kwargs["sendUpdates"] == "none"
