"""Start-up behaviour of the email allowlist in OAuth 2.1 HTTP mode."""

from types import SimpleNamespace

import pytest

import core.server as server_module
from auth.email_allowlist import ALLOWED_EMAILS_ENV, AllowlistConfigError


def _configure(monkeypatch):
    captured = {}

    class FakeProvider:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.client_registration_options = None
            self._default_scope_str = ""

    monkeypatch.setattr(server_module, "get_transport_mode", lambda: "streamable-http")
    monkeypatch.setattr(server_module, "AllowlistGoogleProvider", FakeProvider)
    monkeypatch.setattr(server_module, "get_current_scopes", lambda: ["openid"])
    monkeypatch.setattr(server_module, "set_auth_provider", lambda provider: None)
    monkeypatch.setattr(server_module, "get_oauth_proxy_expiry_kwargs", lambda: {})
    monkeypatch.setattr(server_module, "_auth_provider", server_module._auth_provider)
    monkeypatch.setattr(server_module.server, "auth", server_module.server.auth)
    monkeypatch.delenv("WORKSPACE_MCP_OAUTH_PROXY_STORAGE_BACKEND", raising=False)
    monkeypatch.delenv("WORKSPACE_MCP_OAUTH_PROXY_VALKEY_HOST", raising=False)
    monkeypatch.setattr(
        "auth.oauth_config.get_oauth_config",
        lambda: SimpleNamespace(
            is_oauth21_enabled=lambda: True,
            is_configured=lambda: True,
            is_public_client=lambda: False,
            is_external_oauth21_provider=lambda: False,
            client_id="client-id",
            client_secret="client-secret",
            get_oauth_base_url=lambda: "https://workspace-mcp.example.test",
            redirect_path="/oauth2callback",
        ),
    )
    return captured


@pytest.mark.parametrize("raw", ["", "   ", " , ,"])
def test_whitespace_allowlist_stops_startup(monkeypatch, raw):
    captured = _configure(monkeypatch)
    monkeypatch.setenv(ALLOWED_EMAILS_ENV, raw)
    with pytest.raises(AllowlistConfigError):
        server_module.configure_server_for_http()
    assert captured == {}  # no provider was built


def test_absent_allowlist_starts_with_the_gate_closed_to_everyone(monkeypatch):
    captured = _configure(monkeypatch)
    monkeypatch.delenv(ALLOWED_EMAILS_ENV, raising=False)
    server_module.configure_server_for_http()
    assert captured["allowed_emails"] == frozenset()


def test_configured_allowlist_reaches_the_provider(monkeypatch):
    captured = _configure(monkeypatch)
    monkeypatch.setenv(ALLOWED_EMAILS_ENV, "Mark@Example.com, kay@example.com")
    server_module.configure_server_for_http()
    assert captured["allowed_emails"] == frozenset(
        {"mark@example.com", "kay@example.com"}
    )
