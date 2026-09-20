"""Tests for the verified-identity gate (WORKSPACE_MCP_ALLOWED_EMAILS)."""

import logging
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from fastmcp.server.auth import AccessToken
from fastmcp.server.auth.oauth_proxy.models import OAuthTransaction
from fastmcp.server.auth.providers.google import GoogleProvider
from key_value.aio.stores.memory import MemoryStore
from starlette.requests import Request
from starlette.responses import HTMLResponse

from auth.email_allowlist import (
    ALLOWED_EMAILS_ENV,
    DENIAL_MESSAGE,
    AllowlistConfigError,
    AllowlistGoogleProvider,
    AllowlistTokenVerifier,
    IdentityDenied,
    _GatedCodeStore,
    enforce_allowed_email,
    is_email_allowed,
    load_allowed_emails,
    parse_allowed_emails,
)
from auth.google_auth import handle_auth_callback
from tests.auth.test_google_auth_callback_refresh_token import (
    _patch_successful_callback,
)

pytestmark = pytest.mark.usefixtures("real_allowlist")

ALLOWED = frozenset({"mark@example.com", "kay@example.com"})
OUTSIDER = "outsider@elsewhere.test"
CLIENT_ID = "placeholder.apps.googleusercontent.com"


# --- configuration: deny by default, fail loud on junk --------------------------


def test_absent_variable_is_the_empty_allowlist():
    assert parse_allowed_emails(None) == frozenset()


@pytest.mark.parametrize("raw", ["", "   ", "\t\n", ",", " , ,, "])
def test_whitespace_or_separator_only_allowlist_is_a_config_error(raw):
    with pytest.raises(AllowlistConfigError):
        parse_allowed_emails(raw)


@pytest.mark.parametrize("raw", ["mark", "mark@", "@example.com", "a@b", "a b@c.de"])
def test_non_email_entry_is_a_config_error_and_is_not_echoed(raw):
    with pytest.raises(AllowlistConfigError) as excinfo:
        parse_allowed_emails(f"good@example.com,{raw}")
    assert raw not in str(excinfo.value)


def test_parse_trims_lowercases_and_dedupes():
    assert (
        parse_allowed_emails(" Mark@Example.com ,kay@example.com,mark@example.com,")
        == ALLOWED
    )


def test_load_reads_the_environment(monkeypatch):
    monkeypatch.delenv(ALLOWED_EMAILS_ENV, raising=False)
    assert load_allowed_emails() == frozenset()
    monkeypatch.setenv(ALLOWED_EMAILS_ENV, "   ")
    with pytest.raises(AllowlistConfigError):
        load_allowed_emails()
    monkeypatch.setenv(ALLOWED_EMAILS_ENV, "kay@example.com")
    assert load_allowed_emails() == frozenset({"kay@example.com"})


def test_is_email_allowed_has_no_open_value():
    assert is_email_allowed("MARK@example.com ", ALLOWED)
    assert not is_email_allowed(OUTSIDER, ALLOWED)
    assert not is_email_allowed(None, ALLOWED)
    assert not is_email_allowed("", ALLOWED)
    assert not is_email_allowed("mark@example.com", frozenset())
    assert not is_email_allowed("mark@example.com", None)


def test_enforce_logs_denied_without_the_address(caplog):
    caplog.set_level(logging.DEBUG)
    with pytest.raises(IdentityDenied) as excinfo:
        enforce_allowed_email(OUTSIDER, ALLOWED)
    assert OUTSIDER not in caplog.text
    assert OUTSIDER not in str(excinfo.value)
    assert "denied" in caplog.text
    assert isinstance(excinfo.value, PermissionError)


# --- legacy OAuth 2.0 / stdio path: handle_auth_callback ------------------------

_STATE = {
    "session_id": "session-1",
    "code_verifier": "verifier",
    "expected_user_email": None,
    "enforce_user_email_match": False,
}


async def _run_legacy_callback():
    return await handle_auth_callback(
        scopes=["scope.a"],
        authorization_response="https://mcp.example/callback?state=abc&code=code",
        redirect_uri="https://mcp.example/callback",
        session_id="session-1",
    )


@pytest.mark.asyncio
async def test_legacy_callback_denies_before_storing_and_never_logs_the_address(
    monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv(ALLOWED_EMAILS_ENV, "mark@example.com")
    oauth_store, credential_store = _patch_successful_callback(
        monkeypatch, state_info=_STATE, google_email=OUTSIDER
    )

    with pytest.raises(IdentityDenied):
        await _run_legacy_callback()

    assert credential_store.saved_credentials is None
    assert oauth_store.store_calls == 0
    assert OUTSIDER not in caplog.text


@pytest.mark.asyncio
async def test_legacy_callback_with_absent_allowlist_denies_everyone(monkeypatch):
    monkeypatch.delenv(ALLOWED_EMAILS_ENV, raising=False)
    oauth_store, credential_store = _patch_successful_callback(
        monkeypatch, state_info=_STATE, google_email="mark@example.com"
    )
    with pytest.raises(IdentityDenied):
        await _run_legacy_callback()
    assert credential_store.saved_credentials is None
    assert oauth_store.store_calls == 0


@pytest.mark.asyncio
async def test_legacy_callback_admits_an_allowed_address(monkeypatch):
    monkeypatch.setenv(ALLOWED_EMAILS_ENV, "Mark@Example.com")
    _, credential_store = _patch_successful_callback(
        monkeypatch, state_info=_STATE, google_email="mark@example.com"
    )
    email, _ = await _run_legacy_callback()
    assert email == "mark@example.com"
    assert credential_store.saved_user_email == "mark@example.com"


@pytest.mark.asyncio
async def test_legacy_callback_with_whitespace_allowlist_fails_loud(monkeypatch):
    monkeypatch.setenv(ALLOWED_EMAILS_ENV, "  ")
    _, credential_store = _patch_successful_callback(
        monkeypatch, state_info=_STATE, google_email="mark@example.com"
    )
    with pytest.raises(AllowlistConfigError):
        await _run_legacy_callback()
    assert credential_store.saved_credentials is None


# --- OAuth 2.1 path: token verifier ---------------------------------------------


class _FakeGoogle:
    """Stands in for GoogleTokenVerifier: what Google says about a token."""

    required_scopes = ["openid"]

    def __init__(
        self, email, email_verified=True, valid=True, aud=CLIENT_ID, sub="sub-1"
    ):
        self.claims = {
            "email": email,
            "email_verified": email_verified,
            "aud": aud,
            "sub": sub,
        }
        self.valid = valid

    async def verify_token(self, token):
        if not self.valid:
            return None
        return AccessToken(
            token=token, client_id="sub-1", scopes=["openid"], claims=dict(self.claims)
        )


def _verifier(inner, allowed=ALLOWED):
    return AllowlistTokenVerifier(inner, allowed, expected_client_id=CLIENT_ID)


@pytest.mark.asyncio
async def test_verifier_admits_allowed_verified_email():
    verifier = _verifier(_FakeGoogle("Kay@Example.com"))
    assert (await verifier.verify_token("t")) is not None
    assert verifier.required_scopes == ["openid"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inner",
    [
        _FakeGoogle(OUTSIDER),
        _FakeGoogle(None),
        _FakeGoogle("mark@example.com", email_verified=False),
        _FakeGoogle("mark@example.com", email_verified=None),
        _FakeGoogle("mark@example.com", email_verified="false"),
        _FakeGoogle("mark@example.com", valid=False),
        _FakeGoogle("mark@example.com", aud="someone-elses-client-id"),
        _FakeGoogle("mark@example.com", aud=None),
        _FakeGoogle("mark@example.com", sub=None),
    ],
)
async def test_verifier_refuses_everything_else(inner, caplog):
    caplog.set_level(logging.DEBUG)
    assert (await _verifier(inner).verify_token("t")) is None
    assert OUTSIDER not in caplog.text


@pytest.mark.asyncio
async def test_verifier_with_empty_allowlist_refuses_everyone():
    verifier = _verifier(_FakeGoogle("mark@example.com"), allowed=frozenset())
    assert (await verifier.verify_token("t")) is None


@pytest.mark.asyncio
async def test_verifier_accepts_string_true_from_tokeninfo():
    inner = _FakeGoogle("mark@example.com", email_verified="true")
    assert (await _verifier(inner).verify_token("t")) is not None


def test_verifier_requires_a_client_id():
    with pytest.raises(ValueError):
        AllowlistTokenVerifier(_FakeGoogle("x@y.zz"), ALLOWED, expected_client_id="")


# --- OAuth 2.1 path: the code store ---------------------------------------------


class _RecordingStore:
    def __init__(self):
        self.items = {}

    async def put(self, key, value, ttl=None):  # noqa: ARG002
        self.items[key] = value

    async def get(self, key):
        return self.items.get(key)


def _client_code(access="ya29.x", refresh="1//r"):
    tokens = {"access_token": access}
    if refresh:
        tokens["refresh_token"] = refresh
    return SimpleNamespace(idp_tokens=tokens)


@pytest.mark.asyncio
async def test_code_store_refuses_outsider_and_stores_nothing():
    inner = _RecordingStore()
    store = _GatedCodeStore(inner, _verifier(_FakeGoogle(OUTSIDER)))
    with pytest.raises(IdentityDenied):
        await store.put(key="code-1", value=_client_code(), ttl=300)
    assert inner.items == {}


@pytest.mark.asyncio
async def test_code_store_refuses_when_there_is_no_access_token():
    inner = _RecordingStore()
    store = _GatedCodeStore(inner, _verifier(_FakeGoogle("mark@example.com")))
    with pytest.raises(IdentityDenied):
        await store.put(key="c", value=SimpleNamespace(idp_tokens={}))
    assert inner.items == {}


@pytest.mark.asyncio
async def test_code_store_admits_allowed_and_delegates_the_rest():
    inner = _RecordingStore()
    store = _GatedCodeStore(inner, _verifier(_FakeGoogle("mark@example.com")))
    value = _client_code()
    await store.put(key="code-1", value=value, ttl=300)
    assert inner.items == {"code-1": value}
    assert (await store.get(key="code-1")) is value


def test_denial_makes_no_network_call():
    """Denial is local; revoking the Google grant is an operator action."""
    import auth.email_allowlist as gate

    assert not hasattr(gate, "httpx")
    assert not hasattr(gate, "_revoke_google_token")


# --- OAuth 2.1 path: the provider, against the REAL pinned callback ------------


def _provider(allowed=ALLOWED):
    return AllowlistGoogleProvider(
        client_id=CLIENT_ID,
        client_secret="placeholder-secret-for-tests",
        base_url="https://workspace.example.test",
        redirect_path="/oauth2callback",
        required_scopes=["openid"],
        client_storage=MemoryStore(),
        jwt_signing_key="a-test-signing-key-of-decent-length",
        allowed_emails=allowed,
    )


def test_provider_wraps_both_seams_of_the_real_oauth_proxy():
    provider = _provider()
    assert isinstance(provider._token_validator, AllowlistTokenVerifier)
    assert provider._token_validator._expected_client_id == CLIENT_ID
    assert isinstance(provider._code_store, _GatedCodeStore)
    assert provider._code_store._verifier is provider._token_validator
    callback = next(
        r
        for r in provider.get_routes("/mcp")
        if getattr(r, "path", None) == "/oauth2callback"
    )
    # The route must point at our override, not the ungated parent method.
    assert callback.endpoint.__func__ is AllowlistGoogleProvider._handle_idp_callback


def test_provider_refuses_to_start_if_fastmcp_internals_move(monkeypatch):
    original_init = GoogleProvider.__init__

    def init_without_seam(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        del self._code_store

    monkeypatch.setattr(GoogleProvider, "__init__", init_without_seam)
    with pytest.raises(RuntimeError, match="internals changed"):
        _provider()


CLIENT_REDIRECT = "https://claude.example.test/api/mcp/auth_callback"
GOOGLE_TOKENS = {
    "access_token": "ya29.from-google",
    "refresh_token": "1//from-google",
    "expires_in": 3599,
    "scope": "openid",
    "token_type": "Bearer",
}


async def _real_callback(provider, monkeypatch, google_says):
    """Drive FastMCP's own _handle_idp_callback. Only Google is faked: the code
    exchange (fetch_token) and what tokeninfo says about the returned token."""
    # The consent-binding cookie is a separate upstream control with its own
    # upstream tests; switch it off so the request reaches the code exchange.
    provider._require_authorization_consent = False
    provider._token_validator._inner = google_says

    await provider._transaction_store.put(
        key="txn-1",
        value=OAuthTransaction(
            txn_id="txn-1",
            client_id="mcp-client-1",
            client_redirect_uri=CLIENT_REDIRECT,
            client_state="client-state-1",
            code_challenge="challenge",
            code_challenge_method="S256",
            scopes=["openid"],
            created_at=time.time(),
        ),
    )

    @asynccontextmanager
    async def fake_google_client():
        async def fetch_token(**kwargs):  # noqa: ARG001
            return dict(GOOGLE_TOKENS)

        yield SimpleNamespace(fetch_token=fetch_token)

    monkeypatch.setattr(provider, "_upstream_oauth_client", fake_google_client)

    puts = []
    inner_store = provider._code_store._inner
    original_put = inner_store.put

    async def spy_put(*args, **kwargs):
        puts.append(kwargs.get("key"))
        return await original_put(*args, **kwargs)

    monkeypatch.setattr(inner_store, "put", spy_put)

    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/oauth2callback",
            "query_string": b"code=google-code&state=txn-1",
            "headers": [],
        }
    )
    response = await provider._handle_idp_callback(request)
    return response, puts


@pytest.mark.asyncio
async def test_real_callback_denies_an_outsider_explicitly(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    provider = _provider()
    response, puts = await _real_callback(provider, monkeypatch, _FakeGoogle(OUTSIDER))

    assert response.status_code == 403
    body = response.body.decode()
    assert DENIAL_MESSAGE in body
    assert "location" not in response.headers  # no redirect carrying a code
    assert puts == []  # no authorization code was stored
    for leaked in (OUTSIDER, "ya29.from-google", "1//from-google"):
        assert leaked not in body
        assert leaked not in caplog.text


@pytest.mark.asyncio
async def test_real_callback_with_empty_allowlist_denies_a_household_address(
    monkeypatch,
):
    provider = _provider(allowed=frozenset())
    response, puts = await _real_callback(
        provider, monkeypatch, _FakeGoogle("mark@example.com")
    )
    assert response.status_code == 403
    assert puts == []


@pytest.mark.asyncio
async def test_real_callback_admits_an_allowed_identity(monkeypatch):
    provider = _provider()
    response, puts = await _real_callback(
        provider, monkeypatch, _FakeGoogle("kay@example.com")
    )
    assert response.status_code == 302
    location = urlparse(response.headers["location"])
    assert f"{location.scheme}://{location.netloc}{location.path}" == CLIENT_REDIRECT
    query = parse_qs(location.query)
    assert query["state"] == ["client-state-1"]
    assert puts == query["code"]  # exactly one code, the one handed to the client


@pytest.mark.asyncio
async def test_an_unrelated_500_from_the_parent_is_not_relabelled(monkeypatch):
    provider = _provider()

    async def parent_fails_elsewhere(self, request):  # noqa: ARG001
        return HTMLResponse("Internal server error", status_code=500)

    monkeypatch.setattr(GoogleProvider, "_handle_idp_callback", parent_fails_elsewhere)
    response = await provider._handle_idp_callback(SimpleNamespace())
    assert response.status_code == 500
