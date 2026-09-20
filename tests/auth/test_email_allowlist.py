"""Tests for the verified-identity gate (WORKSPACE_MCP_ALLOWED_EMAILS)."""

import logging
from types import SimpleNamespace

import pytest
from fastmcp.server.auth import AccessToken
from fastmcp.server.auth.providers.google import GoogleProvider
from key_value.aio.stores.memory import MemoryStore
from starlette.responses import HTMLResponse

import auth.email_allowlist as gate
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

ALLOWED = frozenset({"mark@example.com", "kay@example.com"})
OUTSIDER = "outsider@elsewhere.test"


# --- defect (a): a set-but-empty allowlist fails loud, absence does not ---------


def test_absent_variable_means_not_configured():
    assert parse_allowed_emails(None) is None


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
    assert load_allowed_emails() is None
    monkeypatch.setenv(ALLOWED_EMAILS_ENV, "   ")
    with pytest.raises(AllowlistConfigError):
        load_allowed_emails()
    monkeypatch.setenv(ALLOWED_EMAILS_ENV, "kay@example.com")
    assert load_allowed_emails() == frozenset({"kay@example.com"})


def test_is_email_allowed():
    assert is_email_allowed("anyone@x.test", None)  # no gate configured
    assert is_email_allowed("MARK@example.com ", ALLOWED)
    assert not is_email_allowed(OUTSIDER, ALLOWED)
    assert not is_email_allowed(None, ALLOWED)
    assert not is_email_allowed("", ALLOWED)
    assert not is_email_allowed("mark@example.com", frozenset())  # closed to everyone


# --- defect (b): the rejected address is never logged ---------------------------


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


class _FakeVerifier:
    required_scopes = ["openid"]

    def __init__(self, email, email_verified=True, valid=True):
        self.email, self.email_verified, self.valid = email, email_verified, valid
        self.calls = 0

    async def verify_token(self, token):
        self.calls += 1
        if not self.valid:
            return None
        return AccessToken(
            token=token,
            client_id="sub-1",
            scopes=["openid"],
            claims={"email": self.email, "email_verified": self.email_verified},
        )


@pytest.mark.asyncio
async def test_verifier_admits_allowed_verified_email():
    verifier = AllowlistTokenVerifier(_FakeVerifier("Kay@Example.com"), ALLOWED)
    assert (await verifier.verify_token("t")) is not None
    assert verifier.required_scopes == ["openid"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inner",
    [
        _FakeVerifier(OUTSIDER),
        _FakeVerifier(None),
        _FakeVerifier("mark@example.com", email_verified=False),
        _FakeVerifier("mark@example.com", email_verified=None),
        _FakeVerifier("mark@example.com", email_verified="false"),
        _FakeVerifier("mark@example.com", valid=False),
    ],
)
async def test_verifier_refuses_everything_else(inner, caplog):
    caplog.set_level(logging.DEBUG)
    verifier = AllowlistTokenVerifier(inner, ALLOWED)
    assert (await verifier.verify_token("t")) is None
    assert OUTSIDER not in caplog.text


@pytest.mark.asyncio
async def test_verifier_with_empty_allowlist_refuses_everyone():
    verifier = AllowlistTokenVerifier(_FakeVerifier("mark@example.com"), frozenset())
    assert (await verifier.verify_token("t")) is None


@pytest.mark.asyncio
async def test_verifier_accepts_string_true_from_tokeninfo():
    inner = _FakeVerifier("mark@example.com", email_verified="true")
    assert (await AllowlistTokenVerifier(inner, ALLOWED).verify_token("t")) is not None


# --- OAuth 2.1 path: nothing stored, nothing issued -----------------------------


class _RecordingStore:
    def __init__(self):
        self.items = {}
        self.other_calls = []

    async def put(self, key, value, ttl=None):  # noqa: ARG002
        self.items[key] = value

    async def get(self, key):
        self.other_calls.append(("get", key))
        return self.items.get(key)


def _client_code(access="ya29.x", refresh="1//r"):
    tokens = {"access_token": access}
    if refresh:
        tokens["refresh_token"] = refresh
    return SimpleNamespace(idp_tokens=tokens)


@pytest.fixture
def revoked(monkeypatch):
    calls = []

    async def fake_revoke(token):
        calls.append(token)

    monkeypatch.setattr(gate, "_revoke_google_token", fake_revoke)
    return calls


@pytest.mark.asyncio
async def test_code_store_refuses_outsider_stores_nothing_and_revokes(revoked):
    inner = _RecordingStore()
    store = _GatedCodeStore(
        inner, AllowlistTokenVerifier(_FakeVerifier(OUTSIDER), ALLOWED)
    )

    with pytest.raises(IdentityDenied):
        await store.put(key="code-1", value=_client_code(), ttl=300)

    assert inner.items == {}
    assert revoked == ["1//r"]  # the refresh token when there is one


@pytest.mark.asyncio
async def test_code_store_revokes_access_token_when_no_refresh_token(revoked):
    store = _GatedCodeStore(
        _RecordingStore(), AllowlistTokenVerifier(_FakeVerifier(OUTSIDER), ALLOWED)
    )
    with pytest.raises(IdentityDenied):
        await store.put(key="c", value=_client_code(refresh=None))
    assert revoked == ["ya29.x"]


@pytest.mark.asyncio
async def test_code_store_refuses_when_there_is_no_access_token(revoked):
    inner = _RecordingStore()
    store = _GatedCodeStore(
        inner, AllowlistTokenVerifier(_FakeVerifier("mark@example.com"), ALLOWED)
    )
    with pytest.raises(IdentityDenied):
        await store.put(key="c", value=SimpleNamespace(idp_tokens={}))
    assert inner.items == {}


@pytest.mark.asyncio
async def test_code_store_admits_allowed_and_delegates_the_rest(revoked):
    inner = _RecordingStore()
    store = _GatedCodeStore(
        inner, AllowlistTokenVerifier(_FakeVerifier("mark@example.com"), ALLOWED)
    )
    value = _client_code()
    await store.put(key="code-1", value=value, ttl=300)
    assert inner.items == {"code-1": value}
    assert (await store.get(key="code-1")) is value
    assert revoked == []


@pytest.mark.asyncio
async def test_revoke_failure_is_swallowed_and_logs_no_token(monkeypatch, caplog):
    class _Boom:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise RuntimeError("network down ya29.secret")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(gate.httpx, "AsyncClient", _Boom)
    await gate._revoke_google_token("ya29.secret")
    assert "ya29.secret" not in caplog.text


# --- OAuth 2.1 path: the provider ------------------------------------------------


def _provider(allowed=ALLOWED):
    return AllowlistGoogleProvider(
        client_id="placeholder.apps.googleusercontent.com",
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
    assert isinstance(provider._code_store, _GatedCodeStore)
    assert provider._code_store._verifier is provider._token_validator
    paths = {getattr(r, "path", None) for r in provider.get_routes("/mcp")}
    assert "/oauth2callback" in paths
    # The route must point at our override, not the ungated parent method.
    callback = next(
        r
        for r in provider.get_routes("/mcp")
        if getattr(r, "path", None) == "/oauth2callback"
    )
    assert callback.endpoint.__func__ is AllowlistGoogleProvider._handle_idp_callback


def test_provider_refuses_to_start_if_fastmcp_internals_move(monkeypatch):
    original_init = GoogleProvider.__init__

    def init_without_seam(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        del self._code_store

    monkeypatch.setattr(GoogleProvider, "__init__", init_without_seam)
    with pytest.raises(RuntimeError, match="internals changed"):
        _provider()


@pytest.mark.asyncio
async def test_callback_denial_is_an_explicit_403_not_a_generic_500(
    monkeypatch, revoked, caplog
):
    """The parent handler swallows every exception into a 500 page. Reproduce that
    shape and check the override turns a denial into an explicit 403."""
    caplog.set_level(logging.DEBUG)
    provider = _provider()
    inner = _RecordingStore()
    provider._code_store = _GatedCodeStore(
        inner, AllowlistTokenVerifier(_FakeVerifier(OUTSIDER), ALLOWED)
    )

    async def parent_like_upstream(self, request):  # noqa: ARG001
        try:
            await self._code_store.put(key="code-1", value=_client_code(), ttl=300)
            return HTMLResponse("redirect-with-code", status_code=302)
        except Exception:
            return HTMLResponse("Internal server error", status_code=500)

    monkeypatch.setattr(GoogleProvider, "_handle_idp_callback", parent_like_upstream)

    response = await provider._handle_idp_callback(SimpleNamespace())

    assert response.status_code == 403
    assert DENIAL_MESSAGE in response.body.decode()
    assert OUTSIDER not in response.body.decode()
    assert OUTSIDER not in caplog.text
    assert inner.items == {}
    assert revoked == ["1//r"]


@pytest.mark.asyncio
async def test_callback_passes_through_for_an_allowed_identity(monkeypatch, revoked):
    provider = _provider()
    inner = _RecordingStore()
    provider._code_store = _GatedCodeStore(
        inner, AllowlistTokenVerifier(_FakeVerifier("kay@example.com"), ALLOWED)
    )

    async def parent_like_upstream(self, request):  # noqa: ARG001
        await self._code_store.put(key="code-1", value=_client_code(), ttl=300)
        return HTMLResponse("redirect-with-code", status_code=302)

    monkeypatch.setattr(GoogleProvider, "_handle_idp_callback", parent_like_upstream)

    response = await provider._handle_idp_callback(SimpleNamespace())
    assert response.status_code == 302
    assert "code-1" in inner.items
    assert revoked == []


@pytest.mark.asyncio
async def test_an_unrelated_500_from_the_parent_is_not_relabelled(monkeypatch):
    provider = _provider()

    async def parent_fails_elsewhere(self, request):  # noqa: ARG001
        return HTMLResponse("Internal server error", status_code=500)

    monkeypatch.setattr(GoogleProvider, "_handle_idp_callback", parent_fails_elsewhere)
    response = await provider._handle_idp_callback(SimpleNamespace())
    assert response.status_code == 500
