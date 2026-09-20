"""
Verified-identity gate: an allowlist of Google account emails.

Configured with ``WORKSPACE_MCP_ALLOWED_EMAILS`` (comma-separated). The gate is
enforced on both authentication paths:

* OAuth 2.1 (FastMCP ``GoogleProvider``): ``AllowlistGoogleProvider`` refuses a
  non-allowed identity in the Google callback, before an MCP authorization code
  is stored or issued, and ``AllowlistTokenVerifier`` refuses it again on every
  request and refresh.
* Legacy OAuth 2.0 / stdio: ``handle_auth_callback`` calls
  ``enforce_allowed_email`` before any credential is stored.

Rules this module keeps:

* A value that is set but parses to nothing (whitespace, stray commas) or holds
  an entry that is not an email address is a configuration error and stops the
  server at start-up. It never silently means "everyone" or "no one".
* A rejected email address is never logged; the log line says "denied" only.
* A denial is explicit (HTTP 403 with a plain message), never a generic
  server error.
"""

from __future__ import annotations

import contextvars
import logging
import os
from typing import Any, FrozenSet, Optional

import httpx
from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.auth.providers.google import GoogleProvider
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response

logger = logging.getLogger(__name__)

ALLOWED_EMAILS_ENV = "WORKSPACE_MCP_ALLOWED_EMAILS"
DENIAL_MESSAGE = "Access denied: this account is not authorized to use this server."
GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"


class AllowlistConfigError(ValueError):
    """WORKSPACE_MCP_ALLOWED_EMAILS is set but unusable."""


class IdentityDenied(PermissionError):
    """The authenticated Google identity is not on the allowlist.

    The message never contains the email address.
    """

    def __init__(self) -> None:
        super().__init__(DENIAL_MESSAGE)


def parse_allowed_emails(raw: Optional[str]) -> Optional[FrozenSet[str]]:
    """Parse the allowlist value.

    Returns None when the variable is absent (``raw is None``). Raises
    AllowlistConfigError when it is present but empty, all whitespace, only
    separators, or contains an entry that is not an email address.
    """
    if raw is None:
        return None
    entries = [e.strip().lower() for e in raw.split(",")]
    emails = [e for e in entries if e]
    if not emails:
        raise AllowlistConfigError(
            f"{ALLOWED_EMAILS_ENV} is set but contains no email address. "
            "Unset it or list at least one address."
        )
    for entry in emails:
        local, at, domain = entry.partition("@")
        if not (at and local and "." in domain) or any(c.isspace() for c in entry):
            # Do not echo the entry: a typo next to a real address is still an address.
            raise AllowlistConfigError(
                f"{ALLOWED_EMAILS_ENV} contains an entry that is not an email address."
            )
    return frozenset(emails)


def load_allowed_emails() -> Optional[FrozenSet[str]]:
    """Read and validate the allowlist from the environment."""
    return parse_allowed_emails(os.environ.get(ALLOWED_EMAILS_ENV))


def is_email_allowed(email: Optional[str], allowed: Optional[FrozenSet[str]]) -> bool:
    """True when ``email`` may use the server. ``allowed is None`` means no gate."""
    if allowed is None:
        return True
    if not email or not isinstance(email, str):
        return False
    return email.strip().lower() in allowed


def enforce_allowed_email(
    email: Optional[str], allowed: Optional[FrozenSet[str]]
) -> None:
    """Raise IdentityDenied (and log "denied", never the address) if not allowed."""
    if not is_email_allowed(email, allowed):
        logger.warning("OAuth sign-in denied: identity not in %s", ALLOWED_EMAILS_ENV)
        raise IdentityDenied()


def _claim_is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().lower() == "true"


class AllowlistTokenVerifier(TokenVerifier):
    """Wraps a TokenVerifier; a verified token for a non-allowed email is invalid."""

    def __init__(self, inner: TokenVerifier, allowed: FrozenSet[str]) -> None:
        super().__init__(required_scopes=inner.required_scopes)
        self._inner = inner
        self._allowed = allowed

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        validated = await self._inner.verify_token(token)
        if validated is None:
            return None
        claims = getattr(validated, "claims", None) or {}
        if not _claim_is_true(claims.get("email_verified")) or not is_email_allowed(
            claims.get("email"), self._allowed
        ):
            logger.warning("Token denied: identity not in %s", ALLOWED_EMAILS_ENV)
            return None
        return validated

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this wrapper does not define.
        return getattr(self.__dict__["_inner"], name)


_callback_denied: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "workspace_mcp_callback_denied", default=None
)


class _GatedCodeStore:
    """Proxy for OAuthProxy's authorization-code store.

    ``put`` is the single point where the proxy turns Google's tokens into an
    MCP authorization code. The identity is checked there, so a denied sign-in
    stores nothing and is issued nothing.
    """

    def __init__(self, inner: Any, verifier: AllowlistTokenVerifier) -> None:
        self._inner = inner
        self._verifier = verifier

    async def put(self, *args: Any, **kwargs: Any) -> Any:
        value = kwargs.get("value", args[1] if len(args) > 1 else None)
        idp_tokens = getattr(value, "idp_tokens", None) or {}
        access_token = idp_tokens.get("access_token")
        validated = (
            await self._verifier.verify_token(access_token) if access_token else None
        )
        if validated is None:
            flag = _callback_denied.get()
            if flag is not None:
                flag["denied"] = True
            await _revoke_google_token(idp_tokens.get("refresh_token") or access_token)
            raise IdentityDenied()
        return await self._inner.put(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_inner"], name)


async def _revoke_google_token(token: Optional[str]) -> None:
    """Best effort: do not leave a grant behind for an identity we refused."""
    if not token:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(GOOGLE_REVOKE_URL, data={"token": token})
    except Exception:
        logger.warning("Could not revoke the Google grant of a denied sign-in")


def _denied_response() -> HTMLResponse:
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Access denied</title></head><body>"
        f"<h1>Access denied</h1><p>{DENIAL_MESSAGE}</p>"
        "<p>Nothing was stored. You can close this window.</p></body></html>"
    )
    return HTMLResponse(
        content=html, status_code=403, headers={"Cache-Control": "no-store"}
    )


class AllowlistGoogleProvider(GoogleProvider):
    """FastMCP GoogleProvider with the email allowlist on the OAuth 2.1 path."""

    def __init__(self, *args: Any, allowed_emails: FrozenSet[str], **kwargs: Any):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "_token_validator") or not hasattr(self, "_code_store"):
            # The gate hangs on these two OAuthProxy attributes. If a FastMCP
            # upgrade renames them, refuse to start rather than run ungated.
            raise RuntimeError(
                "FastMCP OAuthProxy internals changed; the email allowlist cannot be "
                "attached. Pin fastmcp to the tested version."
            )
        gated = AllowlistTokenVerifier(self._token_validator, allowed_emails)
        self._token_validator = gated
        self._code_store = _GatedCodeStore(self._code_store, gated)

    async def _handle_idp_callback(
        self, request: Request, *args: Any, **kwargs: Any
    ) -> Response:  # type: ignore[override]
        flag: dict = {"denied": False}
        reset = _callback_denied.set(flag)
        try:
            response = await super()._handle_idp_callback(request, *args, **kwargs)
        finally:
            _callback_denied.reset(reset)
        if flag["denied"]:
            return _denied_response()
        return response
