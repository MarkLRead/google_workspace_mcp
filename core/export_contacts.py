"""Print one JSON snapshot of this instance's Google Contacts, read-only.

Run as the instance's own service user, with the instance's own environment
(the launcher's env file), from any code tree whose FastMCP matches the
server's::

    python -m core.export_contacts            # the snapshot on stdout
    python -m core.export_contacts --check    # counts only, no contacts read

It reuses the Google grant the server already holds: the OAuth proxy keeps one
encrypted record per client sign-in in its disk store, and this command opens
that store with the server's own key derivation (``auth/oauth_proxy_storage``).
It never writes the store. The access token is refreshed in memory only.

A grant is used only when Google's sign-in record (OpenID ``userinfo``) names
the ONE address on ``WORKSPACE_MCP_ALLOWED_EMAILS`` as verified. Then it lists
the regular contacts (names, phone numbers, emails, organisations, group names)
and the contact groups, both to the last page. Nothing is printed unless every
page was read: a partial address book never leaves this process.

stdout carries the snapshot and nothing else. stderr carries one line per
problem: a short code and a message, never a token or a contact. One line goes
to the instance's audit log (``WORKSPACE_MCP_AUDIT_LOG``), as for a tool call.

Exit: 0 snapshot printed; 2 no usable grant; 3 Google failed (network,
throttling, a server error); 4 configuration (environment, store, audit log).
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("export_contacts")

SCHEMA = 1
TOOL = "export_contacts"
UPSTREAM_COLLECTION = "mcp-upstream-tokens"
CONTACT_SCOPES = (
    "https://www.googleapis.com/auth/contacts",
    "https://www.googleapis.com/auth/contacts.readonly",
)
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
CONNECTIONS_URL = "https://people.googleapis.com/v1/people/me/connections"
GROUPS_URL = "https://people.googleapis.com/v1/contactGroups"
TOKEN_URI = "https://oauth2.googleapis.com/token"
PERSON_FIELDS = "names,phoneNumbers,emailAddresses,organizations,memberships"
PAGE_SIZE = 1000
MAX_PAGES = 50  # 50,000 contacts; a household book is a few hundred
TIMEOUT_S = 30
RETRY_DELAYS_S = (2, 5)  # two retries on 429 / 5xx

EXIT_OK, EXIT_NO_GRANT, EXIT_GOOGLE, EXIT_CONFIG = 0, 2, 3, 4


class ExportError(Exception):
    """A failure with its exit code and a message that holds no secret."""

    def __init__(self, code: int, tag: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.tag = tag


class CredentialRejected(Exception):
    """Google refused this grant (revoked, expired): try the next one."""


# ---------------------------------------------------------------- the store


def _collection_dir(directory: str) -> str:
    return os.path.join(directory, UPSTREAM_COLLECTION)


def check_store(directory: str) -> None:
    """Refuse a store that is not there, so opening it creates nothing."""
    coll = _collection_dir(directory)
    info = os.path.join(directory, UPSTREAM_COLLECTION + "-info.json")
    if not os.path.isdir(coll) or not os.path.isfile(info):
        raise ExportError(
            EXIT_CONFIG, "store", "the OAuth proxy store has no upstream-token collection here"
        )


def list_grant_keys(directory: str) -> List[str]:
    """The keys of the stored grants: the ``*.json`` names in the collection."""
    names = []
    for name in os.listdir(_collection_dir(directory)):
        if name.endswith(".json") and not name.startswith("."):
            names.append(name[: -len(".json")])
    return sorted(names)


async def read_grants(store: Any, keys: List[str]) -> Tuple[List[Any], Dict[str, int]]:
    """Decrypt and validate each grant; skip what cannot be used, counted by reason."""
    from fastmcp.server.auth.oauth_proxy.models import UpstreamTokenSet
    from key_value.aio.adapters.pydantic import PydanticAdapter

    adapter = PydanticAdapter[UpstreamTokenSet](
        key_value=store,
        pydantic_model=UpstreamTokenSet,
        default_collection=UPSTREAM_COLLECTION,
        raise_on_validation_error=True,
    )
    grants: List[Any] = []
    skipped: Dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for key in keys:
        try:
            grant = await adapter.get(key=key)
        except Exception as exc:  # undecryptable, invalid, a file that vanished
            skip("unreadable" if not isinstance(exc, FileNotFoundError) else "vanished")
            continue
        if grant is None:
            skip("expired_or_gone")
            continue
        if grant.upstream_token_id != key:
            skip("key_mismatch")
            continue
        if not grant.refresh_token:
            skip("no_refresh_token")
            continue
        scopes = set((grant.scope or "").split())
        if not scopes.intersection(CONTACT_SCOPES):
            skip("no_contacts_scope")
            continue
        grants.append(grant)
    grants.sort(key=lambda g: g.created_at or 0, reverse=True)
    return grants, skipped


# ---------------------------------------------------------------- Google


def _timed_request():
    """A google-auth transport whose every call has a deadline."""
    import google.auth.transport.requests as gtr

    class _Timed(gtr.Request):
        def __call__(self, url, method="GET", body=None, headers=None, timeout=None, **kwargs):
            return super().__call__(
                url, method=method, body=body, headers=headers, timeout=TIMEOUT_S, **kwargs
            )

    return _Timed()


def credentials_for(grant: Any, client_id: str, client_secret: str):
    from google.oauth2.credentials import Credentials

    return Credentials(
        token=None,  # always refresh: proves the grant still works
        refresh_token=grant.refresh_token,
        token_uri=TOKEN_URI,
        client_id=client_id,
        client_secret=client_secret,
        scopes=(grant.scope or "").split() or None,
    )


def refresh(creds: Any) -> None:
    """Refresh in memory. A refusal by Google is CredentialRejected; anything else is Google failing."""
    from google.auth.exceptions import RefreshError, TransportError

    try:
        creds.refresh(_timed_request())
    except RefreshError as exc:
        # invalid_grant and friends: this grant is dead, the next one may not be.
        # A 5xx from the token endpoint also lands here; it says "retryable".
        if getattr(exc, "retryable", False):
            raise ExportError(EXIT_GOOGLE, "token_endpoint", "Google's token endpoint failed") from None
        raise CredentialRejected() from None
    except TransportError:
        raise ExportError(EXIT_GOOGLE, "network", "could not reach Google's token endpoint") from None


class Google:
    """GET with a deadline and two retries on 429 / 5xx. Never logs a URL with a token."""

    def __init__(self, creds: Any, sleep: Callable[[float], None] = time.sleep) -> None:
        from google.auth.transport.requests import AuthorizedSession

        self.session = AuthorizedSession(creds)
        self.sleep = sleep

    def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        import requests
        from google.auth.exceptions import RefreshError, TransportError

        delays = list(RETRY_DELAYS_S)
        while True:
            try:
                resp = self.session.get(url, params=params, timeout=TIMEOUT_S)
            except RefreshError:
                # AuthorizedSession refreshes by itself on a 401; a refusal there is this grant dying mid-run
                raise CredentialRejected() from None
            except (requests.RequestException, TransportError):
                status = None
            else:
                status = resp.status_code
                if status == 200:
                    try:
                        data = resp.json()
                    except ValueError:
                        raise ExportError(EXIT_GOOGLE, "bad_json", "Google sent a reply that is not JSON") from None
                    if not isinstance(data, dict):
                        raise ExportError(EXIT_GOOGLE, "bad_json", "Google sent a reply that is not an object")
                    return data
                # 401: the grant is refused. 403 on the identity call: no right to read it with this grant.
                # A 403 from the People API (quota, API off) is Google failing, not the grant: exit 3.
                if status == 401 or (status == 403 and url == USERINFO_URL):
                    raise CredentialRejected()
            if (status is None or status == 429 or status >= 500) and delays:
                self.sleep(delays.pop(0))
                continue
            what = "no answer" if status is None else f"HTTP {status}"
            raise ExportError(EXIT_GOOGLE, "google", f"Google failed ({what})")

    def pages(self, url: str, params: Dict[str, Any], items_key: str) -> List[Dict[str, Any]]:
        """Every item to the last page; a page cap with a token still pending is a failure."""
        items: List[Dict[str, Any]] = []
        token: Optional[str] = None
        for _ in range(MAX_PAGES):
            query = dict(params)
            if token:
                query["pageToken"] = token
            data = self.get(url, query)
            page = data.get(items_key) or []
            if not isinstance(page, list):
                raise ExportError(EXIT_GOOGLE, "bad_json", "Google sent a list that is not a list")
            items.extend(x for x in page if isinstance(x, dict))
            token = data.get("nextPageToken")
            if not token:
                return items
        raise ExportError(EXIT_GOOGLE, "page_cap", f"more than {MAX_PAGES} pages; nothing printed")


def verified_account(google: Google, allowed: str) -> Dict[str, str]:
    """Google's sign-in record for this token must be the allowlisted address, verified."""
    info = google.get(USERINFO_URL)
    email = (info.get("email") or "").strip().lower()
    if info.get("email_verified") is not True or not email or email != allowed:
        raise CredentialRejected()  # a grant for another account: never used
    sub = info.get("sub")
    return {"account": email, "sub": sub if isinstance(sub, str) else ""}


# ---------------------------------------------------------------- the snapshot


def _primary_first(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    good = [x for x in items if isinstance(x, dict)]
    return sorted(good, key=lambda x: not (x.get("metadata") or {}).get("primary", False))


def shape_contact(person: Dict[str, Any], group_names: Dict[str, str]) -> Dict[str, Any]:
    names = _primary_first(person.get("names"))
    contact: Dict[str, Any] = {
        "resource": person.get("resourceName") or "",
        "name": (names[0].get("displayName") or "") if names else "",
        "phones": [],
        "emails": [],
        "orgs": [],
        "groups": [],
    }
    for p in _primary_first(person.get("phoneNumbers")):
        value = p.get("value")
        if isinstance(value, str) and value.strip():
            contact["phones"].append(
                {"value": value, "canonical": p.get("canonicalForm") or "", "type": p.get("type") or ""}
            )
    for e in _primary_first(person.get("emailAddresses")):
        value = e.get("value")
        if isinstance(value, str) and value.strip():
            contact["emails"].append(value)
    for o in _primary_first(person.get("organizations")):
        name, title = o.get("name") or "", o.get("title") or ""
        if name or title:
            contact["orgs"].append({"name": name, "title": title})
    unresolved = []
    for m in person.get("memberships") or []:
        if not isinstance(m, dict):
            continue
        rn = (m.get("contactGroupMembership") or {}).get("contactGroupResourceName")
        if not isinstance(rn, str):
            continue
        if rn not in group_names:
            unresolved.append(rn)  # a group the list did not return: kept as its id, never dropped
        elif group_names[rn] is None:
            unresolved.append(rn)  # a user group without a name
        elif group_names[rn]:
            contact["groups"].append(group_names[rn])
        # "" = a system group (myContacts, starred): known, not a label
    if unresolved:
        contact["groups_unresolved"] = sorted(unresolved)
    return contact


def group_name_map(groups: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """User groups by resource name (None when unnamed); system groups map to "" (known, not a label)."""
    out: Dict[str, Optional[str]] = {}
    for g in groups:
        rn = g.get("resourceName")
        if not isinstance(rn, str):
            continue
        if g.get("groupType") == "USER_CONTACT_GROUP":
            out[rn] = g.get("name") or g.get("formattedName") or None
        else:
            out[rn] = ""
    return out


def build_snapshot(google: Google, who: Dict[str, str]) -> Dict[str, Any]:
    groups = google.pages(GROUPS_URL, {"pageSize": PAGE_SIZE, "groupFields": "name,formattedName,groupType"}, "contactGroups")
    people = google.pages(
        CONNECTIONS_URL, {"pageSize": PAGE_SIZE, "personFields": PERSON_FIELDS}, "connections"
    )
    names = group_name_map(groups)
    contacts = [shape_contact(p, names) for p in people]
    return {
        "schema": SCHEMA,
        "account": who["account"],
        "sub": who["sub"],
        "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(contacts),
        "contacts": contacts,
    }


# ---------------------------------------------------------------- the run


def _config() -> Tuple[str, str, str, bytes]:
    """(the one allowed address, client id, client secret, JWT key) or ExportError."""
    from auth.email_allowlist import load_allowed_emails
    from auth.oauth_config import get_oauth_config
    from auth.oauth_proxy_storage import (
        STORAGE_BACKEND_ENV,
        jwt_signing_key_override,
        validate_and_derive_jwt_key,
    )

    if os.getenv(STORAGE_BACKEND_ENV, "").strip().lower() != "disk":
        raise ExportError(EXIT_CONFIG, "backend", "only the disk storage backend is supported")
    try:
        allowed = sorted(load_allowed_emails())
    except Exception:
        raise ExportError(EXIT_CONFIG, "allowlist", "the email allowlist cannot be read") from None
    if len(allowed) != 1:
        raise ExportError(EXIT_CONFIG, "allowlist", f"the allowlist must hold exactly one address (holds {len(allowed)})")
    config = get_oauth_config()
    if not config.client_id or not config.client_secret:
        raise ExportError(EXIT_CONFIG, "client", "the Google client id or secret is missing")
    try:
        jwt_key = validate_and_derive_jwt_key(jwt_signing_key_override(), config.client_secret)
    except ValueError:
        raise ExportError(EXIT_CONFIG, "key", "no signing key to open the store with") from None
    return allowed[0], config.client_id, config.client_secret, jwt_key


def _open_audit():
    """The instance's audit log. Required: an export that leaves no trace is not run."""
    from core.audit_log import AuditLogConfigError, load_audit_log

    try:
        log = load_audit_log()
    except AuditLogConfigError:
        raise ExportError(EXIT_CONFIG, "audit", "the audit log cannot be opened for append") from None
    if log is None:
        raise ExportError(EXIT_CONFIG, "audit", "WORKSPACE_MCP_AUDIT_LOG is not set; nothing is exported untraced")
    return log


def _audit(ok: bool, started: float, items: Optional[int], err: Optional[str]) -> None:
    log = _open_audit()
    record: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": TOOL,
        "ok": ok,
        "ms": int((time.monotonic() - started) * 1000),
        "caller": "scheduled",
    }
    if items is not None:
        record["items"] = items
    if err:
        record["err"] = err
    try:
        written = log.write(record)
    finally:
        log.close()
    if not written:
        raise ExportError(EXIT_CONFIG, "audit", "the audit line could not be written; nothing printed")


def run(check_only: bool, out=None, sleep: Callable[[float], None] = time.sleep) -> int:
    from auth.oauth_proxy_storage import disk_directory, encrypted_disk_store

    out = out if out is not None else sys.stdout  # resolved at call time (tests replace sys.stdout)
    started = time.monotonic()
    allowed, client_id, client_secret, jwt_key = _config()
    directory = disk_directory()
    check_store(directory)
    _open_audit().close()  # the audit log must open before Google is called

    store = encrypted_disk_store(directory, jwt_key)
    grants, skipped = asyncio.run(read_grants(store, list_grant_keys(directory)))
    if skipped:
        print("export_contacts: skipped " + ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())), file=sys.stderr)
    if check_only:
        print(json.dumps({"usable_grants": len(grants), "skipped": skipped}, sort_keys=True), file=out)
        return EXIT_OK if grants else EXIT_NO_GRANT

    rejected = 0
    try:
        for grant in grants:
            creds = credentials_for(grant, client_id, client_secret)
            try:
                refresh(creds)
                google = Google(creds, sleep=sleep)
                who = verified_account(google, allowed)
                snapshot = build_snapshot(google, who)
            except CredentialRejected:
                rejected += 1
                continue
            text = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
            _audit(True, started, snapshot["count"], None)
            out.write(text + "\n")
            out.flush()
            return EXIT_OK
    except ExportError as exc:
        if exc.tag != "audit":
            _audit(False, started, None, exc.tag)
        raise
    except Exception:  # anything unforeseen still leaves its trace, then the generic exit
        _audit(False, started, None, "internal")
        raise
    _audit(False, started, None, "no_grant")
    raise ExportError(
        EXIT_NO_GRANT, "no_grant", f"no usable grant ({len(grants)} tried, {rejected} refused by Google)"
    )


def main(argv: Optional[List[str]] = None) -> int:
    class _Parser(argparse.ArgumentParser):
        def error(self, message):  # a usage error is configuration (4), never "no usable grant" (argparse's 2)
            print(f"export_contacts: usage: {message}", file=sys.stderr)
            sys.exit(EXIT_CONFIG)

    parser = _Parser(prog="python -m core.export_contacts", description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="counts of usable grants only; no Google call")
    args = parser.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="export_contacts: %(levelname)s %(message)s")
    try:
        return run(args.check)
    except ExportError as exc:
        print(f"export_contacts: {exc.tag}: {exc}", file=sys.stderr)
        return exc.code
    except Exception as exc:  # never a traceback that could carry a value
        print(f"export_contacts: internal: {type(exc).__name__}", file=sys.stderr)
        return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
