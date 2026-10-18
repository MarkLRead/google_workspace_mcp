"""core/export_contacts.py: the read-only contacts snapshot (market-pulse backlog #384)."""

import asyncio
import io
import json
import os
import time

import pytest
from cryptography.fernet import Fernet
from fastmcp.server.auth.jwt_issuer import derive_jwt_key
from fastmcp.server.auth.oauth_proxy.models import UpstreamTokenSet
from key_value.aio.adapters.pydantic import PydanticAdapter

import core.export_contacts as ec
from auth.oauth_proxy_storage import (
    derive_storage_encryption_key,
    encrypted_disk_store,
    validate_and_derive_jwt_key,
)

SIGNING = "fixture-key-material-0123"
ALLOWED = "owner@example.com"
REFRESH_FIXTURE = "FIXTURE-REFRESH-0001"
ACCESS_FIXTURE = "FIXTURE-ACCESS-0002"
CONTACTS = "https://www.googleapis.com/auth/contacts"


# ---------------------------------------------------------------- helpers


def _grant(token_id, scope=CONTACTS + " openid email", refresh=REFRESH_FIXTURE, created=None):
    now = time.time()
    return UpstreamTokenSet(
        upstream_token_id=token_id,
        access_token=ACCESS_FIXTURE,
        refresh_token=refresh,
        refresh_token_expires_at=None,
        expires_at=now + 3600,
        token_type="Bearer",
        scope=scope,
        client_id="mcp-client",
        created_at=created if created is not None else now,
    )


async def _put(directory, key_bytes, grant, ttl=3600):
    """Write a grant exactly the way FastMCP's OAuthProxy does."""
    adapter = PydanticAdapter[UpstreamTokenSet](
        key_value=encrypted_disk_store(directory, key_bytes),
        pydantic_model=UpstreamTokenSet,
        default_collection=ec.UPSTREAM_COLLECTION,
        raise_on_validation_error=True,
    )
    await adapter.put(key=grant.upstream_token_id, value=grant, ttl=ttl)


def _key():
    return validate_and_derive_jwt_key(SIGNING, None)


def _tid(n):  # the shape the proxy uses: 43 url-safe characters
    return (f"T{n}" + "x" * 43)[:43]


class FakeGoogle:
    """Scripted replies per URL; each entry is a dict (a page) or an exception."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def get(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        item = self.script[url].pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    pages = ec.Google.pages


def _userinfo(email=ALLOWED, verified=True):
    return {"email": email, "email_verified": verified, "sub": "123"}


# ---------------------------------------------------------------- the key helper


def test_key_helper_matches_the_old_inline_derivation():
    old_jwt = derive_jwt_key(low_entropy_material=SIGNING, salt="fastmcp-jwt-signing-key")
    old_storage = derive_jwt_key(
        high_entropy_material=old_jwt.decode(), salt="fastmcp-storage-encryption-key"
    )
    assert validate_and_derive_jwt_key(SIGNING, "ignored-secret") == old_jwt
    assert derive_storage_encryption_key(old_jwt) == old_storage
    from_secret = derive_jwt_key(high_entropy_material="cs", salt="fastmcp-jwt-signing-key")
    assert validate_and_derive_jwt_key(None, "cs") == from_secret
    with pytest.raises(ValueError):
        validate_and_derive_jwt_key(None, None)
    Fernet(old_storage)  # a usable Fernet key


# ---------------------------------------------------------------- reading the store


def test_store_written_by_the_proxy_is_read_back_and_bad_grants_are_skipped(tmp_path):
    d = str(tmp_path)
    key = _key()
    other_key = validate_and_derive_jwt_key("fixture-other-material-9", None)

    async def scenario():
        await _put(d, key, _grant(_tid(1), created=100))
        await _put(d, key, _grant(_tid(2), created=300))
        await _put(d, key, _grant(_tid(3), scope="openid email"))
        await _put(d, key, _grant(_tid(4), refresh=None))
        await _put(d, other_key, _grant(_tid(5)))
        await _put(d, key, _grant(_tid(6)), ttl=1)
        await asyncio.sleep(1.2)
        ec.check_store(d)
        store = encrypted_disk_store(d, key)
        return await ec.read_grants(store, ec.list_grant_keys(d))

    grants, skipped = asyncio.run(scenario())
    assert [g.upstream_token_id for g in grants] == [_tid(2), _tid(1)]  # newest first
    assert skipped == {
        "no_contacts_scope": 1,
        "no_refresh_token": 1,
        "unreadable": 1,
        "expired_or_gone": 1,
    }


def test_reading_never_writes_the_store(tmp_path):
    d = str(tmp_path)
    key = _key()
    asyncio.run(_put(d, key, _grant(_tid(1))))

    def snapshot():
        out = {}
        for root, _, files in os.walk(d):
            for f in files:
                p = os.path.join(root, f)
                with open(p, "rb") as fh:
                    out[p] = (fh.read(), os.stat(p).st_mtime_ns)
        return out

    before = snapshot()
    store = encrypted_disk_store(d, key)
    asyncio.run(ec.read_grants(store, ec.list_grant_keys(d)))
    assert snapshot() == before


def test_a_missing_store_is_refused_and_not_created(tmp_path):
    d = str(tmp_path / "nothing-here")
    with pytest.raises(ec.ExportError) as err:
        ec.check_store(d)
    assert err.value.code == ec.EXIT_CONFIG
    assert not os.path.exists(d)


# ---------------------------------------------------------------- Google: identity and pages


def test_identity_must_be_the_allowlisted_verified_address():
    ok = FakeGoogle({ec.USERINFO_URL: [_userinfo()]})
    assert ec.verified_account(ok, ALLOWED) == {"account": ALLOWED, "sub": "123"}
    for info in (_userinfo(email="stranger@example.com"), _userinfo(verified=False), {"sub": "1"}):
        with pytest.raises(ec.CredentialRejected):
            ec.verified_account(FakeGoogle({ec.USERINFO_URL: [info]}), ALLOWED)


def test_both_lists_are_paged_to_the_end_and_groups_resolve():
    g = FakeGoogle(
        {
            ec.GROUPS_URL: [
                {"contactGroups": [{"resourceName": "contactGroups/a", "groupType": "USER_CONTACT_GROUP", "name": "Family"}], "nextPageToken": "g2"},
                {"contactGroups": [{"resourceName": "contactGroups/myContacts", "groupType": "SYSTEM_CONTACT_GROUP"}]},
            ],
            ec.CONNECTIONS_URL: [
                {"connections": [{"resourceName": "people/1", "names": [{"displayName": "Aunt Jane"}],
                                  "phoneNumbers": [{"value": "(555) 010-0001", "canonicalForm": "+15550100001", "type": "mobile"}],
                                  "memberships": [{"contactGroupMembership": {"contactGroupResourceName": "contactGroups/a"}},
                                                  {"contactGroupMembership": {"contactGroupResourceName": "contactGroups/myContacts"}},
                                                  {"contactGroupMembership": {"contactGroupResourceName": "contactGroups/zzz"}}]}],
                 "nextPageToken": "p2"},
                {"connections": [{"resourceName": "people/2", "emailAddresses": [{"value": "doc@example.org"}],
                                  "organizations": [{"name": "Care Home", "title": "Nurse"}]}]},
            ],
        }
    )
    snap = ec.build_snapshot(g, {"account": ALLOWED, "sub": "123"})
    assert snap["count"] == 2 == len(snap["contacts"])
    jane, doc = snap["contacts"]
    assert jane["name"] == "Aunt Jane" and jane["groups"] == ["Family"]
    assert jane["groups_unresolved"] == ["contactGroups/zzz"]
    assert jane["phones"] == [{"value": "(555) 010-0001", "canonical": "+15550100001", "type": "mobile"}]
    assert doc["emails"] == ["doc@example.org"] and doc["orgs"] == [{"name": "Care Home", "title": "Nurse"}]
    tokens = [p.get("pageToken") for u, p in g.calls]
    assert tokens == [None, "g2", None, "p2"]


def test_page_cap_with_a_token_pending_fails(monkeypatch):
    monkeypatch.setattr(ec, "MAX_PAGES", 2)
    g = FakeGoogle({ec.GROUPS_URL: [{"contactGroups": [], "nextPageToken": "x"}] * 3})
    with pytest.raises(ec.ExportError) as err:
        g.pages(ec.GROUPS_URL, {}, "contactGroups")
    assert err.value.tag == "page_cap" and err.value.code == ec.EXIT_GOOGLE


class _Resp:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


class _Session:
    def __init__(self, replies):
        self.replies = list(replies)
        self.timeouts = []

    def get(self, url, params=None, timeout=None):
        self.timeouts.append(timeout)
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def _google_with(replies):
    g = ec.Google.__new__(ec.Google)
    g.session = _Session(replies)
    g.slept = []
    g.sleep = g.slept.append
    return g


def test_retries_on_429_and_5xx_then_fails_and_401_is_a_credential_refusal():
    import requests

    g = _google_with([_Resp(429), _Resp(503), _Resp(200, {"ok": 1})])
    assert g.get("u") == {"ok": 1} and g.slept == list(ec.RETRY_DELAYS_S)
    assert set(g.session.timeouts) == {ec.TIMEOUT_S}

    g = _google_with([requests.ConnectionError(), _Resp(500), _Resp(502)])
    with pytest.raises(ec.ExportError) as err:
        g.get("u")
    assert err.value.code == ec.EXIT_GOOGLE

    with pytest.raises(ec.CredentialRejected):
        _google_with([_Resp(401)]).get("u")


# ---------------------------------------------------------------- the whole run


@pytest.fixture
def instance(tmp_path, monkeypatch):
    d = tmp_path / "oauth-proxy"
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("WORKSPACE_MCP_OAUTH_PROXY_STORAGE_BACKEND", "disk")
    monkeypatch.setenv("WORKSPACE_MCP_OAUTH_PROXY_DISK_DIRECTORY", str(d))
    monkeypatch.setenv("FASTMCP_SERVER_AUTH_GOOGLE_JWT_SIGNING_KEY", SIGNING)
    monkeypatch.setenv("WORKSPACE_MCP_ALLOWED_EMAILS", ALLOWED)
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "cid.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "fixture-client-cs")
    monkeypatch.setenv("WORKSPACE_MCP_AUDIT_LOG", str(audit))
    import auth.oauth_config as oc

    monkeypatch.setattr(oc, "_oauth_config", None, raising=False)
    monkeypatch.setattr(oc, "get_oauth_config", lambda: type("C", (), {"client_id": "cid", "client_secret": "cs"})())
    return d, audit


def _script_ok():
    return {
        ec.USERINFO_URL: [_userinfo()],
        ec.GROUPS_URL: [{"contactGroups": []}],
        ec.CONNECTIONS_URL: [{"connections": [{"resourceName": "people/1", "names": [{"displayName": "A"}]}]}],
    }


def _wire(monkeypatch, refresh_results, google_scripts):
    """refresh_results: per grant tried, None (works) or an exception; google_scripts likewise."""
    refreshes = list(refresh_results)
    scripts = list(google_scripts)

    def fake_refresh(creds):
        r = refreshes.pop(0)
        if r is not None:
            raise r

    monkeypatch.setattr(ec, "refresh", fake_refresh)
    monkeypatch.setattr(ec, "Google", lambda creds, sleep=None: FakeGoogle(scripts.pop(0)))


def test_falls_through_a_refused_grant_and_prints_one_snapshot(instance, monkeypatch, capsys):
    d, audit = instance
    asyncio.run(_put(str(d), _key(), _grant(_tid(1), created=100)))
    asyncio.run(_put(str(d), _key(), _grant(_tid(2), created=200)))
    _wire(monkeypatch, [ec.CredentialRejected(), None], [_script_ok()])
    out = io.StringIO()
    assert ec.run(False, out=out) == ec.EXIT_OK
    snap = json.loads(out.getvalue())
    assert snap["schema"] == 1 and snap["account"] == ALLOWED and snap["count"] == 1
    line = json.loads(audit.read_text().strip())
    assert line["tool"] == "export_contacts" and line["ok"] is True and line["items"] == 1
    assert line["caller"] == "scheduled"


def test_wrong_account_grant_is_never_used(instance, monkeypatch):
    d, _ = instance
    asyncio.run(_put(str(d), _key(), _grant(_tid(1))))
    _wire(monkeypatch, [None], [{ec.USERINFO_URL: [_userinfo(email="stranger@example.com")]}])
    out = io.StringIO()
    with pytest.raises(ec.ExportError) as err:
        ec.run(False, out=out)
    assert err.value.code == ec.EXIT_NO_GRANT and out.getvalue() == ""


def test_a_failed_page_prints_nothing_and_no_secret_anywhere(instance, monkeypatch, capsys):
    d, audit = instance
    asyncio.run(_put(str(d), _key(), _grant(_tid(1))))
    script = _script_ok()
    script[ec.CONNECTIONS_URL] = [
        {"connections": [], "nextPageToken": "p2"},
        ec.ExportError(ec.EXIT_GOOGLE, "google", "Google failed (HTTP 500)"),
    ]
    _wire(monkeypatch, [None], [script])
    out = io.StringIO()
    monkeypatch.setattr(ec.sys, "stdout", out)
    code = ec.main([])
    assert code == ec.EXIT_GOOGLE and out.getvalue() == ""
    err = capsys.readouterr().err
    record = audit.read_text()
    for text in (err, record):
        assert REFRESH_FIXTURE not in text and "FIXTURE-ACCESS" not in text and SIGNING not in text
    assert json.loads(record.strip())["err"] == "google"


def test_a_people_api_403_is_google_failing_not_the_grant():
    g = _google_with([_Resp(403)])
    with pytest.raises(ec.ExportError) as err:
        g.get(ec.CONNECTIONS_URL)
    assert err.value.code == ec.EXIT_GOOGLE and g.slept == []
    with pytest.raises(ec.CredentialRejected):
        _google_with([_Resp(403)]).get(ec.USERINFO_URL)


def test_a_refusal_during_the_sessions_own_refresh_is_a_credential_refusal():
    from google.auth.exceptions import RefreshError

    with pytest.raises(ec.CredentialRejected):
        _google_with([RefreshError("invalid_grant")]).get(ec.CONNECTIONS_URL)


def test_an_unnamed_user_group_is_kept_as_unresolved():
    names = ec.group_name_map([
        {"resourceName": "contactGroups/u", "groupType": "USER_CONTACT_GROUP"},
        {"resourceName": "contactGroups/starred", "groupType": "SYSTEM_CONTACT_GROUP", "name": "starred"},
    ])
    person = {"memberships": [
        {"contactGroupMembership": {"contactGroupResourceName": "contactGroups/u"}},
        {"contactGroupMembership": {"contactGroupResourceName": "contactGroups/starred"}},
    ]}
    c = ec.shape_contact(person, names)
    assert c["groups"] == [] and c["groups_unresolved"] == ["contactGroups/u"]


def test_no_audit_log_configured_means_no_export(instance, monkeypatch):
    d, _ = instance
    asyncio.run(_put(str(d), _key(), _grant(_tid(1))))
    monkeypatch.delenv("WORKSPACE_MCP_AUDIT_LOG")
    with pytest.raises(ec.ExportError) as err:
        ec.run(False, out=io.StringIO())
    assert err.value.code == ec.EXIT_CONFIG and err.value.tag == "audit"


def test_a_failed_audit_write_prints_nothing(instance, monkeypatch):
    d, _ = instance
    asyncio.run(_put(str(d), _key(), _grant(_tid(1))))
    _wire(monkeypatch, [None], [_script_ok()])
    import core.audit_log as al

    monkeypatch.setattr(al.AuditLog, "write", lambda self, record: False)
    out = io.StringIO()
    with pytest.raises(ec.ExportError) as err:
        ec.run(False, out=out)
    assert err.value.tag == "audit" and out.getvalue() == ""


def test_an_unforeseen_error_is_audited_and_exits_4(instance, monkeypatch, capsys):
    d, audit = instance
    asyncio.run(_put(str(d), _key(), _grant(_tid(1))))
    _wire(monkeypatch, [None], [{ec.USERINFO_URL: [RuntimeError("boom " + REFRESH_FIXTURE)]}])
    assert ec.main([]) == ec.EXIT_CONFIG
    assert REFRESH_FIXTURE not in capsys.readouterr().err
    assert json.loads(audit.read_text().strip())["err"] == "internal"


def test_a_usage_error_exits_4_not_2(capsys):
    with pytest.raises(SystemExit) as stop:
        ec.main([""])
    assert stop.value.code == ec.EXIT_CONFIG


def test_check_mode_counts_without_calling_google(instance, monkeypatch):
    d, _ = instance
    asyncio.run(_put(str(d), _key(), _grant(_tid(1))))
    monkeypatch.setattr(ec, "refresh", lambda creds: (_ for _ in ()).throw(AssertionError("no Google in --check")))
    out = io.StringIO()
    assert ec.run(True, out=out) == ec.EXIT_OK
    assert json.loads(out.getvalue()) == {"skipped": {}, "usable_grants": 1}


def test_configuration_refusals(instance, monkeypatch):
    monkeypatch.setenv("WORKSPACE_MCP_ALLOWED_EMAILS", ALLOWED + ",second@example.com")
    with pytest.raises(ec.ExportError) as err:
        ec.run(True, out=io.StringIO())
    assert err.value.tag == "allowlist"
    monkeypatch.setenv("WORKSPACE_MCP_ALLOWED_EMAILS", ALLOWED)
    monkeypatch.setenv("WORKSPACE_MCP_OAUTH_PROXY_STORAGE_BACKEND", "valkey")
    with pytest.raises(ec.ExportError) as err:
        ec.run(True, out=io.StringIO())
    assert err.value.tag == "backend"
