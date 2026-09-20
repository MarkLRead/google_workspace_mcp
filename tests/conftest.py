"""Suite-wide fixtures for this fork.

The fork denies every identity unless WORKSPACE_MCP_ALLOWED_EMAILS admits it,
and refuses authentication modes the allowlist does not cover. Upstream's tests
predate that and exercise those paths with arbitrary addresses, so by default
the gate's two entry points are neutralised here. Tests of the gate itself
request the ``real_allowlist`` fixture to get the production behaviour.
"""

import pytest


# Upstream tests that reload core.server (which undoes the patch below) to boot
# an authentication mode this fork refuses to start.
_REFUSED_MODE_TESTS = {
    "test_external_oauth_metadata_matches_mcp_resource_and_challenge",
}


def pytest_collection_modifyitems(items):
    skip = pytest.mark.skip(reason="EXTERNAL_OAUTH21_PROVIDER is refused in this fork")
    for item in items:
        if item.name in _REFUSED_MODE_TESTS:
            item.add_marker(skip)


@pytest.fixture
def real_allowlist():
    """Opt out of the suite-wide neutralisation below."""


@pytest.fixture(autouse=True)
def _upstream_tests_run_ungated(request, monkeypatch):
    if "real_allowlist" in request.fixturenames:
        return
    monkeypatch.setattr(
        "auth.google_auth.enforce_allowed_email", lambda email, allowed: None
    )
    monkeypatch.setattr("core.server._refuse_ungated_external_provider", lambda: None)
