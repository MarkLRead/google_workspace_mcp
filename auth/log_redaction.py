"""
Address redaction for WARNING/ERROR log lines.

The server's log files are scanned for e-mail addresses (log hygiene). No
failure path may be the one that writes one. These helpers replace an address
with a stable, non-reversible stand-in: the first 12 hex characters of its
sha256, so two lines about the same account can still be correlated.
"""

import hashlib
import re
from typing import Any, Optional

# Same shape as the log-hygiene scanner: a local part, an @, a dotted domain.
_ADDRESS_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")


def redact_email(email: Optional[str]) -> str:
    """The 12-char sha256 prefix of an address; "none" for an empty one."""
    if not email:
        return "none"
    return hashlib.sha256(email.encode("utf-8")).hexdigest()[:12]


def redact_text(text: Any, email: Optional[str] = None) -> str:
    """
    Stringify a value for a log line with every address-shaped token replaced
    by a delimited marker `[email:<sha256 prefix>]`. `email` is accepted for
    call-site clarity; it is redacted like any other address the text may carry
    (a mismatched account, a token's address, an upstream message).

    The marker is delimited on purpose: a bare hash is a valid local part, so
    replacing "a@b.com" inside "a@b.com@c.com" with the hash alone would leave
    "<hash>@c.com", an address-shaped token the scanner would flag.
    """
    value = str(text)
    return _ADDRESS_RE.sub(lambda m: f"[email:{redact_email(m.group(0))}]", value)
