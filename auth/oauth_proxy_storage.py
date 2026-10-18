"""Keys and the disk store of FastMCP's OAuth proxy, in one place.

The server (``core/server.py``) and offline commands such as
``core/export_contacts.py`` must open the proxy's storage the same way, or a
command run beside the server would read nothing (a different key) or the
wrong directory. Both call these functions; nothing here writes to the store.
"""

import logging
import os
from typing import Optional

from cryptography.fernet import Fernet
from fastmcp.server.auth.jwt_issuer import derive_jwt_key
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper

from core.storage import make_sanitized_file_store

logger = logging.getLogger(__name__)

JWT_SIGNING_KEY_ENV = "FASTMCP_SERVER_AUTH_GOOGLE_JWT_SIGNING_KEY"
DISK_DIRECTORY_ENV = "WORKSPACE_MCP_OAUTH_PROXY_DISK_DIRECTORY"
STORAGE_BACKEND_ENV = "WORKSPACE_MCP_OAUTH_PROXY_STORAGE_BACKEND"

_JWT_SALT = "fastmcp-jwt-signing-key"
_STORAGE_SALT = "fastmcp-storage-encryption-key"


def jwt_signing_key_override() -> Optional[str]:
    """The signing key from the environment, or None when unset or blank."""
    return os.getenv(JWT_SIGNING_KEY_ENV, "").strip() or None


def validate_and_derive_jwt_key(
    jwt_signing_key_override: Optional[str], client_secret: Optional[str]
) -> bytes:
    """Validate JWT signing key override and derive the final JWT key."""
    if jwt_signing_key_override:
        if len(jwt_signing_key_override) < 12:
            logger.warning(
                "OAuth 2.1: FASTMCP_SERVER_AUTH_GOOGLE_JWT_SIGNING_KEY is less than 12 characters; "
                "use a longer secret to improve key derivation strength."
            )
        return derive_jwt_key(
            low_entropy_material=jwt_signing_key_override,
            salt=_JWT_SALT,
        )
    if client_secret:
        return derive_jwt_key(
            high_entropy_material=client_secret,
            salt=_JWT_SALT,
        )
    raise ValueError(
        "Public client OAuth 2.1 requires FASTMCP_SERVER_AUTH_GOOGLE_JWT_SIGNING_KEY "
        "when GOOGLE_OAUTH_CLIENT_SECRET is not set."
    )


def derive_storage_encryption_key(jwt_signing_key: bytes) -> bytes:
    """The Fernet key that encrypts every value in the proxy's storage."""
    return derive_jwt_key(
        high_entropy_material=jwt_signing_key.decode(),
        salt=_STORAGE_SALT,
    )


def disk_directory() -> str:
    """Where the disk backend keeps its files (env, else FASTMCP_HOME, else ~)."""
    directory = os.getenv(DISK_DIRECTORY_ENV, "").strip()
    if directory:
        return directory
    fastmcp_home = os.getenv("FASTMCP_HOME", "").strip()
    if fastmcp_home:
        return os.path.join(fastmcp_home, "oauth-proxy")
    return os.path.expanduser("~/.fastmcp/oauth-proxy")


def encrypted_disk_store(directory: str, jwt_signing_key: bytes) -> FernetEncryptionWrapper:
    """The proxy's disk store, wrapped in the same encryption the server uses."""
    return FernetEncryptionWrapper(
        key_value=make_sanitized_file_store(directory),
        fernet=Fernet(key=derive_storage_encryption_key(jwt_signing_key)),
    )
