"""API key authentication via Authorization: Bearer <token>."""
import hmac
import secrets
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import settings

_bearer = HTTPBearer(auto_error=False)


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> str:
    """
    Validates the Bearer token using a constant-time comparison to prevent
    timing attacks. Raises HTTP 401 if the token is missing or incorrect.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Constant-time comparison prevents timing-based key enumeration
    token_valid = hmac.compare_digest(
        credentials.credentials.encode("utf-8"),
        settings.rag_api_key.encode("utf-8"),
    )

    if not token_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials
