"""Generate cryptographically secure random keys for the .env file."""
import secrets
import sys
import os


def generate_key(length: int = 32) -> str:
    """Returns a URL-safe base64 encoded random key of `length` bytes."""
    return secrets.token_urlsafe(length)


if __name__ == "__main__":
    print("# Auto-generated secrets — keep this file private (chmod 600)")
    print(f"RAG_API_KEY={generate_key(32)}")
    print(f"CHROMA_TOKEN={generate_key(32)}")
    print(f"WEBUI_SECRET_KEY={generate_key(32)}")
