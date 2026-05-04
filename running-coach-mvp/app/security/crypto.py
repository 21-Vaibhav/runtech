"""Cryptographic helpers for token encryption at rest."""

from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

from app.config import settings


def _fernet() -> Fernet:
    """Build Fernet cipher from configured key."""

    key = settings.token_encryption_key
    if not key:
        raise ValueError("TOKEN_ENCRYPTION_KEY is required")
    return Fernet(key.encode("utf-8"))


def encrypt_secret(value: str) -> str:
    """Encrypt sensitive text before persisting in DB."""

    if not value:
        return ""
    return _fernet().encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_secret(value: str) -> str:
    """Decrypt sensitive text from DB with backward-compatible fallback.

    If the value is not valid Fernet ciphertext (legacy plaintext rows),
    this function returns the raw value so old data keeps working.
    """

    if not value:
        return ""
    try:
        return _fernet().decrypt(value.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        return value

