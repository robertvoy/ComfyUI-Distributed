"""Shared classification for credential-bearing configuration fields."""


def is_secret_field_name(key: object) -> bool:
    """Return whether a field name identifies an API key or credential."""
    if not isinstance(key, str):
        return False
    normalized = "".join(character for character in key.casefold() if character.isalnum())
    return normalized in {"apikey", "clearapikey"} or "credential" in normalized