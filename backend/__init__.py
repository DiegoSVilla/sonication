"""minimalVoice backend package."""

# Corporate networks here terminate TLS with an internal root CA. truststore
# makes Python's ssl use the OS trust store (where that root CA is installed),
# so httpx trusts the intercepted chain without disabling verification.
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - fall back to certifi if unavailable
    pass
