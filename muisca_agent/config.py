"""Settings, read from environment variables (or a .env file in the project root)."""

import os

from dotenv import load_dotenv

load_dotenv()

# Where the Laravel API lives. Local by default; production is https://api.muisca.ca
MUISCA_API_URL = os.getenv("MUISCA_API_URL", "http://localhost:8000").rstrip("/")

# Which Claude model answers. Sonnet is the balanced choice; claude-haiku-5-5 is cheaper.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

# Browser origins allowed to call POST /chat, comma-separated, e.g. "https://muisca.ca".
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]


def api_key_is_set() -> bool:
    """The Anthropic SDK reads the key from this variable. It is never stored in the code."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))
