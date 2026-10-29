"""Load and validate project configuration from the .env file."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
ENV_PATH = BASE_DIR / ".env"


class ConfigError(RuntimeError):
    """Raised when a required setting is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(repr=False)  # never print the key by accident
    openai_model: str


def mask_secret(secret: str) -> str:
    """Show only the first 3 and last 4 characters of a secret."""
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:3]}...{secret[-4:]}"


def load_settings() -> Settings:
    if not ENV_PATH.exists():
        raise ConfigError(f".env file not found at: {ENV_PATH}")

    load_dotenv(ENV_PATH)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ConfigError("OPENAI_API_KEY is missing or empty in .env")
    if api_key == "your-key-here":
        raise ConfigError("OPENAI_API_KEY still has the placeholder value")

    model = os.getenv("OPENAI_MODEL", "").strip()
    if not model:
        raise ConfigError("OPENAI_MODEL is missing or empty in .env")

    return Settings(openai_api_key=api_key, openai_model=model)  # <-- same level as the lines above