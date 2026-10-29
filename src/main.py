"""Smoke test: confirm the configuration loads and the API key is accepted."""
import openai
from openai import OpenAI

from config import ConfigError, load_settings, mask_secret


def main() -> None:
    try:
        settings = load_settings()
    except ConfigError as e:
        print(f"[CONFIG ERROR] {e}")
        return

    print(f"[OK] Config loaded. API key: {mask_secret(settings.openai_api_key)}")

    # Optional: verify the key with the API (lists models, does not generate text)
    client = OpenAI(api_key=settings.openai_api_key)
    try:
        client.models.list()
        print("[OK] API key accepted by OpenAI.")
    except openai.AuthenticationError:
        print("[API ERROR] Invalid API key.")
    except openai.OpenAIError as e:
        print(f"[API ERROR] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()