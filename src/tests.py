"""Phase 2: test several prompts and handle common API errors."""
import openai
from openai import OpenAI

from config import ConfigError, load_settings

PROMPTS = [
    "What is overfitting? Answer in one sentence.",
    "ترجم إلى الإنجليزية: التعلم الآلي يغيّر العالم",
    "Write a Python function that reverses a string.",
]


def ask(client: OpenAI, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0,   # less randomness; supported by Ollama's OpenAI-compatible endpoint as far as I know
)
    return response.choices[0].message.content


def safe_ask(client: OpenAI, model: str, prompt: str) -> str:
    try:
        return ask(client, model, prompt)
    except openai.AuthenticationError:          # HTTP 401
        return "[401] Invalid, revoked, or missing API key."
    except openai.NotFoundError as e:           # HTTP 404
        return f"[404] Model not found or no access: '{model}' (code={e.code})"
    except openai.RateLimitError as e:          # HTTP 429
        if e.type == "insufficient_quota":
            return "[429] No credits left. Add credits in Billing, retrying won't help."
        retry_after = e.response.headers.get("retry-after")
        return f"[429] Rate limit (code={e.code}). Wait {retry_after or 'a few'} seconds and retry."
    except openai.APIConnectionError:
        return "[NETWORK] Could not reach the API. Check your internet."
    except openai.APIStatusError as e:          # any other HTTP error
        return f"[{e.status_code}] {e.message}"


def main() -> None:
    try:
        settings = load_settings()
    except ConfigError as e:
        print(f"[CONFIG ERROR] {e}")
        return

    client = OpenAI()

    print("=== Step 7: different prompts ===")
    for prompt in PROMPTS:
        print(f"\nQ: {prompt}\nA: {safe_ask(client, settings.openai_model, prompt)}")

    print("\n=== Step 8: provoke errors on purpose ===")
    print("Invalid model ->", safe_ask(client, "not-a-real-model", "hi"))
    bad_client = OpenAI(api_key="sk-this-is-wrong")
    print("Bad key       ->", safe_ask(bad_client, settings.openai_model, "hi"))


if __name__ == "__main__":
    main()