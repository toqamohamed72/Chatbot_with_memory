"""Phase 2: first chat completion request."""
from openai import OpenAI

from config import ConfigError, load_settings


def main() -> None:
    try:
        settings = load_settings()  # also loads .env into os.environ
    except ConfigError as e:
        print(f"[CONFIG ERROR] {e}")
        return

    client = OpenAI()  # Step 1: reads OPENAI_API_KEY from the environment

    messages = [  # Step 2: model + user message
        {"role": "user", "content": "Explain what an API key is in two sentences."}
    ]

    response = client.chat.completions.create(  # Steps 3-4: send, receive
        model=settings.openai_model,
        messages=messages,
    )

    answer = response.choices[0].message.content  # Step 5: extract the answer
    print(answer)  # Step 6: display

    print("\n--- metadata ---")
    print("model:", response.model)
    print("finish_reason:", response.choices[0].finish_reason)
    print("tokens (prompt + completion = total):",
          response.usage.prompt_tokens, "+", response.usage.completion_tokens,
          "=", response.usage.total_tokens)


if __name__ == "__main__":
    main()