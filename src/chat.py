"""Phase 4: improved terminal chatbot."""
from openai import OpenAI

from chatbot import ChatBot, ChatError
from config import ConfigError, load_settings

HELP = """Commands:
  /help     show this help
  /history  show stored messages
  /clear    forget the conversation (keeps the system prompt)
  /exit     quit"""


def print_history(bot: ChatBot) -> None:
    print("\n--- conversation history ---")
    for i, msg in enumerate(bot.history):
        text = msg["content"].replace("\n", " ")
        text = text[:70] + "..." if len(text) > 70 else text
        print(f"[{i}] {msg['role']:<9} {text}")
    print(f"--- {len(bot.history)} messages (max {bot.max_turns} turns kept) ---")
    if bot.last_usage:
        print(f"last request: {bot.last_usage.prompt_tokens} prompt tokens")
    print()


def main() -> None:
    try:
        settings = load_settings()
    except ConfigError as e:
        print(f"[CONFIG ERROR] {e}")
        return

    client = OpenAI(timeout=120.0, max_retries=2)
    bot = ChatBot(client, settings.openai_model)
    print(f"Chatbot ready (model: {settings.openai_model}).\n{HELP}\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if user_text.startswith("/"):
            command = user_text.lower()
            if command == "/exit":
                print("Bye!")
                break
            elif command == "/help":
                print(HELP)
            elif command == "/history":
                print_history(bot)
            elif command == "/clear":
                bot.clear_conversation()
                print("Memory cleared.\n")
            else:
                print(f"Unknown command: {user_text}. Type /help.\n")
            continue

        try:
            print(f"Bot: {bot.send(user_text)}\n")
        except ChatError as e:
            print(f"[ERROR] {e}\n")


if __name__ == "__main__":
    main()