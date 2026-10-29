"""Reusable chatbot logic: system prompt, history, API call, friendly errors."""
import openai
from openai import OpenAI

DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly study assistant for AI engineering students.\n"
    "Rules:\n"
    "- Reply in the same language the user writes in.\n"
    "- Be concise: short paragraphs, bullet points for lists.\n"
    "- Put Python code in code blocks.\n"
    "- If you are not sure about something, say so instead of guessing.\n"
)


class ChatError(Exception):
    """Raised by ChatBot.send() with a user-friendly message."""


def describe_error(e: Exception) -> str:
    """Translate an OpenAI SDK exception into a clear, actionable message."""
    if isinstance(e, openai.AuthenticationError):
        return "Invalid or missing API key. Check OPENAI_API_KEY in .env."
    if isinstance(e, openai.NotFoundError):
        return "Model not found. Check OPENAI_MODEL in .env (see list_models.py or `ollama list`)."
    if isinstance(e, openai.RateLimitError):
        if e.type == "insufficient_quota":
            return "No credits left on the API account. Add credits in Billing; retrying won't help."
        return "Too many requests. Wait a few seconds and try again."
    if isinstance(e, openai.BadRequestError):
        return f"The request was rejected: {e.message}"
    if isinstance(e, openai.APITimeoutError):   # must come before APIConnectionError (it is a subclass)
        return "The request timed out. Try again, or use a smaller/faster model."
    if isinstance(e, openai.APIConnectionError):
        return ("Cannot reach the server. If you use Ollama, make sure it is running "
                "and OPENAI_BASE_URL is correct.")
    if isinstance(e, openai.APIStatusError):
        return f"Server error ({e.status_code}): {e.message}"
    return f"Unexpected error: {type(e).__name__}: {e}"


class ChatBot:
    def __init__(self, client: OpenAI, model: str,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT, max_turns: int = 10):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.max_turns = max_turns          # how many user/assistant pairs to remember
        self.history: list[dict] = []
        self.last_usage = None
        self.clear_conversation()

    def clear_conversation(self) -> None:
        """Forget the conversation but keep the system message."""
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.last_usage = None

    def send(self, user_text: str) -> str:
        """One turn: add user message, call the API, store and return the answer."""
        self.history.append({"role": "user", "content": user_text})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
            )
        except openai.OpenAIError as e:
            self.history.pop()              # keep history consistent: no unanswered user message
            raise ChatError(describe_error(e)) from e

        answer = response.choices[0].message.content or ""
        self.history.append({"role": "assistant", "content": answer})
        self.last_usage = response.usage
        self.prune_history()
        return answer

    def prune_history(self) -> None:
        """Sliding window: keep the system message + the last max_turns pairs."""
        keep = self.max_turns * 2
        body = self.history[1:]
        if len(body) > keep:
            self.history = [self.history[0]] + body[-keep:]