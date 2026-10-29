"""Offline tests for ChatBot: no network, no API key, no Ollama needed."""
from types import SimpleNamespace

import openai
import pytest

from chatbot import ChatBot, ChatError, DEFAULT_SYSTEM_PROMPT, describe_error
try:                      # openai >= 3.x ships its own fork of the HTTP library
    import httpx2 as httpx
except ImportError:       # openai 2.x uses the regular httpx
    import httpx

# ---------- helpers: a fake client that mimics client.chat.completions.create ----------

class FakeCompletions:
    def __init__(self, error=None):
        self.error = error
        self.calls = []  # one snapshot of `messages` per API call

    def create(self, model, messages):
        # Snapshot: the bot keeps mutating its list after the call returns.
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        text = f"reply {len(self.calls)}"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(prompt_tokens=len(messages)),
        )


class FakeClient:
    def __init__(self, error=None):
        self.completions = FakeCompletions(error)
        self.chat = SimpleNamespace(completions=self.completions)


def make_bot(max_turns=10, error=None, system_prompt=DEFAULT_SYSTEM_PROMPT):
    client = FakeClient(error)
    bot = ChatBot(client, "fake-model", system_prompt=system_prompt, max_turns=max_turns)
    return bot, client


def make_api_error(cls, status, body=None):
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return cls("boom", response=response, body=body)


# ---------- initial state ----------

def test_new_bot_starts_with_only_the_system_message():
    bot, _ = make_bot()
    assert bot.history == [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]


def test_custom_system_prompt_is_used():
    bot, _ = make_bot(system_prompt="You are a pirate.")
    assert bot.history[0] == {"role": "system", "content": "You are a pirate."}


# ---------- conversation memory ----------

def test_send_stores_messages_in_the_right_order():
    bot, _ = make_bot()
    bot.send("hello")
    bot.send("how are you?")
    roles = [m["role"] for m in bot.history]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    assert bot.history[1]["content"] == "hello"
    assert bot.history[3]["content"] == "how are you?"


def test_send_returns_the_model_answer_and_stores_it():
    bot, _ = make_bot()
    answer = bot.send("hi")
    assert answer == "reply 1"
    assert bot.history[-1] == {"role": "assistant", "content": "reply 1"}


def test_second_request_contains_the_first_turn():
    bot, client = make_bot()
    bot.send("my name is Toqa")
    bot.send("what is my name?")
    second_call = client.completions.calls[1]
    contents = [m["content"] for m in second_call]
    assert "my name is Toqa" in contents       # earlier user message is re-sent
    assert "reply 1" in contents               # earlier assistant reply is re-sent
    assert second_call[-1]["content"] == "what is my name?"


def test_none_content_is_stored_as_empty_string():
    bot, client = make_bot()
    original_create = client.completions.create

    def create_with_none(model, messages):
        response = original_create(model, messages)
        response.choices[0].message.content = None
        return response

    client.completions.create = create_with_none
    assert bot.send("hi") == ""
    assert bot.history[-1]["content"] == ""


# ---------- clear_conversation ----------

def test_clear_conversation_keeps_the_system_message():
    bot, _ = make_bot()
    bot.send("one")
    bot.send("two")
    bot.clear_conversation()
    assert bot.history == [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]


def test_clear_conversation_removes_previous_messages_and_usage():
    bot, _ = make_bot()
    bot.send("secret: 12345")
    assert bot.last_usage is not None
    bot.clear_conversation()
    assert all("12345" not in m["content"] for m in bot.history)
    assert len(bot.history) == 1
    assert bot.last_usage is None


def test_bot_forgets_after_clear_in_the_next_request():
    bot, client = make_bot()
    bot.send("my name is Toqa")
    bot.clear_conversation()
    bot.send("what is my name?")
    last_call = client.completions.calls[-1]
    assert [m["role"] for m in last_call] == ["system", "user"]


# ---------- history pruning ----------

def test_pruning_keeps_system_message_and_latest_messages():
    bot, _ = make_bot(max_turns=2)
    for i in range(1, 6):
        bot.send(f"q{i}")
    assert bot.history[0]["role"] == "system"
    assert [m["content"] for m in bot.history[1:]] == ["q4", "reply 4", "q5", "reply 5"]


def test_pruned_history_always_starts_with_a_user_message():
    bot, _ = make_bot(max_turns=2)
    for i in range(7):
        bot.send(f"q{i}")
        assert bot.history[1]["role"] == "user"


def test_no_pruning_below_the_limit():
    bot, _ = make_bot(max_turns=5)
    for i in range(3):
        bot.send(f"q{i}")
    assert len(bot.history) == 1 + 2 * 3


def test_pruning_bounds_the_size_of_every_api_request():
    max_turns = 3
    bot, client = make_bot(max_turns=max_turns)
    for i in range(20):
        bot.send(f"q{i}")
    # system + (max_turns kept pairs) + the new user message
    limit = 1 + 2 * max_turns + 1
    assert max(len(call) for call in client.completions.calls) <= limit


# ---------- errors ----------

def test_failed_request_raises_chat_error_and_keeps_history_clean():
    error = make_api_error(openai.AuthenticationError, 401)
    bot, _ = make_bot(error=error)
    with pytest.raises(ChatError) as info:
        bot.send("hello")
    assert "API key" in str(info.value)
    assert bot.history == [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]  # no dangling user message


@pytest.mark.parametrize(
    "cls, status, body, expected_text",
    [
        (openai.AuthenticationError, 401, None, "API key"),
        (openai.NotFoundError, 404, None, "Model not found"),
        (openai.RateLimitError, 429, {"type": "insufficient_quota", "code": "credit_balance_exhausted"}, "No credits"),
        (openai.RateLimitError, 429, {"type": "requests", "code": "rate_limit_exceeded"}, "Too many requests"),
    ],
)
def test_describe_error_gives_actionable_messages(cls, status, body, expected_text):
    error = make_api_error(cls, status, body)
    assert expected_text in describe_error(error)


def test_connection_error_mentions_ollama():
    request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
    error = openai.APIConnectionError(request=request)
    assert "Ollama" in describe_error(error)


def test_timeout_is_reported_as_timeout_not_connection_error():
    request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
    error = openai.APITimeoutError(request=request)
    assert "timed out" in describe_error(error)