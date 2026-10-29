"""Print the model IDs available to this API key."""
from dotenv import load_dotenv
from openai import OpenAI

from config import ENV_PATH

load_dotenv(ENV_PATH)
client = OpenAI()

for model in sorted(client.models.list(), key=lambda m: m.id):
    print(model.id)