import os, dotenv
from langchain_openai import OpenAI, ChatOpenAI
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
dotenv.load_dotenv(PROJECT_ROOT / ".env")
RWKV_BASE_URL = os.getenv("RWKV_BASE_URL")
RWKV_API_KEY = os.getenv("RWKV_API_KEY")
RWKV_MODEL = os.getenv("RWKV_MODEL")

rwkv_complations = OpenAI(
    base_url=RWKV_BASE_URL,
    api_key=RWKV_API_KEY,
    model=RWKV_MODEL,
    max_tokens=10240,
    temperature=1,
    top_p=0.3,
    presence_penalty=0.0,
    frequency_penalty=0.01,
    model_kwargs={"stop": ["\n\nUser:", "</tool_call>\n"]},
)

rwkv_chat_complations = ChatOpenAI(
    base_url=RWKV_BASE_URL,
    api_key=RWKV_API_KEY,
    model=RWKV_MODEL,
    max_tokens=10240,
    temperature=1,
    top_p=0.1,
    presence_penalty=0.0,
    frequency_penalty=0.01,
)