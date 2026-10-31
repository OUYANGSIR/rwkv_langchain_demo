import time

from langchain_core.prompts import PromptTemplate
from common.stream_manager import StreamManager

from llms.llm_rwkv import rwkv_complations

from dataclasses import dataclass

@dataclass
class Config:
    temperature: float = 0.0
    top_p: float = 0.3
    max_tokens: int = 8000
    stream: bool = False

stream_manager = StreamManager()

def ccompletions_user_assistant(
        history: PromptTemplate,
        user_input: str,
        assistant_input: str,
        config: Config=None
):
    config = config or Config()
    global stream_manager
    user_assistant_template = PromptTemplate.from_template("""User: {user_input}\n\nAssistant: {assistant_input}""")

    chain = history + user_assistant_template | rwkv_complations.bind(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
    )

    tid = stream_manager.start(
        chain,
        {
            "user_input": user_input,
            "assistant_input": assistant_input
        }
    )

    try:
        print(f"Assistant: {assistant_input}", end='')

        if config.stream:
            current_count = 0
            while stream_manager.is_running(tid):
                currnet_text = stream_manager.text(tid)
                print(currnet_text[current_count:], end='')
                current_count = len(currnet_text)
                time.sleep(0.2)
            currnet_text = stream_manager.text(tid)
            print(currnet_text[current_count:])
        assistant_result = stream_manager.wait(tid)
        return assistant_result
    except KeyboardInterrupt:
        return stream_manager.stop(tid)
    finally:
        stream_manager.clear(tid)

def ccompletions_tool_user_assistant():
    pass