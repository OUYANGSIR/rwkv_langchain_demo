from llms.llm_rwkv import rwkv_chat_complations

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage, tool

system_msg = SystemMessage(
    "You are Nexora, a AI assistant"
)

history_message: list[BaseMessage] = [
    system_msg
]

while True:
    input_text = input(">")

    if input_text == '':
        continue
    if input_text == "exit":
        break

    current_user_msg = HumanMessage(
        input_text.strip()
    )

    history_message.append(current_user_msg)

    try:
        chunk = []
        print("AI >",end='')
        for result in rwkv_chat_complations.stream(history_message):
            current_ai_msg = result.content_blocks
            if len(current_ai_msg) == 0:
                break
            text = current_ai_msg[0].get('text')
            chunk.append(text)
            print(text, flush=True, end='')
        current_ai_msg = AIMessage("".join(chunk).strip())
        history_message.append(current_ai_msg)
    except Exception as e:
        raise e

