import os, re, json

from langchain_core.prompts import PromptTemplate

from llms.llm_rwkv import rwkv_complations

system_prompt = PromptTemplate.from_template("System: {system_prompt}\n\n")

user_prompt = PromptTemplate.from_template("User: ```html\n{html_text}```\n{user_prompt}\n\n")

assistant_prompt = PromptTemplate.from_template("{assistant_prompt}")

chain = system_prompt + user_prompt + assistant_prompt | rwkv_complations

html_file = os.path.join(os.path.dirname(__file__), "index.html")


def read_html() -> str:
    with open(html_file, "r", encoding='utf-8') as f:
        html = f.read()
    return re.escape(html)

def write_html(html: str):
    with open(html_file, "w", encoding='utf-8') as f:
        f.write(html)

if not os.path.exists(html_file):
    write_html("")

while True:
    user_input = input("> ")
    if user_input == "exit":
        break
    if user_input == "":
        continue

    chunk = ["```json\n"]
    print("Think ...")
    for data in chain.stream(
        {
            "system_prompt": "你是一个html画板，根据用户输入决定是否修改html的内容，对于特殊字符需要适配json转义\n输出json:\n{\"html\": \"html code\",\n\"answer\":\"answer to user\"}",
            "user_prompt": user_input,
            "html_text": read_html(),
            "assistant_prompt": "Assistant: <think></think>```json",
        },
        config={
            "temperature": 0,
            "top_p": 0.3
        }
    ):
        print(data, end="")
        chunk.append(data)

    json_result = "".join(chunk).strip()

    pattern = r"```json(.*?)```"
    matchs = re.findall(pattern, json_result, re.DOTALL)
    json_str = json.loads(matchs[0].strip())

    write_html(json_str['html'])

    print(f"AI > {json_str['answer'].strip()}")