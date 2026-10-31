import re
import json

pattern = r"<tool_call>(.*?)</tool_call>"

def handler(context):
    matches = re.findall(pattern, context, re.DOTALL)
    print(matches)
    if len(matches):
        tool_data = json.loads(matches)
        function_name = tool_data['function']['name']