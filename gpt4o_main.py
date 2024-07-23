##https://platform.openai.com/docs/guides/text-generation

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv("./.env")

openai_api_key = str = os.getenv('OPENAI_API_KEY')

model = "gpt-4o"
# client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
client = OpenAI(api_key=openai_api_key)
# print(client)


completion = client.chat.completions.create(
    model=model,
    # response_format={ "type": "json_object" },
    messages=[
        # {"role":"system", "content": "You are an amazingly supportive assistant designed to output JSON. Please answer my questions"},
        {"role":"system", "content": "You are an amazingly supportive assistant. Please answer my questions"},
        {"role":"user", "content": "I want to know about dob studio. Please tell me about it"},
        {"role":"assistant", "content":"dob studio is a virtual avatar company."}
    ]
)

print("Asisstant: " + completion.choices[0].message.content)