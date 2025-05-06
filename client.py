from openai import OpenAI

client = OpenAI(
    api_key="sk-test",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="your-custom-model",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
print(response.choices[0].message.content)
