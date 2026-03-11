from openai import OpenAI
 
client = OpenAI(
    base_url="http://127.0.0.1:8045/v1",
    api_key="sk-54e8acb6671340e59cb00eab1f5b447c"
)
 
response = client.chat.completions.create(
    model="claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Hello"}]
)
 
print(response.choices[0].message.content)