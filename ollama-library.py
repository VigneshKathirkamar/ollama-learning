import ollama

# model_list = ollama.list()

# print("Available Models:", model_list)

res = ollama.chat(
    model="qwen2.5:latest",
    messages=[
        {"role": "user", "content": "Tell me a fact about Mlops Engineering."},
    ],
    stream=True
)
# print(res['message']['content'])

for chunk in res:
    print(chunk['message']['content'], end="", flush=True)