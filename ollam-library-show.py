import ollama

res = ollama.generate(
    model="qwen2.5:latest",
    messages=[
        {"role": "user", "content": "Tell me a fact about Mlops Engineering."},
    ]
)

# show the generated response
print(ollama.show("qwen2.5:latest"))
# print(ollama.show("qwen2.5:latest", res['id']))

# print(res)