
import ollama

# ollama.create(
#     model="linkedin-growth-expert",
#     from_="qwen2.5:latest",
#     system="You are a linkedin growth expert. You have to suggest strategies and posts to user that will increase his \
#         followers in Linkedin.",
#     parameters={"temperature": 0.7},
# )

res = ollama.generate(model="linkedin-growth-expert",prompt="How to increase my linkedin followers")
print(res["response"])