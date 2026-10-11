import ollama
import os

model = "qwen2.5:latest"

input_file = "./data/grocery_list.txt"
output_file = "./data/categorized_grocery_list.txt"

with open(input_file, "r") as f:
    grocery_list = f.read().strip()

PROMPT = f"""
Please organize the grocery into categories. The grocery list is as follows:
{grocery_list}
Please:
1. Categorize them into Veg, Non-Veg, Dairy, Bakery, Beverages, Snacks etc.
2. Arrange them in a alphabetical order within each category.
3. Present it in a clear and structured format, with each category as a heading followed by the items in that category.
"""

response = ollama.generate(
    model=model,
    prompt=PROMPT,
)
generated_response = response.get("response", "")
# print("The generated response is: ", response)

try:
    with open(output_file,"w") as f:
        f.write("Categorized Grocery List:\n\n")
        f.write(generated_response.strip())
    print(f"Categorized grocery list has been written to {output_file}")
except Exception as e:
    print(f"An error occurred while writing to the file: {e}")

