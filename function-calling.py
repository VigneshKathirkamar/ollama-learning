import os
import json
import asyncio
import random
from ollama import AsyncClient

model = "qwen2.5:latest"

# load the grocery list from a text file
def load_grocery_list(file_path):
    if not os.path.exists(file_path):
        print(f"{file_path} doesn't exist")
        return []
    with open(file_path,"r") as file:
        items = [ line.strip() for line in file if line.strip()]
    return items

async def fetch_price_and_nutrition(item):
    print(f"Fetching price and nutrition data for {item}")
    await asyncio.sleep(0.1) # To simulate network delay
    return {
        "item":item,
        "price":f"${random.uniform(1,10):.2f}",
        "calories":f"{random.randint(50,500)} kcal",
        "fat":f"{random.randint(1,20)} g",
        "protein": f"{random.randint(1,30)} g",
    }

async def fetch_recipe(category):
    print(f"fetching recipe for the item'{category}' category ")
    await asyncio.sleep(0.1)
    return {
        "category":category,
        "recipe":f"Sample recipe for {category} category",
        "instructions":"Mix and cook for 20 minutes",
    }

# Map the tool names the model can call to the actual python functions
available_functions = {
    "fetch_price_and_nutrition": fetch_price_and_nutrition,
    "fetch_recipe": fetch_recipe,
}

async def categorize_items(client, grocery_items):
    # Ask the model to group the items, and return JSON so we can parse it
    prompt = f"""
    Categorize the following grocery items into categories like Produce, Dairy, Meat, Bakery, Pantry etc.
    Return only JSON where each key is a category and each value is a list of items in that category.
    Items: {", ".join(grocery_items)}
    """
    response = await client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    return json.loads(response.message.content)

async def main():
    grocery_items = load_grocery_list("data/grocery_list.txt")
    if not grocery_items:
        print("Grocery items not found")
        return

    # Initialize the ollama client
    client = AsyncClient()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "fetch_price_and_nutrition",
                "description": "Fetches the price and nutrition data for a grocery item",
                "parameters": {
                    "type": "object",
                    "properties":{
                        "item":{
                            "type": "string",
                            "description": "The name of the grocery item",
                        },
                    },
                    "required": ["item"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_recipe",
                "description": "Fetches a recipe for a grocery category",
                "parameters": {
                    "type": "object",
                    "properties":{
                        "category":{
                            "type": "string",
                            "description": "The category of grocery items, e.g. Produce or Dairy",
                        },
                    },
                    "required": ["category"],
                },
            },
        },
    ]

    # Step 1: Categorize the grocery items
    categorize_prompt = f"""
    You are an assistant that categorizes grocery items.

    ** Instructions **
    - Return the result **only** as a valid JSON object.
    - Do **not** include explanations, greetings or additional texts
    - Use double quotes ("") for all strings
    - Ensure the JSON is properly formatted
    - The JSON should have categories as keys and list of items as values

    ** Example format **
    {{
        "Produce": ["Apples","Bannanas"],
        "Dairy": ["Butter","Milk"]
    }}

    ** Grocery items **
    {','.join(grocery_items)}

    """
    # categorized_items = await categorize_items(client, grocery_items)
    # print("Categorized items:")
    # print(json.dumps(categorized_items, indent=2))

    # # Step 2: Pick a random category and let the model decide which tools to call for it
    # category = random.choice(list(categorized_items.keys()))
    # category_items = categorized_items[category]
    # print(f"\nSelected category: {category}")

    messages = [
        {
            "role": "user",
            "content": categorize_prompt
        }
    ]
    response = await client.chat(model=model, messages=messages, tools=tools)
    messages.append(response["message"])
    print(response["message"]["content"])

    # Parse the model's response
    assistant_message = response["message"]["content"]
    try:
        categorized_items = json.loads(assistant_message)
        print("Categorized items")
        print(categorized_items)
    except json.JSONDecodeError:
        print("Failed to parse the model't response as JSON")
        print("Model's message")
        print(assistant_message)
        return


    # Step 2: Fetch price and nutrition data using function calling
    # Construct a message to instruct the model to fetch data for each item
    # We will ask the model to decide which items to fetch data for by using function calling
    fetch_prompt = f"""
    for each item in the grocery list, use the 'fetch_price_and_nutrition' function to get it's price and nutrition data.
    """
    messages.append({"role": "user", "content": fetch_prompt})

    # Second api call to the model with the updated messages and tools
    response = await client.chat(model=model, messages=messages, tools=tools)
    messages.append(response["message"])

    # Process function calls made by the model
    if response["message"].get("tool_calls"):
        print("Function calls made by the model: ")
        available_functions = {
            "fetch_price_and_nutrition": fetch_price_and_nutrition
        }
        item_details = []
        for tool_call in response["message"]["tool_calls"]:
            function_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            functions_to_call = available_functions.get(function_name)
            if functions_to_call:
                result = await functions_to_call(**arguments)
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result)
                })
                item_details.append(result)
                print(item_details)
            else:
                print(f"Function '{function_name}' is not available.")
                return
            
    # Step 3: Fetch recipe for a random category using function calling

    # choose a random category
    random_category = random.choice(list(categorized_items.keys()))
    recipe_prompt = f"""
    Fetch a recipe for the '{random_category}' category using the 'fetch_recipe' function.
    """
    messages.append({"role": "user", "content": recipe_prompt})

    # Third api call
    response = await client.chat(
        model= model,
        messages= messages,
        tools = tools,
    )

    messages.append(response["message"])

    if response["message"].get("tool_calls"):
            print("Function calls made by the model: ")
            available_functions = {
                "fetch_recipe": fetch_recipe
            }
            item_details = []
            for tool_call in response["message"]["tool_calls"]:
                function_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
                functions_to_call = available_functions.get(function_name)
                if functions_to_call:
                    result = await functions_to_call(**arguments)
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(result)
                    })
                    item_details.append(result)
                    print(item_details)
                else:
                    print(f"Function '{function_name}' is not available.")
                    return
                
            final_response = await client.chat(
                model= model,
                messages= messages,
                tools= tools
            )

            print("\n Assistant's final response:")
            print(final_response["message"]["content"])

if __name__ == "__main__":
    asyncio.run(main())
