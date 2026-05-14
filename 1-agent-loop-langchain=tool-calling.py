from dotenv import load_dotenv

load_dotenv()

import os

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# --- Tools (langchain @tools decorator) ---

@tool
def get_product_price(product:str) -> float:
    """
    Lookup the price of a product in the catalog
    """
    print(f"\t >> executing tool get_product_price(product='{product}')")
    price = {"laptop": 1499.99, "mouse": 17.48, "keyboard": 20, "monitor": 300}
    return price.get(product, "Product not found")

@tool
def apply_discount(discount_tier:str, price:float) -> float:
    """
    Apply a discount tier to a price and return the final price.
    available discount tiers are: "gold", "silver", "bronze"
    """
    print(f"\t >> executing tool apply_discount(discount_tier='{discount_tier}', price='{price}')")
    discount_percentage = {"gold": 23, "silver": 12, "bronze": 5}
    discount = discount_percentage.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)
    
# --- Agent Loop ---

@traceable(name="LangChain Agent Loop")
def run_agent(question:str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0, base_url=os.getenv("OLLAMA_HOST_URL"))
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("="*60)

    messages = [
        SystemMessage(
            content=
            "You are a helpful shopping assistant."
            "You have access to a product catalog tool"
            "and a discount tool.\n\n"
            "STRICT RULES - You must to follow the rules exactly:\n"
            "1. NEVER guess or assume any product price."
            "You MUST call get_product_price first to get the real price.\n"
            "2. Only call apply_discount AFTER you have received"
            "a price from get_product_price. Pass the exact price"
            "retured by get_product_price - do NOT pass a made-up number.\n"
            "3. NEVER calculate discounts yourself using math."
            "Always use the apply_discount tool to get the final price.\n"
            "4. If the user does not specify a discount tier"
            "ask them which tier to use - do NOT assume one.\n"
            ),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n===[Iteration {iteration}]===\n")
        
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls
        
        # if no tool calls, we have the final answer
        if not tool_calls:
            print(f"\nFinal answer: {ai_message.content}")
            return ai_message.content
        
        # Process only the First Tool Call - force one tool per iteration for simplicity
        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print(f"\t [Tool Selected] {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict[tool_name]

        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"\t [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=observation, tool_call_id=tool_call_id)
        )
    
    print("ERROR: Max iterations reached without finding the final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    


