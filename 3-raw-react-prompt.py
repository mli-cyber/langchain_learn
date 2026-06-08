import re
import inspect
from dotenv import load_dotenv
import os

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b" # "gpt-oss:20b" # "qwen3:1.7b"
OLLAMA_HOST = os.getenv("OLLAMA_HOST_URL") or os.getenv("OLLAMA_HOST")
ollama_client = ollama.Client(host=OLLAMA_HOST)


# --- Tools (LangChain @tool decorator) ---


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

def get_tool_description(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (e.g., @traceable adds *, config=None)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function)
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_description(tools)
tool_names = ", ".join(tools.keys())

# --- React Prompt ---

react_prompt = f"""
You are a ReAct agent that solves the user's question by reasoning step-by-step and using tools when needed.

You have access to the following tools:

{tool_descriptions}

Tool names:
[{tool_names}]

General rules:
1. Use tools when the answer depends on information or computation that a tool can provide.
2. Do not guess values that can be obtained from a tool.
3. Before calling a tool, make sure you have all required inputs for that tool.
4. If a required input is missing, first use another tool or ask the user for the missing information.
5. After each Observation, update what you know.
6. Do not call the same tool with the same input again if the previous Observation was valid.
7. If an Observation gives you a value needed by another tool, use that observed value exactly.
8. Do not perform calculations yourself if there is a tool designed for that calculation.
9. When you have enough information to answer the original question, stop using tools and give the Final Answer.
10. Action must be exactly one of [{tool_names}].
11. Do not invent tool names.
12. Do not write your own Observation. Observations come only from tool results.

Use this exact format:

Question: the input question you must answer
Thought: reason about what information is needed and which tool should be used
Action: the action to take, one of [{tool_names}]
Action Input: the input to the action, as comma-separated values
Observation: the result of the action
... this Thought/Action/Action Input/Observation sequence may repeat
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {{question}}
Thought:
"""


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama_client.chat(model=model, messages=messages, options=options)

# --- Agent Loop ---


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):


    print(f"Question: {question}")
    print("=" * 60)

    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        full_prompt = prompt + scratchpad

        # Stop token prevents the llm from Generating its own observations
        # we inject real tool instead.        
        response = ollama_chat_traced(
            model=MODEL, 
            messages=[{"role": "user", "content": full_prompt}], 
            options={"stop": ["\nObservation:"], "temperature": 0.0}
        )

        content = getattr(response.message, "content", "") or ""
        thinking = getattr(response.message, "thinking", "") or ""

        output = content.strip() or thinking.strip()

        print(f"  [LLM Output] {output}")

        print(f"    [Parsing] Looking for Final Answer in LLM Output")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"    [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer
        
        print(f"    [Parsing] Looking for Action and Action Input in LLM Output")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "   [Parsing] ERROR:Could not find Action and Action Input in LLM Output"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

         # Split comma-separated args; strip key= prefix if LLM outputs key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))
        
        print(f"  [Tool Result] {observation}")

        # CHANGE 7: History is one growing string re-sent every iteration (replaces messages.append).
        scratchpad += f"{output}\nObservation: {observation}\nThought:"


    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")