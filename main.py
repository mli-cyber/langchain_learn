from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching the web for {query}")
    results = tavily_client.search(query)
    return results

# llm = ChatOpenAI(model="gpt-5-mini")
llm = ChatOllama(model="gpt-oss:20b", temperature=0.2)
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    # print("Hello, World!")
    response = agent.invoke({"messages": HumanMessage(content="list 5 jobs posting on linkedin for a ai engineer agentic ai applications in Colorado Springs")})
    print(response)
if __name__ == "__main__":
    main()
