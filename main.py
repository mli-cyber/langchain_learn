# load the environment variables
from dotenv import load_dotenv
load_dotenv()

import os

from typing import List
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """ Schema for a source used by agent"""

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """ Schema for agent response with answer and souces"""

    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(default_factory=list, description="List of sources used to generate answers")


# llm = ChatOpenAI(model="gpt-5-mini")
llm = ChatOllama(model="gpt-oss:20b", base_url=os.getenv("OLLAMA_HOST_URL"), temperature=0.2)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    # print("Hello, World!")
    # response = agent.invoke({"messages": HumanMessage(content="list 5 jobs posting on linkedin for a ai engineer agentic ai applications in Colorado Springs")})
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content= (
                "List 5 LinkedIn job postings for AI Engineer roles focused on "
                "agentic AI applications in the Colorado Springs area. "
                "For each posting, include the job title, company, location, brief summary, "
                "required skills, and link if available."
                )
            )
        }
    )
    # result = agent.invoke({"messages": HumanMessage(content="give me the lastest 3 news on CNN?")})

    print(result)
if __name__ == "__main__":
    main()
