import os

from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize the LLM
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    model_name="gpt-4o"
)

# Load the math tool
tools = load_tools(['llm-math'], llm=llm)

# Extract tool names and descriptions
tool_names = ", ".join([tool.name for tool in tools])
tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

react_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template=f"""
You are an AI that solves math problems step-by-step using the following tools:

{tool_descriptions}

Use the format:
Question: {{input}}

Thought: Think about what action is required.

Action: Choose a tool.

Action Input: Provide input for the chosen tool.

Observation: Receive the result.

{{agent_scratchpad}}

Final Answer: Provide the solution to the original question.
"""
)

# Create the ReAct agent
react_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt.partial(
        tools=tools,
        tool_names=tool_names
    ),
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=react_agent, tools=tools, verbose=True
)

result = agent_executor.invoke({"input": "What is (4.5 * 2.1)^2.2?"})
print(result)
