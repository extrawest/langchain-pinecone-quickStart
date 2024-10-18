import os

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define the prompt template
template = """Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=['question'])

# Initialize the HuggingFaceEndpoint LLM
hf_llm = HuggingFaceEndpoint(
    endpoint_url='https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B-Instruct',
    model_kwargs={
        "headers": {"Authorization": f"Bearer {HF_API_KEY}"},
        "stream": True,
    },
    temperature=0.5,
)

store = {}
# Example session ID
session_id = "session_1"


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# Initialize RunnableWithMessageHistory
conversation = RunnableWithMessageHistory(
    runnable=hf_llm,
    get_session_history=get_session_history,
)

# Example usage: send a message
response = conversation.invoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
], config={"configurable": {"session_id": session_id}})

print(response)
ConversationBufferMemory()
