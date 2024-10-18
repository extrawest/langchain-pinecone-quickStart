import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define the prompt template
template = """Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=['question'])

# Initialize the HuggingFaceEndpoint LLM
hub_llm = HuggingFaceEndpoint(
    endpoint_url='https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B-Instruct',
    model_kwargs={
        "headers": {"Authorization": f"Bearer {HF_API_KEY}"},
        "stream": True,
    },
    temperature=0.5,
)

# Chain the prompt and LLM using RunnableSequence
llm_chain = prompt | hub_llm

# Define the user question
question = "Who is the president of France?"

# Run the chain and print the result
result = llm_chain.invoke({"question": question})
print(result)
