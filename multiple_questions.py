import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate  # Corrected import
from langchain_huggingface import HuggingFaceEndpoint  # Updated import

# Load environment variables
load_dotenv()

HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define the prompt template
multi_template = """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""
prompt = PromptTemplate(template=multi_template, input_variables=['questions'])

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
llm_chain = prompt | hub_llm  # Cleaner way to chain Runnables

# Define questions
questions_str = (
    "Which NFL team won the Super Bowl in the 2010 season?\n" +
    "If I am 6 ft 4 inches, how tall am I in centimeters?\n" +
    "Who was the 12th person on the moon?" +
    "How many eyes does a blade of grass have?"
)


# Run the chain and print the result
result = llm_chain.invoke({"questions": questions_str})
print(result)
