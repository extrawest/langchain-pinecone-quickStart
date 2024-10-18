import os

from dotenv import load_dotenv
from langchain import FewShotPromptTemplate
from langchain import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables
load_dotenv()

HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# create our examples
examples = [
    {
        "query": "How are you?",
        "answer": "I can't complain but sometimes I still do."
    }, {
        "query": "What time is it?",
        "answer": "It's time to get a watch."
    }
]

example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative  and funny responses to the users questions. Here are some
examples: 
"""

# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

# now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n"
)

hub_llm = HuggingFaceEndpoint(
    endpoint_url='https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B-Instruct',
    model_kwargs={
        "headers": {"Authorization": f"Bearer {HF_API_KEY}"},
        "stream": True,
    },
    temperature=1.0,
)

query = "What is the meaning of life?"

# Chain the prompt and LLM using RunnableSequence
llm_chain = few_shot_prompt_template | hub_llm

# Run the chain and print the result
result = llm_chain.invoke({"query": query})
print(result)
