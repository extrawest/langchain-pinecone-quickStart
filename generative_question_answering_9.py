import os

from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

text_field = "text"

model_name = 'text-embedding-3-small'

# Create embeddings
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

# Initialize Pinecone client and index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("quickstart-openai")

# Create PineconeVectorStore with correct embed_query method
vectorstore = PineconeVectorStore(index=index, embedding=embed, text_key=text_field)

# Create OpenAI LLM
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4o',
    temperature=0.0
)

# Initialize the chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Query
query = "who was Benito Mussolini?"

# Run the chain using invoke (run is deprecated)
response = chain.invoke({"query": query})
answer, docs = response['result'], response['source_documents']
print(answer)
print(docs)
