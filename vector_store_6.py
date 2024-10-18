import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

model_name = 'text-embedding-3-small'
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

pc = Pinecone(api_key=PINECONE_API_KEY)


def create_index(index_name, dimension):
    pc.create_index(
        name=index_name,
        dimension=len(dimension),  # Your model dimensions
        metric='dotproduct',  # Your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


if __name__ == '__main__':
    texts = [
        'this is the first chunk of text',
        'then another second chunk of text is here'
    ]

    res = embed.embed_documents(texts)
    print(len(res))
    print(len(res[0]))
    index_name = "quickstart-openai"

    create_index(index_name=index_name, dimension=res[0])
    print(pc.list_indexes())

    print(pc.describe_index(index_name))
