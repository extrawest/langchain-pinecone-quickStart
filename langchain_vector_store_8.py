from langchain_pinecone import PineconeVectorStore

from vector_store_6 import pc, embed

text_field = "text"

# switch back to normal index for langchain
index = pc.Index("quickstart-openai")

vectorstore = PineconeVectorStore(index, embed.embed_query, text_field)

query = "who was Benito Mussolini?"

search_result = vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
)

print(search_result)
