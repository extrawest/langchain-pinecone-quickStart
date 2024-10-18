from uuid import uuid4

from tqdm.auto import tqdm

from tokenizer_example_5 import data, text_splitter
from vector_store_6 import embed, pc


def create_embeddings(texts, metadatas):
    for i, record in enumerate(tqdm(data)):
        # first get metadata fields for this record
        metadata = {
            'wiki-id': str(record['id']),
            'source': record['url'],
            'title': record['title']
        }

        # now we create chunks from the record text
        record_texts = text_splitter.split_text(record['text'])

        # create individual metadata dicts for each chunk
        record_metadatas = [{"chunk": j, "text": text, **metadata} for j, text in enumerate(record_texts)]

        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)

        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas))
            texts = []
            metadatas = []


if __name__ == '__main__':
    batch_limit = 100
    texts = []
    metadatas = []

    index = pc.Index("quickstart-openai")

    create_embeddings(texts=texts, metadatas=metadatas)
    
    stats = index.describe_index_stats()
    print(stats)
