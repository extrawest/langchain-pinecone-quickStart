import tiktoken
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter


def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


data = load_dataset("wikipedia", "20220301.simple", split='train[:10000]', trust_remote_code=True)
print(data[6])

tokenizer = tiktoken.get_encoding('p50k_base')

tokens_len = tiktoken_len("hello I am a chunk of text and using the tiktoken_len function "
                          "we can find the length of this chunk of text in tokens")
print(tokens_len)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

if __name__ == '__main__':
    chunks = text_splitter.split_text(data[6]['text'])[:3]
    print(chunks)
    chunks_len = tiktoken_len(chunks[0]), tiktoken_len(chunks[1]), tiktoken_len(chunks[2])
    print(chunks_len)
