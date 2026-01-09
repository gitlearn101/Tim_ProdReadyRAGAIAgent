from openai import OpenAI
import google.genai as genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os

load_dotenv()

#client = OpenAI()
#EMBED_MODEL = "text-embedding-3-large"

api_key = os.getenv("GOOGLE_API_KEY")
genai_client = genai.client.Client(api_key=api_key)
EMBED_MODEL = "models/embedding-001"

EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

"""""
def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model = EMBED_MODEL,
        input = texts,
    )
    return [item.embedding for item in response.data]
 """   
def embed_texts(texts: list[str]) -> list[list[float]]:
    result = genai_client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
    )
    return [item for item in result["embedding"]]