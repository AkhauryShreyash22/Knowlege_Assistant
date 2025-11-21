from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

from django.conf import settings

embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

chunk_embeddings = embedder.encode(chunks)

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(chunk_embeddings).astype("float32"))

def search_similar(question, top_k=3):
    q_emb = embedder.encode([question]).astype("float32")
    distances, indices = index.search(q_emb, top_k)
    retrieved = [chunks[i] for i in indices[0]]
    return retrieved
