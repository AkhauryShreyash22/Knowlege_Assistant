from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from drf_spectacular.utils import extend_schema, OpenApiRequest

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from functools import lru_cache
import hashlib


rag_llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=300
)

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

model = SentenceTransformer("all-MiniLM-L6-v2")


def make_cache_key(question: str):
    return hashlib.md5(question.strip().lower().encode()).hexdigest()


@lru_cache(maxsize=1000)
def cached_answer(question: str):

    query_embedding = model.encode([question]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_chunks = results.get("documents", [[]])[0]
    retrieved_metadata = results.get("metadatas", [[]])[0]

    if not retrieved_chunks:
        return {
            "answer": "Sorry, no relevant information found in the knowledge base.",
            "sources": []
        }

    sources_list = []
    for meta in retrieved_metadata:
        safe_meta = meta or {}
        doc = safe_meta.get("source", "Unknown.pdf")
        page = safe_meta.get("page", "N/A")
        sources_list.append(f"{doc} - Page {page}")

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
Only answer using the information given in CONTEXT.
If the answer is not found in the context, reply:
"Sorry, no relevant information found in the knowledge base."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    hf_response = rag_llm(prompt)
    answer = hf_response[0]["generated_text"].strip()

    question_keywords = [w.lower() for w in question.split() if len(w) > 4]
    context_lower = context.lower()
    relevant = any(k in context_lower for k in question_keywords)

    if not relevant:
        return {
            "answer": "Sorry, no relevant information found in the knowledge base.",
            "sources": sources_list
        }

    return {
        "answer": answer,
        "sources": sources_list
    }


class AskQuestionAPIView(APIView):

    @extend_schema(
        summary="Ask a question from your indexed knowledge base",
        request=OpenApiRequest(
            {
                "type": "object",
                "properties": {
                    "question": {"type": "string"}
                },
                "example": {"question": "What is the use of mitochondria?"}
            }
        ),
        responses={200: None}
    )
    def post(self, request):

        question = request.data.get("question")

        if not question:
            return Response({"error": "Question is required"}, status=400)

        response_data = cached_answer(question)

        return Response(response_data)
