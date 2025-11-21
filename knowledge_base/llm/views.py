from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from drf_spectacular.utils import extend_schema, OpenApiRequest

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from functools import lru_cache
import hashlib
import json



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
    """Generate a stable key based on question text."""
    return hashlib.md5(question.strip().lower().encode()).hexdigest()


@lru_cache(maxsize=1000)
def cached_answer(question: str):

    query_embedding = model.encode([question]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_chunks = results.get("documents", [[]])[0]

    if not retrieved_chunks:
        return {
            "question": question,
            "answer": "Sorry, no relevant information found in the knowledge base.",
            "chunks_used": 0
        }

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

    # ---------------------------------------------
    # soft hallucination check (FIX)
    # ---------------------------------------------
    # check if ANY important keyword from question or context is missing
    question_keywords = [w.lower() for w in question.split() if len(w) > 4]
    context_lower = context.lower()

    relevant = False
    for k in question_keywords:
        if k in context_lower:
            relevant = True
            break

    # If context seems relevant → keep answer
    if relevant:
        return {
            "question": question,
            "answer": answer,
            "chunks_used": len(retrieved_chunks)
        }

    # If context not relevant → fallback
    return {
        "question": question,
        "answer": "Sorry, no relevant information found in the knowledge base.",
        "chunks_used": len(retrieved_chunks)
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
