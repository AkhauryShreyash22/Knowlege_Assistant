from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status

from drf_spectacular.utils import extend_schema

import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from transformers import pipeline

rag_llm = pipeline(
    "text-generation",
    model="google/flan-t5-large",
    max_new_tokens=300
)


chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)
model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(file):
    name = file.name.lower()

    if name.endswith(".pdf"):
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        return text

    elif name.endswith(".md") or name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    return ""


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


class KnowledgeBaseDocumentUpload(APIView):
    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(
        summary="Upload a PDF, Markdown, or Text file",
        request={
            'multipart/form-data': {
                'type': 'object',
                'properties': {
                    'file': {
                        'type': 'string',
                        'format': 'binary',
                        'description': 'Upload PDF / MD / TXT'
                    }
                },
                'required': ['file'],
            }
        },
        responses={200: None},
    )
    def post(self, request):
        file = request.FILES.get("file")

        if not file:
            return Response({"error": "File is required"}, status=400)

        text = extract_text(file)
        if not text.strip():
            return Response({"error": "Unable to extract text from file"}, status=400)

        chunks = chunk_text(text)

        embeddings = model.encode(chunks).tolist()

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"{file.name}_{i}" for i in range(len(chunks))]
        )

        return Response({
            "message": "File uploaded and indexed",
            "chunks_stored": len(chunks)
        }, status=200)
