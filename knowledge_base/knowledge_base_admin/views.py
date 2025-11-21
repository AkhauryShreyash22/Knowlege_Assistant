from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from drf_spectacular.utils import extend_schema, OpenApiTypes, OpenApiParameter

from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

import os
import uuid  

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

model = SentenceTransformer("all-MiniLM-L6-v2")


class KnowledgeBaseDocumentUpload(APIView):
    parser_classes = (MultiPartParser, FormParser)
    
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
        file_obj = request.FILES.get("file")
        if not file_obj:
            return Response({"error": "File is required"}, status=400)

        save_path = os.path.join("uploaded_files", file_obj.name)
        os.makedirs("uploaded_files", exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in file_obj.chunks():
                f.write(chunk)

        reader = PdfReader(save_path)
        chunks = []
        metadatas = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                page_chunks = [text[j:j + 500] for j in range(0, len(text), 500)]
                for chunk in page_chunks:
                    chunks.append(chunk)
                    metadatas.append({
                        "source": file_obj.name,
                        "page": i + 1
                    })

        embeddings = model.encode(chunks).tolist()

        ids = [str(uuid.uuid4()) for _ in chunks]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return Response({
            "message": "File uploaded and indexed successfully",
            "total_chunks": len(chunks)
        })
