from django.urls import path
from .views import KnowledgeBaseDocumentUpload

urlpatterns = [
    path('upload-document/', KnowledgeBaseDocumentUpload.as_view(), name='upload-document'),
]