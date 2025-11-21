from django.urls import path
from .views import AskQuestionAPIView

urlpatterns = [
    path('ask-question/', AskQuestionAPIView.as_view(), name='ask-question'),
]