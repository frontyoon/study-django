from django.urls import path
from .views import PredictView

urlpatterns = [
    path('model', PredictView.as_view()),
]
