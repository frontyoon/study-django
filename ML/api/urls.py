from django.urls import path
from api import views

pred = views.PredictView

urlpatterns = [
    path('api/', pred.post()),
]
