from django.views import View
import json
#from rest_framework.response import Response
#from serializers import emotion_objectSerializer
from .ai.djanog_fn import _output
from django.http import JsonResponse


class PredictView(View):
    def get(self, request):
 
        json_object = json.loads(request.body)
        url = json_object['imageUrl']

        model_pth = 'ai/best_model.pth'

        
        emotion = _output(url, model_pth)

        return JsonResponse(emotion, status=200)

