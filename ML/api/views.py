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

        img_pth = 'https://consolationbucket.s3.ap-northeast-2.amazonaws.com/b7db6140-7978-465b-be05-f1f2fe058db4.png'
        model_pth = '/Users/saaeyun/projects/study-django/ML/api/ai/best_model.pth'

        
        emotion = _output(url, model_pth)

        return JsonResponse(emotion, status=200)

class testView(View):
    def get(self, request):
        
        return JsonResponse({"Hello":"World"}, status=200)
