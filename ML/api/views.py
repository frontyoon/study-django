from django.views import View
import json
from .ai.djanog_fn import _output
from django.http import JsonResponse
import os

# ai 함수 실행 View
class PredictView(View):
    def post(self, request):
        # body의 json값 받아와서 url 이라는 변수에 저장
        json_object = json.loads(request.body)
        url = json_object

        # 모델 위치 상대 경로
        file_path = os.path.abspath(__file__)
        folder_path = os.path.dirname(file_path)
        model_pth = os.path.join(folder_path, 'ai/best_model.pth')
        
        # 모델 돌리기
        emotion = _output(url, model_pth)

        # json으로 반환
        return JsonResponse(emotion, status=200)

