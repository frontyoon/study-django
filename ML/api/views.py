from rest_framework.views import APIView
from rest_framework.response import Response
#from serializers import emotion_objectSerializer
from models.djanog_fn import _output



class PredictView(APIView):
    def post(self, request):
        img_pth = 'C:\\Project\\study-django\\ML\\api\\models\\img\\jung.jpg'
        model_pth = 'C:\\Project\\study-django\\ML\\api\\models\\best_model.pth'

        emotion = _output(img_pth, model_pth)

        return Response(status=200, emotion=emotion)