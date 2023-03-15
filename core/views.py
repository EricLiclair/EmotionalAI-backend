from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response
from core.classifier import model
import pandas as pd

class PredictMood(APIView):
    def post(self, request, format=None):
        """takes `data` in the post body which is a list of values for the brain signal"""
        data = request.data.get('data')
        x = pd.DataFrame(data=data)
        x = x.transpose()
        response = {
            'message': 'this is coming from PredictMood',
            'prediction': model.predict(X=x)
        }
        return Response(response, status=status.HTTP_200_OK)

class GetMusicRecommendation(APIView):
    def post(self, request, format=None):
        response = {
            'message': 'this is coming from GetMusicRecommendation'
        }
        return Response(response, status=status.HTTP_200_OK)