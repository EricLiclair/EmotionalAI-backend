from django.urls import path

from .views import (PredictMood, GetMusicRecommendation)

urlpatterns = [

    # profile Login Logout
    path('predict/', PredictMood.as_view()),
    path('recommend/', GetMusicRecommendation.as_view()),
]
