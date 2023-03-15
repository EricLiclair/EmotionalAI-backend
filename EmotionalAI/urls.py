from django.contrib import admin
from django.urls import path, include
from core.views import DefaultView
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/' , include( 'core.urls')),
    path('', DefaultView.as_view())
]
