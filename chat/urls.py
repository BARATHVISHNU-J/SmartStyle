from django.urls import path
from . import views

app_name = 'chat'

urlpatterns = [
    path('', views.chat_view, name='chat'),
    path('api/chat/', views.chat_api, name='chat_api'),
    path('api/preferences/', views.update_preferences, name='update_preferences'),
]
