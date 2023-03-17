from django.urls import path
from . import views


urlpatterns = [
    path('', views.main, name='prediction'),
    path('medium', views.medium, name='prediction_medium'),
    path('basic', views.basic, name='prediction_basic'),
    path('sar', views.main_sar, name='prediction_sar'),
    path('mediumsar', views.medium_sar, name='prediction_medium_sar'),
    path('basicsar', views.basic_sar, name='prediction_basic_sar'),
    ]