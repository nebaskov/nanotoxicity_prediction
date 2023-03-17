from django.urls import path
from . import views



urlpatterns = [
    path('sar', views.sar, name='sar'),
    path('r1', views.r1, name='r1'),
    path('r2', views.r2, name='r2'),
    path('offer', views.offer, name='offer'),
]