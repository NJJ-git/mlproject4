from django.urls import path
from . import views

urlpatterns = [
    path('index', views.index, name='index'),
    path('upload', views.upload, name='upload'),
    path('order', views.order, name='order'),
]
 