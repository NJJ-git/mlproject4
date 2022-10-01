# -*- coding: utf-8 -*-

"""
Created on Sat Sep 24 12:22:15 2022

@author: user

user 폴더에 urls.py
"""
from django.urls import path
from . import views
urlpatterns = [
    path('', views.main, name='main'),
    path('login/', views.login, name='login'),  # user/login/ 요청
    path('join/', views.join, name='join'),
    path('main/', views.main, name='main'),
    path('logout/', views.logout, name='logout'),
    # path('info/<str:id>/', views.info, name='info'), # /user/info/apple/ #id : apple
    #path('update/<str:id>/', views.update, name='update'),
    #path('delete/<str:id>/', views.delete, name='delete'),
    #   path('password/<str:id>/', views.password, name='password'),
    #   path('list/', views.list,name='list'),
    #   path('picture/', views.picture,name='pircure'),
]
