# -*- coding: utf-8 -*-

"""
Created on Sat Sep 24 12:22:15 2022

@author: voice

voice 폴더에 urls.py
"""
from django.urls import path
from . import views
urlpatterns = [
    path('',views.main,name='main'),
    path('main/', views.main, name='main'),
    path('logout/', views.logout, name='logout'),
    path('menulist/', views.menulist, name='menulist'),
]