from http.client import HTTPResponse
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.contrib import auth 
from faceDetect.forms import *


# Create your views here.
def main(request):
    return render(request, 'menu/main.html')
def logout(request) :    
    auth.logout(request) #로그아웃
    return HttpResponseRedirect("/faceDetect/home.html")