from http.client import HTTPResponse
from unicodedata import category
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.contrib import auth 
from faceDetect.forms import *
from .models import Menu


# Create your views here.
def main(request):
    return render(request, 'menu/main.html')
def logout(request) :    
    request.session.flush() #로그아웃
    return HttpResponseRedirect("/faceDetect/")