from http.client import HTTPResponse
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.contrib import auth 
from faceDetect.forms import *
import menu

# Create your views here.
def main(request):
    return render(request, 'menu/main.html')
def logout(request) :    
    request.session.flush() #로그아웃
    return HttpResponseRedirect("/faceDetect/")





