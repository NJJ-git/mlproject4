from http.client import HTTPResponse
from unicodedata import category
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.contrib import auth
from faceDetect.forms import *
from .models import Menu


# Create your views here.
def main(request):
    menulist = Menu.objects.all()
    tealist = Menu.objects.filter(category_id=30)
    beveragelist = Menu.objects.filter(category_id=20)
    coffeelist = Menu.objects.filter(category_id=10)

    return render(request, 'menu/main.html', {'menulist': menulist, 'tealist': tealist, 'coffeelist': coffeelist, 'beveragelist': beveragelist})


def logout(request):
    request.session.flush()  # 로그아웃
    return HttpResponseRedirect("/faceDetect/")
