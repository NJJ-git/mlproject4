from http.client import HTTPResponse
from django.shortcuts import render
from django.http import HttpResponseRedirect


# Create your views here.
def main(request):
    return render(request, 'menu/main.html')