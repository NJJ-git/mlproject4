from http.client import HTTPResponse
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .chatbot import run_chatbot


# Create your views here.
def main(request):
    return render(request, 'voice/main.html')


@csrf_exempt
def chat(request):
    print({"text": request.POST['text']})
    ans = run_chatbot(request.POST['text'])
    print("챗봇:", ans)
    return JsonResponse({'result': ans})
