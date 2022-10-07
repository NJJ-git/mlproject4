from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
import os
from mlproject import settings # 필요 부분 기존 파일에서 가져와야 함.
import random
import base64
from django.http import JsonResponse

# Create your views here.
# http://localhost:8000/writing/index

def index(request) :
    return render(request, template_name='writing/index.html')

def order(request) :
    return render(request, template_name='writing/order.html')    

@csrf_exempt
def upload(request) :
    data = request.POST.__getitem__('data')
    data = data[22:]
    number = random.randrange(1,10000)
    uploadpath = "mlproject/static/writing_images"
    path = "mlproject/static/writing_images"
    filename = 'writing.png'
    image = open(path+filename, mode="wb")
    image.write(base64.b64decode(data))
    image.close()
    answer = {"filename":"writing.png"}
    return JsonResponse(answer)
    
       
    
    