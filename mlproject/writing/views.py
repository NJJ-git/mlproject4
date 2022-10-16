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

# 여기부터는 아무거나 친거임
def coffee_order(request) :
    return render(request, template_name='writing/coffee_order.html')  


## 변수 선언
coffee=0

## 함수 정의
@csrf_exempt
def coffee_machine(button) :
    print("#1. 뜨거운 물을 준비한다.");
    print("#2. 종이컵을 준비한다");

    if coffee == 1:
        print("#3. : 밀크커피 가루 준비")
    elif coffee == 2:
        print("#3. : 설탕커피 가루 준비")
    elif coffee == 3:
        print("#3. : 블랙커피 가루 준비")
    else:
        print(("#3. 아무거나 탄다\n"))

    print("#4. 물을 붓는다.");
    print("#5. 스푼 준비");
    print()
    print("커피 완료 신호 ");

## 메인 코드

    coffee=int(input("A손님, 어떤 커피를 드릴까요? (1:밀크, 2:설탕, 3:블랙) "))
    coffee_machine(coffee)
    print("A손님 커피 완료")

    coffee=int(input("B손님, 어떤 커피를 드릴까요? (1:밀크, 2:설탕, 3:블랙) "))
    coffee_machine(coffee)
    print("B손님 커피 완료")

    coffee=int(input("C손님, 어떤 커피를 드릴까요? (1:밀크, 2:설탕, 3:블랙) "))
    coffee_machine(coffee)
    print("C손님 커피 완료")


    
       
    
    