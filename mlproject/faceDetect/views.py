from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from faceDetect.detection import FaceRecognition
from .forms import *
from .models import *
from django.contrib import messages, auth

faceRecognition = FaceRecognition()


def home(request):
    return render(request, 'faceDetect/home.html')


def register(request):
    if request.method == "POST":
        form = ResgistrationForm(request.POST or None)
        if form.is_valid():
            form.save()
            print("IN HERE")
            messages.success(request, "계정이 성공적으로 등록되었습니다!")
            addFace(request.POST['face_id'])
            return redirect('home')
        else:
            messages.error(request, "계정 등록에 실패했습니다!")
    else:
        form = ResgistrationForm()

    return render(request, 'faceDetect/register.html', {'form': form})


def addFace(face_id):
    face_id = face_id
    faceRecognition.faceDetect(face_id)
    faceRecognition.trainFace()
    return redirect('/')


def login(request):
    # if request.method != 'POST':
    #     return render(request, 'faceDetect/login.html')
    # else:
    user_id = faceRecognition.recognizeFace()
    print("def login의 face_id : "+str(user_id))
    # id1 = request.POST.get('face_id')
    # print(id1)
    try:
        # user = UserProfile.objects.get(face_id=id1)
        if user_id is not None:
            # session_id=UserProfile.objects.get(face_id=user_id)
            # print(session_id)
            request.session['login'] = user_id
            # print("session의 face_id: "+str_id))
            return redirect('greeting', str(user_id))
    #     # return HttpResponseRedirect('greeting', str(face_id))
        else:
            context = {'msg': '얼굴이 일치하지 않습니다', 'url': '../login/'}
            return render(request, 'alert.html', context)
    except:
        context = {'msg': '얼굴을 다시 인식해 주세요!'}
        print("except 실행")
        return render(request, 'faceDetect/login.html', context)

# def logout(request):
#     request.session.flush()  # 로그아웃
#     return HttpResponseRedirect('home')


def Greeting(request, face_id):
    # user_id = request.session.get()
    # print(user_id)
    context = {
        'user': UserProfile.objects.get(face_id=face_id)
    }
    # if user_id:
    #     user = UserProfile.objects.get(pk=face_id)
    #     print('session!')
    #     # return HttpResponseRedirect(f'{user} 로그인 성공')
    return render(request, 'faceDetect/greeting.html', context=context)


def voice(request):
    return HttpResponseRedirect(request, '/voice/main/')
