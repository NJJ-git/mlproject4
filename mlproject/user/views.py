from django.shortcuts import render
from .models import User
from django.http import HttpResponseRedirect
from django.contrib import auth   

# Create your views here.
#http://127.0.0.1:8000/member/login/
def login(request) :
    if request.method != 'POST' :
        return render(request,'user/login.html')
    else :
        id1 = request.POST['id'] #id 파라미터
        pass1 = request.POST['pass'] #pass 파라미터
        try :
            #User 테이블에서 id 컬럼의 값이 id1인 객체 리턴
            #  해당 정보가 없는 경우 예외 발생함
            user = User.objects.get(id=id1) 
            # user.pass1 : db의 저장된 비밀번호
            # pass1 : 입력된 비밀번호
            if user.pass1 == pass1 :  #정상 로그인환경
                #request.session : session 객체.
                session_id = request.session.session_key
                request.session['login'] = id1  #session객체에 로그인 정보 저장
                return HttpResponseRedirect('../main/') #redirect 설정 
            else :  #아이디는 존재. 비밀번호가 틀린경우 
                context = {'msg':'비밀번호가 틀립니다.','url':'../login/'}
                return render(request,'alert.html',context)
        except :
                context = {'msg':'아이디를 확인하세요.'}
                return render(request,'user/login.html',context)
            
def join(request) :
    if request.method != 'POST' :
       return render(request,'user/join.html')
    else :
        user = User(id=request.POST['id'],\
                        pass1=request.POST['pass'],\
                        name=request.POST['name'],\
                        gender=request.POST['gender'],\
                        tel=request.POST['tel'],\
                        email=request.POST['email'],\
                        picture=request.POST['picture'])
        user.save()  #member 객체를 통해서 데이터베이스에 insert 됨
        return HttpResponseRedirect("../login/")

def main(request):
    return render(request, 'user/main.html')

 
def logout(request) :    
    auth.logout(request) #로그아웃
    return HttpResponseRedirect("../login/")

#info/<str:id>/
def info(request,id):  #id : apple 아이디정보
    try :
        login = request.session["login"]  #세션정보. 로그인 정보 
    except :
        login = ""  #로그인 정보가 없는 경우 login=빈문자열 저장
        
    if login != "" : #로그인 된경우
       #login == 'admin' : 관리자로 로그인한 상태. 다른사용자 정보 조회가능
       if login == id or login == 'admin': #정상적인 기능
          user = User.objects.get(id=id) #id에 해당하는 회원 정보
          return render(request, 'user/info.html',{"mem":user})
       else : #다른 사용자의 정보를 조회
          context = {"msg": "본인 정보만 조회가능합니다.", "url": "../../main/"}
          return render(request, 'alert.html', context)
    else : #로그아웃상태
        context = {"msg": "로그인 하세요.", "url": "../../login/"}
        return render(request, 'alert.html', context)
    
'''
  수정 검증 : 
    1. 로그인 필요
    2. 본인 정보만 수정 가능
'''
def update(request,id):  #id : apple 아이디정보
    try :
        login = request.session["login"]  #세션정보. 로그인 정보 
    except :
        login = ""  #로그인 정보가 없는 경우 login=빈문자열 저장
        
    if login != "" : #로그인 된경우
       if login == id : #관리자인 경우 수정불가
           return update_rtn(request,id)
       else : #다른 사용자의 정보를 조회
          context = {"msg": "본인 정보만 수정가능합니다.", "url": "../../main/"}
          return render(request, 'alert.html', context)
    else : #로그아웃상태
        context = {"msg": "로그인 하세요.", "url": "../../login/"}
        return render(request, 'alert.html', context)
'''
   검증 통과한 경우 실행함수
1. GET 방식인 경우 
   id에 해당하는 db 정보를 읽어서 update.html 페이지로 전송

2. POST 방식인 경우 
   - 비밀번호 검증
   - 비밀번호가 일치 - db에 내용 수정. 상세보기 페이지 이동
   - 비밀번호가 불일치 - 확인 메세지 출력 후, update.html 페이지로 다시 보여 주기
'''
def update_rtn(request,id) :    
    if request.method != 'POST':
       user = User.objects.get(id=id)  # id에 해당하는 회원 정보
       return render(request, 'user/update.html', {"mem": user})
    else :   #POST 방식인 경우
        user = User.objects.get(id=id)  # id에 해당하는 회원 정보
        #member.pass1 : db에 등록된 비밀번호
        #request.POST['pass'] : 화면에 입력된 비밀번호
        if user.pass1 == request.POST['pass'] :
           user = User(id=request.POST['id'],      name=request.POST['name'],\
                           pass1=request.POST['pass'], gender=request.POST['gender'],\
                           tel=request.POST['tel'],email=request.POST['email'],\
                           picture=request.POST['picture'])
           user.save() # 기본키가 존재하면 update
           return HttpResponseRedirect("../../info/"+id+"/") #회원정보 URL 정보 복원
        else :  #비밀번호 오류 
           context = {"msg": "회원 정보 수정 실패. \\n비밀번호 오류 입니다.",\
                      "url": "../../update/"+id+"/"}
           return render(request, 'alert.html', context)
       
def delete(request,id) :
    try :
        login = request.session["login"]
    except :
        login = ""

    if login != "" :
       if login == id :
           return delete_rtn(request,id)
       else :
          context = {"msg": "3.본인만 탈퇴 가능합니다.", "url": "../../main/"}
          return render(request, 'alert.html', context)
    else : #로그아웃상태
        context = {"msg": "3.로그인 하세요.", "url": "../../login/"}
        return render(request, 'alert.html', context)

def delete_rtn(request,id) :
    if request.method != 'POST':
       return render(request, 'member/delete.html', {"id":id})
    else :
        user = User.objects.get(id=id)
        if user.pass1 == request.POST['pass'] :
            user.delete()  #member에 해당하는 내용을 delete
            auth.logout(request)  #로그아웃
            context = {"msg": "회원님 탈퇴처리가 완료 되었습니다.", "url": "../../login/"}
            return render(request, 'alert.html', context)
        else :
           context = {"msg": "비밀번호 오류 입니다.", "url": "../../delete/"+id+"/"}
           return render(request, 'alert.html', context)