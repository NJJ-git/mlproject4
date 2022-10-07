import cv2
import numpy as np
from os import makedirs
from os.path import isdir
from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse

class VideoCamera(object):
    # 얼굴 검출 함수
    def face_extractor(img):
        face_classifier = cv2.CascadeClassifier('face\haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        # 얼굴이 없으면 패스!
        if faces ==():
            return None
        # 얼굴이 있으면 얼굴 부위만 이미지로 만들고
        for(x,y,w,h) in faces:
            cropped_face = img[y:y+h, x:x+w]
        # 리턴!
        return cropped_face

    # 얼굴만 저장하는 함수
    def take_pictures(self, name):
        face_dirs = 'faces/'
        # 해당 이름의 폴더가 없다면 생성
        if not isdir(face_dirs+name):
            makedirs(face_dirs+name)

        # 카메라 ON    
        cap = cv2.VideoCapture(0)
        count = 0

        while True:
            # 카메라로 부터 사진 한장 읽어 오기
            ret, frame = cap.read()
            # 사진에서 얼굴 검출 , 얼굴이 검출되었다면 
            if VideoCamera.face_extractor(frame) is not None:
                
                count+=1
                # 200 x 200 사이즈로 줄이거나 늘린다음
                face = cv2.resize(VideoCamera.face_extractor(frame),(200,200))
                # 흑백으로 바꿈
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                # 200x200 흑백 사진을 faces/얼굴 이름/userxx.jpg 로 저장
                file_name_path = face_dirs + name+ '/user'+str(count)+'.jpg'
                cv2.imwrite(file_name_path,face)

                cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.imshow('Face Cropper',face)
            else:
                print("Face not Found")
                pass
            
            # 얼굴 사진 100장을 다 얻었거나 enter키 누르면 종료
            if cv2.waitKey(1)==13 or count==100:
                break

        def __del__(self):
            cap.release()
            cv2.destroyAllWindows()
            print('Colleting Samples Complete!!!')

if __name__ == "__main__":
    # 사진 저장할 이름을 넣어서 함수 호출
    VideoCamera.take_pictures('test')

def gen(camera):
    while True:
        #이미지의 배열값
        frame = camera.take_pictures('test')
        #gen()함수를 종료하지 않고, 중간에 데이터를 전달
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# <img src="/camera/video_feed/" ..>
@gzip.gzip_page  #이미지 영상을 template로 전송. 압축해서.
def video_feed(request):
    try:
        cam = VideoCamera() #객체화. 생성자 호출
        #스트림형태(이미지데이터)로 브라우저로 응답
        return StreamingHttpResponse(gen(cam),\
            content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        cam.video.release()
        pass

# http://localhost:8000/face
def main(request) :
    return render(request, 'face/main.html')
