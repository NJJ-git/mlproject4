import cv2
import os
# import sqlite3
import numpy as np
from PIL import Image
from mlproject.settings import BASE_DIR2

#분류기
detector = cv2.CascadeClassifier(BASE_DIR2+'/faceDetect/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

class FaceRecognition:    
    #카메라 세팅 
    def faceDetect(self, Entry1,):
        face_id = Entry1
        cam = cv2.VideoCapture(0) #초기화, 카메라 번호 (0:default, 1: 2대 이상의 카메라가 있거나 외장 카메라를 사용할 때)
        count = 0 #데이터로 저장할 얼굴의 수
       
        #영상 처리 및 출력
        while(True):

            ret, img = cam.read() #카메라 상태, 프레임
            # img = cv2.flip(img, -1) #상하반전
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #흑백으로
            faces = detector.detectMultiScale(gray, 1.2, 5)
            # 오픈소스에서는 디폴트 값이 1.2, 3여서 검출률을 상승시키기 위해 minNeighbors의 값을 5로 지정
            #scaleFactor 1에 가깝게 해주면 정확도가 상승하나 시간이 오래걸림
            #minNeighbors 얼굴 사이 최소 간격(픽셀) 높여주면 검출률이 상승하나 오탐지율도 상승
            
            #얼굴에 대해 rectangle 출력 -> rectangle의 크기만큼 얼굴을 크롭하여 이미지 추출
            for (x,y,w,h) in faces: #(x,y):얼굴의 좌상단 위치, (w,h):가로,세로

                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2) #이미지,좌상단좌표, 우하단좌표, 색상, 선두께)
                count += 1

                # Save the captured image into the datasets folder
                cv2.imwrite(BASE_DIR2+'/faceDetect/dataset/User.' + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

                cv2.imshow('Register Face', img)
            #종료조건
            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 30: # Take 30 face sample and stop video
                break
    
    
        cam.release() # 캠 종료(메모리 해제)
        cv2.destroyAllWindows() #모든 윈도우 창 닫기 -> 팝업으로 실행된 cam 윈도우창을 닫는다.

    
    def trainFace(self): # 훈련 시작
        # Path for face image database
        path = BASE_DIR2+'/faceDetect/dataset' #경로: dataset폴더 -> 위에서 추출되어 저장된 이미지를 불러옴.

        # function to get the images and label data
        def getImagesAndLabels(path):

            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
            #os.listdir : 해당 디렉토리 내 파일 리스트
            #path + file Name : 경로 list 만들기     
            faceSamples=[]
            ids = []

            for imagePath in imagePaths: #각 파일마다
                #흑백 변환
                PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
                img_numpy = np.array(PIL_img,'uint8') # 그레이스케일로 변환된 이미지를 0~255 사이의 정수 배열로 변환

                #user id
                face_id = int(os.path.split(imagePath)[-1].split(".")[1])
                print("face_id",face_id)
                #마지막 index: -1
                #split(file full path)[-1]: User.1.99.jpg 형식의 파일 처리
                #split(file full path)[-1].split(".")[1]: (0번째 user) 1번째 id값
                #face_id에 int로 입력한 유저의 아이디 값을 저장

                #학습을 위한 얼굴 샘플
                faces = detector.detectMultiScale(img_numpy) # 이미지 배열 불러오기
                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w]) # 이미지 배열을 x, y 값으로 저장 [[],[]]
                    ids.append(face_id) #유저의 아이이디를 배열로 저장

            return faceSamples,ids

        print ("\n Training faces. It will take a few seconds. Wait ...")
        faces,ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids)) # 로그인시 실시간으로 들어오는 이미지를 다시 학습

        # 훈련된 데이터를 모델(trainer.yml)에 저장
        recognizer.save(BASE_DIR2+'/faceDetect/trainer/trainer.yml')
        #훈련된 이미지를 아이디와 함께 확인
        print("\n {0} faces trained. Exiting Program".format(len(np.unique(ids))))


    # 로그인 시 인식되는 부분을 다시 학습해 출력하는 함수
    def recognizeFace(self):
        recognizer.read(BASE_DIR2+'/faceDetect/trainer/trainer.yml') #학습된 모델 읽어오기
        cascadePath = BASE_DIR2+'/faceDetect/haarcascade_frontalface_default.xml' 
        faceCascade = cv2.CascadeClassifier(cascadePath) #분류기 읽어오기

        font = cv2.FONT_HERSHEY_SIMPLEX

        confidence = 0 #일치율 변수
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) #초기화, 카메라 번호 (0:내장, 1:외장) DirectShow 빠른 캠 열림
        # cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        minW = 0.1*cam.get(3)
        minH = 0.1*cam.get(4)

        #영상 처리 및 출력
        while True:

            ret, img =cam.read() #카메라 상태, 프레임

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #흑백으로

            #분류작업
            faces = faceCascade.detectMultiScale( 
                gray,   #흑백으로
                scaleFactor = 1.2,  #원본이미지, scaleFactor 1에 가깝게 해주면 정확도가 상승하나 시간이 오래걸림
                minNeighbors = 5,   # minNeighbors 얼굴 사이 최소 간격(픽셀) 높여주면 검출률이 상승하나 오탐지율도 상승
                minSize = (int(minW), int(minH)), #검출할 객체의 가로, 세로 최소 크기
            )  
            
            #얼굴에 대해 rectangle 출력 -> 얼굴위치 표시용
            for(x,y,w,h) in faces: #(x,y):얼굴의 좌상단 위치, (w,h):가로,세로

                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) #이미지,좌상단좌표, 우하단좌표, 색상, 선두께)

                face_id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                #recognizer.predict(src) : 얼굴 예측 (id와 확률값 반환)
                #confidence가 0에 가까울수록 label과 일치
                
                if (confidence < 100):
                    name = 'Detected'
                else:
                    name = "Unknown"
                
                cv2.putText(img, str(name), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
                #cv2.putText(img, text, bottom-left corner, font, fontScale, color, thickness) : label과 예측값(여기서는 얼굴 인식의 정확도)을 이미지에 폰트로 출력한다.
            cv2.imshow('Detect Face',img) 
            
            #종료조건
            k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
            if confidence > 60:
                break

        print("\n Exiting Program")
        cam.release()
        cv2.destroyAllWindows()
        print(face_id)
        return face_id