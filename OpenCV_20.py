# OpenCV 강좌 : 제 20강 - 캡쳐 및 녹화

# 캡쳐 및 녹화(Capture & Record)

'''
영상이나 이미지를 캡쳐하거나 녹화하기 위해 사용함
영상이나 이미지를 연속적 또는 순간적으로 캡쳐하거나 녹화할 수 있다.
'''

# 메인 코드

import os

os.chdir(r'C:\Users\USER\Desktop\vs코드\wkit_vs\OpenCV_study')

import datetime
import cv2

capture = cv2.VideoCapture('./data/test.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False

while True:
    if (capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        capture.open('./data/test.mp4')

    ret, frame = capture.read()
    cv2.imshow('VideoFrame', frame)

    now = datetime.datetime.now().strftime('%d_%H-%M-%S')
    key = cv2.waitKey(33)


    if key == 27:
        break
    elif key == 26:
        print('캡쳐')
        cv2.imwrite('D:/'+str(now)+'.png', frame)
    elif key == 24:
        print('녹화 시작')
        record = True
        video = cv2.VideoWriter('D:/'+str(now)+'.avi',fourcc,20.0,(frame.shape[1], frame.shape[0]))
    elif key == 3:
        print('녹화 중지')
        record = False
        video.release()

# 세부 코드

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# record = False

'''
fourcc를 생성하여 디지털 미디어 포맷 코드를 생성함
cv2.VideoWriter_fourcc(*'코덱')을 사용하여 인코딩 방식을 성정함
record 변수를 생성하여 녹화 유/무를 설정함
Tip: FourCC(Four Character Cod): 디지털 미디어 포맷 코드임.
즉, 코덱의 인코딩 방식을 의미함
'''

# import datetime
# now = datetime.datetime.now().strftime('%d_%H-%M-%S')

'''
datetime 모듈을 포함하여 현재 시간을 받아와 제목으로 사용함
now에 파일의 제목을 설정함. 날짜_시간-분-초의 형식으로 제목이 생성됨
'''

# key = cv2.waitKey(33)

'''
key 값에 현재 눌러진 키보드 키의 값이 저장됩니다. 33ms 마다 갱신됨
'''

# if key == 27:
#     break
# elif key == 26:
#     print('캡쳐')
#     cv2.imwrite('D:/' + str(now) + '.png', frame)
# elif key == 24:
#     print('녹화 시작')
#     record = True
#     video = cv2.VideoWriter('D:/' + str(now) + '.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
# elif key == 3:
#     print('녹화 중지')
#     record = False

'''
if-elif 문을 이용하여 눌러진 키의 값을 판단한다.
27=ESC, 26=Ctrl + Z, 24=Ctrl + X, 3=Ctrl + C 를 의미함
ESC 키를 눌렀을 경우, 프로그램을 종료함
Ctrl + Z를 눌렀을 경우, 현재 화면을 캡쳐함.
cv2.imwrite('경로 및 제목', 이미지)를 이용하여 해당 이미지를 저장함
Ctrl + X를 눌렀을 경우, 녹화를 시작함.
video에 녹화할 파일 형식을 설정함
cv2.VideoWriter('경로 및 제목', 비디오 포맷 코드, FPS, (녹화 파일 너비, 녹화 파일 높이))를 의미함
Ctrl + C를 눌렸을 경우, 녹화를 중지함
video.release()를 사용하여 메모리를 해제함.
녹화 시작할 때, record를 True로, 
녹화를 중지할 때 record 를 False로 변경함
Tip: key 값은 아스키 값을 사용함
Tip: FPS(Frame Per Second): 영상이 바뀐느 속도를 의미함. 즉, 화면의 부드러움을 의미함
Tip: frame.shape는 (높이,너비,채널)의 값이 저장되어있다.
'''

# if record == True:
#     print('녹화 중..')
#     video.write(frame)

'''
if 문을 이용하여 record가 True일때 video에 프레임을 저장한다.
video.write(저장할 프레임)을 사용하여 프레임을 저장할 수 있다.
'''

# 추가 정보
'''
- FourCC 종류

CVID< Default, DIB, DIVX, H261, H264, IV32, IV4, IV50,
IYUB, MJPG, MP42, MP43, MPG4, MSVC,PIM1, Prompt, XVID

Tip: 단일 채널 이미지의 경우, 사용할 수 없는 디지털 미디어 포맷 코드가 존재함
'''
