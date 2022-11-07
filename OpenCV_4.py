# OpenCV 강좌: 제 4강 - 비디오 출력

'''
동영상 파일에서 순차적으로 프레임을 읽어 이미지의 형태로 출력함.
동영상 파일을 읽으려면 컴퓨터에 동영상 코덱을 읽을 수 있는 라이브러리가 설치돼 있어야 함
OpenCV는 FFmpeg를 지원하므로 *.avi나 *.mp4 등 다양한 형식의 동영상 파일을 손쉽게 읽을 수 있다.
이미지 파일 중, GIF 확장자는 프레임이 존재하므로, 동영상 파일을 읽는 방법과 동일하게 처리함
'''

# 메인 코드
'''
import os

os.chdir(r'C:\Users\USER\Desktop\vs코드\wkit_vs\OpenCV_study')
import cv2

capture = cv2.VideoCapture('./data/angry_man_pixabay.mp4')

while cv2.waitKey(33) < 0:
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = capture.read()
    cv2.imshow('VideoFrame', frame)

capture.release()
cv2.destroyAllWindows()
'''

# 세부 코드

# capture = cv2.VideoCapture('Image/Star.mp4')

'''
비디오 출력 클래스(cv2.VideoCapture)를 통해 동영상 파일에서 정보를 받아올 수 있다.
capture = cv2.VideoCapture(fileName)는 파일 경로(fileName)의 동영상 파일을 불러옴.
단, 카메라의 장치 번호가 아닌, 동영상 파일의 경로를 지정함
'''

# if (capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
#     capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

'''
비디오 속성 반환 메서드(capture.get)로 비디오의 속성을 반환함
비디오의 정보 중, 동영상의 현재 프레임 수(cv2.CAP_PROP_POS_FRAMES)와 동영상의 총 프레임 수(cv2.CAP_PROP_FRAME_COUNT)를 받아옴
분기문(if)을 이용하여 동영상의 현재 프레임 수와 동영상의 총 프레임 수를 비교함
현재 프레임의 수가 총 프레임 수가 같다면, 현재 재생되고 있는 프레임은 가장 마지막이 됨
마지막 프레임은 동영상이 종료되는 시점이 되므로, 비디오 속성 설정 메서드(capture.get)로 동영상의 현재 프레임을 초기화함
TIP: 또는, 동영상 파일 읽기 메서드(capture.open)를 이용하여 다시 동영상 파일을 불러올 수도 있다.
'''

# 추가 정보

'''
- VideoCapture 메서드
capture.isOpened() : 동영상 파일 열기 성공 여부 확인
capture.open(filename) : 동영상 파일 열기
capture.set(propid, value) : 동영상 속성 설정
capture.get(propid) : 동영상 속성 반환
capture.release() : 동영상 파일을 닫고 메모리 해제

- VideoCapture 속성
cv2.CAP_PROP_FRAME_WIDTH 프레임의 너비 -
cv2.CAP_PROP_FRAME_HEIGHT 프레임의 높이 -
cv2.CAP_PROP_COUNT 총 프레임 수 - 
cv2.CAP_PROP_FPS 프레임 속도 -
cv2.CAP_PROP_FOURCC 코덱 코드 -
cv2.CAP_PROP_BRIGHTNESS 이미지 밝기 카메라만 해당
cv2.CAP_PROP_CONTRAST 이미지 대비 카메라만 해당
cv2.CAP_PROP_SATURATION 이미지 채도 카메라만 해당
cv2.CAP_PROP_HUE 이미지 색상 카메라만 해당
cv2.CAP_PROP_GAIN 이미지 게인 카메라만 해당
cv2.CAP_PROP_EXPOSURE 이미지 노출 카메라만 해당
cv2.CAP_PROP_POS_MSEC 프레임 재생 시간 ms 반환
cv2.CAP_PROP_POS_FRAMES 현재 프레임 프레임의 총 개수 미만
CAP_PROP_POS_AVI_RATIO 비디오 파일 상대 위치 0=시작, 1=끝
'''

