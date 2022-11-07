# OpenCV 강좌: 제 3강 - 이미지 출력

'''
OpenCV는 래스터 그래픽스 이미지 파일 포맷을 쉽게 불러올 수 있는 별도의 함수를 제공함
이 함수는 불러온 압축 해제된 데이터 구조에 필요한 메모리 할당과 같은 복잡한 작업을 처리하며, 파일 시그니처(File Signature)를 읽어 적절한 코덱을 결정함
OpenCV에서 이미지를 불러올 때는 확장자를 확인하는 방식이 아닌 파일 시그니처를 읽어 파일의 포맷을 분석함
파일 시그니처는 파일 매직 넘버(File Magic Number)라고도 하며, 각 파일 형식마다 몇 개의 바이트가 지정되어 있습니다.
예로, PNG 확장자의 경우 89 50 4E 47 ... 형태로 파일 헤더에 포함되어 있다
이미지 입력 함수는 운영체제의 코덱을 사용해 운영체제 별로 픽셀값이 다를 수 있다.
'''

# 메인 코드
'''
# import os

# os.chdir(r'C:\Users\USER\Desktop\vs코드\wkit_vs\OpenCV_study')
import cv2

image = cv2.imread('./data/test.jpg', cv2.IMREAD_ANYCOLOR)
cv2.imshow('Moon', image)
cv2.waitKey()
cv2.destroyAllWindows()
'''


# 세부 코드

# image = cv2.imread('./data/test.jpg', cv2.IMREAD_ANYCOLOR)
'''
이미지 입력 함수(cv2.imread)를 통해 로컬 경로의 이미지 파일을 읽어올 수 있다.
image = cv2.imread(fileName, flages)는 파일 경로(fileName)의 이미지 파일을 플래그(flags) 설정에 따라 불러온다.
파일 경로(fileName)는 상대 경로 또는 절대 경로를 사용하여 이미지를 불러옵니다.
flags은 이미지를 초기에 불러올 때 적용할 초기 상태를 의미함.

- flags
* cv2.IMREAD_UNCHANGED: 원본 사용
* cv2.IMREAD_GRAYSCALE: 1채널, 그레이스케일 적용 
* cv2.IMREAD_COLOR: 3 채널, BGR 이미지 사용
* cv2.IMREAD_ANYDEPTH: 이미지에 따라 정밀도를 16/32비트 또는 8 비트로 사용
* cv2.IMREAD_ANYCOLOR: 가능한 3 채널, 색상 이미지로 사용
* cv2.IMREAD_REDUCED_GRAYSCALE_2: 1 채널, 1/2 크기, 그레이스 케일 적용
* cv2.IMREAD_REDUCED_GRAYSCALE_4: 1 채널, 1/4 크기, 그레이스 케일 적용
* cv2.IMREAD_REDUCED_GRAYSCALE_8: 1 채널, 1/8 크기, 그레이스 케일 적용
* cv2.IMREAD_REDUCED_COLOR_2: 3 채널, 1/2 크기, BGR 이미지 사용
* cv2.IMREAD_REDUCED_COLOR_4: 3 채널, 1/4 크기, BGR 이미지 사용
* cv2.IMREAD_REDUCED_COLOR_8: 3 채널, 1/8 크기, BGR 이미지 사용
'''

# cv2.imshow('Moon', image)
# cv2.waitKey()
# cv2.destroyAllWindows()
'''
이미지 표시 함수(cv2.imshow)와 키 입력 대기 함수(cv2.WaitKey)로 윈도우 창에 이미지를 띄울 수 있다.
키 입력 대기 함수를 사용하지 않을 경우 윈도우 창이 유지되지 않고 프로그램이 종료됨
키 입력 이후, 모든 윈도우 창 제거 함수(cv2.destroyAllWindows)를 이용하여 모든 윈도우 창을 닫음
'''

# 추가 정보

# height, width channel = image.shape
# print(height, width, channel)

'''
결과 
1920 1280 3

height, width, channel = image.shape를 이용하여 해당 이미지의 높이, 너비, 채널의 값을 확인
이미지의 속성은 크기, 정밀도, 채널을 주요한 속성으로 사용함
- 크기: 이미지의 높이와 너비를 의미
- 정밀도: 이미지의 처리 결과의 정밀성을 의미함
- 채널: 이미지의 색상 정보를 의미함
Tip: 유효 비트가 많을수록 더 정밀해짐
Tip: 채널이 3일 경우 다색 이미지임. 채널이 1일 경우 단색 이미지임
'''