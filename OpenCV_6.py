# OpenCV 강좌 : 제 6강 - 회전

# 회전(Rotate)

'''
회전(Rotate)은 선형 변환 중 하나에 포함되며, 회전 변환 행렬(Rotation matrix)을 통해 변환이 진행됨
회전 변환 행렬은 임의의 점을 중심으로 물체를 회전시킴.
횐전 변환 행렬의 일부는 반사 행렬(Reflection matrix)과 같은 값을 지닐 수 있다.
2차원 유클리드 공간에서의 회전은 크게 두 가지 회전 행렬을 갖는다.
좌푯값을 회전시키는 회전 행렬과 좌표 축을 회전시키는 회전 행렬이 있다.
좌표 회전 행렬은 원점을 중심으로 좌푯값을 회전시켜 매핑하며,
좌표 축 회전 행렬은 원점을 중심으로 행렬 자체를 회전시켜 새로운 행렬의 값을 구성함.
OpenCV의 회전 함수는 좌표 축의 회전 이동 행렬과 동일한 형태이며,
비율을 조정하거나 중심점의 기준을 변경하여 회전할 수 있다.
'''

# 메인 코드
import os

os.chdir(r'C:\Users\USER\Desktop\vs코드\wkit_vs\OpenCV_study')
import cv2

src = cv2.imread('./data/test.jpg', cv2.IMREAD_COLOR)

height, width, channel = src.shape
matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
dst = cv2.warpAffine(src, matrix, (width, height))

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

# 세부 코드

# src = cv2.imread('./data/test.jpg', cv2.IMREAD_COLOR)

'''
이미지 입력 함수(cv2.imread)를 통해 원본 이미지로 사용할 src를 선언하고 로컬 경로에서 이미지 파일을 읽어 옴
'''

# height, width, channel = src.shape
'''
height, width, channel = src.shape를 이용하여 해당 이미지의 높이, 너비, 채널의 값을 저장함
높이와 너비를 이용하여 회전 중심점을 설정함
'''

# matrix= cv2.getRotationMatrix2D((width/2, height/2), 90, 1)

'''
2X3 회전 행렬 생성 함수 (cv2.getRotationMatrix2D)로 회전 변환 행렬을 계산함
matrix = cv2.getRotationMatrix2D(center, angle, scale)는 중심점(center), 각도(angle), 비율(scale)로 매핑 변환 행렬(matrix)을 생성함
중심점(center)은 튜플(tuple) 형태로 사용하며 회전의 기준점
각도(angle)는 중심점을 기준으로 회전할 각도
비율(scale)은 이미지의 확대 및 축소 비율
'''

# dst = cv2.warpAffine(src, matrixm (width, height))

'''
아핀 변환 함수(cv2.warpAffine)로 회전 변환을 계산합니다.
dst = cv2.warpAffine(src, M, dsize)는 원본 이미지(src)에 M(아핀 맵 행렬)을 적용하고
출력 이미지 크기(dsize)로 변형해서 출력 이미지(dst)를 반환함
아핀 맵 행렬(M)은 회전 행렬 생성 함수에서 반환된 매핑 변환 행렬을 사용함
출력 이미지 크기(dsize)는 튜플(tuple) 형태로 사용하여 출력 이미지의 너비와 높이를 의미함
아핀 맵 행렬에 따라 회전된 이미지를 반환
'''