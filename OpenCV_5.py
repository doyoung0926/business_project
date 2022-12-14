# OpenCV 강좌 : 제 5강 - 대칭

# 대칭 (Flip, Symmetry)

'''
대칭(Flip)은 기하학적인 측면에서 반사(reflection)의 의미를 갖는다.
2차원 유클리드 공간에서의 기하학적인 변환의 하나로 R^2(2차원 유클리드 공간) 위의 선형 변환을 진행함
대칭은 변환할 행렬(이미지)에 대해 2X2 행렬을 왼쪽 곱셈을 진행함.
즉, 'p' 형태의 물체에 Y축 대칭을 적용한다면 'q'형태를 갖게 됨
그러므로, 원본 행렬(이미지)에 각 축에 대한 대칭을 적용했을 때, 단순히 원본 행렬에서 축에 따라 재매핑을 적용하면 대칭된 행렬을 얻을 수 있다.
'''

# 메인 코드
import os

os.chdir(r'C:\Users\USER\Desktop\vs코드\wkit_vs\OpenCV_study')
import cv2

src = cv2.imread('./data/test.jpg', cv2.IMREAD_COLOR)
dst = cv2.flip(src, 0)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

# 세부 코드

# src = cv2.imread('./data/test.jpg', cv2.IMREAD_COLOR)
'''
이미지 입력 함수(cv3.imread)를 통해 원본 이미지로 사용할 src를 선언하고 로컬 경로에서 이미지 파일을 읽어 옴
'''

# dst = cv2.flip(src, 0)
'''
대칭함수(cv2.flip)로 이미지를 대칭
dst = cv2.flip(src, flipCode)는 원본 이미지(src)에 대칭 축(flipCode)을 기준으로 대칭한 출력 이미지(dst)를 반환
대칭 축은 상술ㄹ 입력해 대칭할 축을 설정할 수 있다.
flipCode < 0 XY축 대칭(상하좌우 대칭)
flipCode = 0 X 축 대칭(상하 대칭)
flipCode > 0 Y 축 대칭(좌우 대칭)
'''

# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()

'''
이미지 표시 함수(cv2.imshow)와 키 입력 대기 함수(cv2.waitKey)로 윈도우 창에 이미지를 띄울 수 있다.
이미지 표시 함수는 여러 개의 윈도우 창을 띄울 수 있으며, 동일한 이미지도 여러 개의 윈도우 창으로도 띄울 수 있다.
단, 윈도우 창의 제목은 중복되지 않게 작성함
키 입력 대기 함수로 키가 입력될 때 까지 윈도우 창이 유지되도록 구성함
키 입력 이후, 모든 윈도우 창 제거 함수(cv2.destroyAllWindows)를 이용하여 모든 윈도우 창을 닫음
'''