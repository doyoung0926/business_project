# OpenCV 강좌 : 제 26강 - 모폴리지 변환

# 모폴로지 변환(Morphological Transformation)

'''
모폴로지 변환(Perspective Transformation)은 영상이나 이미지를 형태학적 관점에서 접근하는 기법을 의미
모폴로지 변환은 주로 영상 내 픽셀값 대체에 사용됨.
이를 응용해서 노이즈 제거, 요소 결합 및 분리, 강도 피크 검출 등에 이용할 수 있다.

집합의 포함 관계, 이동(translation), 대칭(reflection), 여집합(complement), 차집합(difference) 등의 성질을 사용함
기본적인 모폴로지 변환으로는 팽창(dilation)과 침식(erosion)이 있다.

팽창과 침식은 이미지와 커널의 컨벌루션 연산이며, 이 두 가지 기본 연산을 기반으로
복잡하고 다양한 모폴로지 연산을 구현할 수 있다.
'''


'''
- 팽창(Dilation)

팽창은(dilation)은 커널 영역 안에 존재하는 모든 픽셀의 값을 커널 내부의 극댓값(local maximum)으로 대체함
즉, 구조 요소(element)를 활용해 이웃한 픽셀들을 최대 픽셀값으로 대체함
팽창 연산을 적용하면 어두운 영역이 줄어들고 밝은 영역이 늘어남
커널의 크기나 반복 횟수에 따라 밝은 영역이 늘어나 스펙클(speckle)이
커지며 객체 내부의 홀(holes)이 사라짐
팽창 연산은 노이즈 제거 후 줄어든 크기를 복구하고자 할 때 주로 사용함

- 침식(Erosion)

침식(erosion)은 커널 영역 안에 존재하는 모든 픽셀의 값을 커널 내부의 극솟값(local minimum)으로 대체함
즉, 구조 요소(element)를 활용해 이웃한 픽셀을 최소 픽셀값으로 대체함
침식 연산을 적용하면 밝은 영역이 줄어들고 어두운 영역이 늘어남
커널의 크기나 반복 횟수에 따라 어두운 영역이 늘어나 스펙클(speckle)이 사라지며,
객체 내부의 홀(holes)이 커짐
침식 연산은 노이즈 제거에 주로 사용함
'''

# 메인 코드

import os

os.chdir(r'C:\Users\USER\Desktop\vs코드\wkit_vs\OpenCV_study')

import numpy as np
import cv2

src = cv2.imread('./data/test.jpg')

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9,9))
dilate = cv2.dilate(src, kernel, anchor=(-1,-1), iterations=5)
erode = cv2.erode(src, kernel, anchor=(-1,-1), iterations=5)

dst = np.concatenate((src, dilate, erode), axis=1)

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 세부 코드

# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9,9))

'''
cv2.getStructuringElement()를 활용해 구조요소르 생성함
cv2.getStructuringElement(커널의 형태, 커널의 크기, 중심점)로 구조 요소을 생성함
커널의 형태는 직사각형(Rect), 십자가(Cross), 타원(Ellipse)이 있다.
커널의 크기는 구조 요소의 크기를 의미함
이땨, 커널의 크기가 너무 작다면 커널의 형태는 여향을 받지 않는다.
고정점은 커널의 중심 위치를 나타냄
필수 매개변수가 아니며, 설정하지 않을 경우 사용되는 함수에서 값이 결정됨

Tip: 고정점을 할당하지 않을 경우 조금 더 유동적인 커널이 됨
'''

# dilate = cv2.dilate(src, kernel, anchor=(-1,-1), iterations=5)
# erode = cv2.erode(src, kernel, anchor=(-1,-1), iterations=5)

'''
생성된 구조 요소를 활용해 모폴로지 변환을 적용함
팽창 함수(cv2.dilate)와 침식 함수(cv2.erode)로 모폴로지 변환을 진행함
cv2.dilate(원본 배열, 구조 요소, 고정점, 반복 횟수, 테두리 외삽법, 테두리 색상)로 팽창 연산을 진행함
cv2.erode(원본 배열, 구조 요소, 고정점, 반복 횟수, 테두리 외삽법, 테두리 색상)로 침식 연산을 진행함
팽창 함수와 침식 함수의 매개변수 순서와 의미는 동일함
단, 팽창 연산의 경우 밝은 영역이 커지며, 침식 연산의 경우 어두운 영역이 커짐

Tip: 고정점을 (-1, -1)로 할당할 경우, 커널의 중심부에 고정점이 위치하게 됨
'''

# dst = np.concatenate((src, dilate, erode), axis = 1)

'''
Numpy 함수 중 연결 함수(np.concatenate)로 원본 이미지, 팽창 결과, 침식 결과를 하나의 이미지로 연결함
np.concatenate(연결할 이미지 배열들, 축 방향)로 이미지를 연결함

Tip: axis=0으로 사용할 경우, 세로 방향으로 연결됨
Tip: OpenCV의 함수 중, 수평 연결 함수(cv2.hconcat)와 수직 연결 함수(cv2.vconcat)로도 이미지를 연결할 수 있다.
'''