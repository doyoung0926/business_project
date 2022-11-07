# OpenCV 강좌 : 제 14강 - 가장자리 검출

# 가장자리 검출(Edge)

'''
가장자리(Edge)는 가장 바깥 부분의 둘레를 의미하며, 객체의 테두리로 볼 수 있다.
이미지 상에서 가장자리는 전경(Foreground)과 배경(Background)이 구분되는 지점이며,
전경과 배경 사이에서 밝기가 큰 폭으로 변하는 지점이 객체의 가장자리가 됨
그러므로 가장자리는 픽셀의 밝기가 급격하게 변하는 부분으로 간주할 수 있다.
가장자리를 찾기 위해 미분(Derivative)과 기울기(Gradient) 연산을 수행하며,
이미지 상에서 픽셀의 밝기 변화율이 높은 경계선을 찾는다.
'''

# 메인 코드
import os

os.chdir(r'C:\Users\USER\Desktop\vs코드\wkit_vs\OpenCV_study')

import cv2

src = cv2.imread('./data/test.jpg', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
canny = cv2.Canny(src, 100, 255)

cv2.imshow('sobel', sobel)
cv2.imshow('laplacian', laplacian)
cv2.imshow('canny', canny)
cv2.waitKey()
cv2.destroyAllWindows()

# 세부 코드 (Sobel)

# sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)

'''
소벨 함수(cv2.Sobel)로 입력 이미지에서 가장자리를 검출할 수 있다.
미분 값을 구할 때 가장 많이 사용되는 연산자이며,
인접한 픽셀들의 차이로 기울기(Gradient)의 크기를 구한다.
이때 인접한 픽셀들의 기울기를 계산하기 위해 컨벌루션 연산을 수행함.
dst = cv2.Sobel(src,ddepth,dx,dy,ksize,scale,delta,borderType)은
입력 이미지(src)에 출력 이미지 정밀도(ddepth)를 설정하고
dx(X 방향 미분 차수), dy(Y 방향 미분 차수), 커널 크기(ksize),
비율(scale), 오프셋(delta), 테두리 외삽법(borderType)을
설정하여 결과 이미지(dst)를 반환함
출력 이미지 정밀도는 반환되는 결과 이미지의 정밀도를 설정함
X 방향 미분 차수는 이미지에서 X 방향으로 미분할 차수를 설정함
Y 방향 미분 차수는 이미지에서 Y 방향으로 미분할 차수를 설정함
커널 크기는 소벨 마스크의 크기를 설정함
1, 3, 5, 7 등의 홀수 값을 사용하며, 최대 31까지 설정할 수 있다.
비율과 오프셋은 출력 이미지를 반환하기 전에 적용되며,
주로 시각적으로 확인하기 위해 사용함
픽셀 외삽법은 이미지 가장자리 부분의 처리 방식을 설정함
Tip: X 방향 미분 차수와 Y 방향 미분 차수는 합이 1 이상이여야 하며,
0의 값은 해당 방향으로 미분하지 않음을 의미함
'''

# 세부 코드(Laplacian)

# canny = cv2.Canny(src, 100, 255)

'''
캐니 함수(cv2.Canny)로 입력 이미지에서 가장자리를 검출할 수 있다.
캐니 엣지는 라플라스 필터 방식을 개선한 방식으로 x와 y에 대해 1차 미분을 계산한 다음, 네 방향으로 미분함
네 방향으로 미분한 결과로 극댓값을 갖는 지점들이 가장자리가 됨
앞서 설명한 가장자리 검출기보다 성능이 월등히 좋으며 노이즈에 민감하지 않아
강한 가장자리를 검출하는 데 목적을 둔 알고리즘이다.

dst = cv2.Canny(src, threshold1, threshold2, apertureSize, L2gradient)는
입력 이미지(src)를 하위 임곗값(threshold1), 상위 임곗값(threshold2),
소벨 연산자 마스크 크기(apertureSize), L2 그레이디언트(L2gradient)을 설정하여
결과 이미지(dst)를 반환함

하위 임곗값과 상위 임곗값으로 픽셀이 갖는 최솟값과 최댓값을 설정해 검출을 진행함
픽셀이 상위 임곗값보다 큰 기울기를 가지면 픽셀을 가장자리로 간주하고,
하위 임곗값보다 낮은 경우 가장자리로 고려하지 않는다.
소벨 연산자 마스크 크기는 소벨 연산을 활용하므로, 소벨 마스크의 크기를 설정함.
L2 그레이디언트는 L2-norm으로 방향성 그레이디언트를 정확하게 계산할지,
정확성은 떨어지지만 속도가 더 빠른 L1-norm으로 계산할지를 선택함

L1그라디언트 : L_1 = dI/dx  dI/dy
L2그라디언트 : ((dI/dx)^2 + (dI/dy)^2)^1/2
'''

# 추가 정보

'''
- 픽셀 외삽법 종류
cv2.BORDER_CONSTANT   iiiiii | abcdefgh | iiiiiii
cv2.BORDER_REPLICATE   aaaaaa | abcdefgh | hhhhhhh
cv2.BORDER_REFLECT   fedcba | abcdefgh | hgfedcb
cv2.BORDER_WRAP   cdefgh | abcdefgh | abcdefg
cv2.BORDER_REFLECT_101   gfedcb | abcdefgh | gfedcba
cv2.BORDER_REFLECT101   gfedcb | abcdefgh | gfedcba
cv2.BORDER_DEFAULT   gfedcb | abcdefgh | gfedcba
cv2.BORDER_TRANSPARENT   uvwxyz | abcdefgh | ijklmno
cv2.BORDER_ISOLATED   관심 영역 (ROI) 밖은 고려하지 않음
'''