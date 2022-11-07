# OpenCV 강좌 : 제 12강 - 이진화

# 이진화(Binary)

'''
이진화(Binary)는 어느 지점을 기준으로 값이 높거나 낮은 픽셀의 값을 대상으로 특정 연산을 수행할 때 사용
일반적으로 값이 높거나 낮은 픽셀을 검은색 또는 흰색의 값으로 변경함
기준값에 따라 이분법적으로 구분해 픽셀을 참 또는 거짓으로 나누는 연산이며,
이미지 행렬에서 모든 픽셀에 대해 연산이 수행됨
'''

# 메인 코드

import os

os.chdir(r'C:\Users\USER\Desktop\vs코드\wkit_vs\OpenCV_study')

import cv2

src = cv2.imread('./data/test.jpg', cv2.IMREAD_COLOR)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

ret, dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

# 세부 코드

# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# ret, dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

'''
이진화 함수(cv2.threshold)로 그레이스케일 이미지에 이진화를 적용할 수 있다.
retval, dst = cv2.threshold(src, thresh, maxval, type)는
입력 이미지(src)를 임곗값 형식(type)에 따라 임곗값(thresh)과 최댓값(maxval)을 활용하여
설정 임곗값(retval)과 결과 이미지(dst)를 반환한다.
입력 이미지는 단일 채널 이미지(그레이스케일)을 입력해 사용한다.
임곗값 형식은 임곗값을 초과한 값은 최댓값으로 변경하고 임곗값 이하의 값은 0으로 바꾸는 등의 연산을 적용함
설정 임곗값은 일반적으로 임곗값과 동일하지만,
임곗값을 대신 계산해주는 알고리즘인 Ostu나 Triangle를 사용한다면,
해당 알고리즘에서 계산해준 임곗값을 알 수 있다.
예제에서는 임곗값을 100, 최댓값을 255, 임곗값 형식을 cv2.THRESH_BINARY로 사용하였으므로,
픽셀의 값이 100을 초과하는 경우에는 255의 값으로 변경되며, 100 이하의 값은 0으로 변경됨
수식으로 표현한다면 dst = (src > thresh) ? maxval : 0 으로 표현할 수 있다.

Tip: 다중 채널 이미지를 입력 이미지로 사용하였을 때, 각 채널마다 이미지를 분리해 이진화 연산을 적용함
'''

# 추가 정보

'''
- 임계값 형식
cv2.THRESH_BINARY  dst = (src > thresh) ? maxval : 0
                  (임곗값을 초과할 경우 maxval, 아닐 경우 0)
cv2.THRESH_BINARY_INV  dst = (src > thresh) ? 0: maxval
                       (임곗값을 초과할 경우 0, 아닐 경우 maxval)
cv2.THRESH_TRUNC  dst = (src > thresh) ? thresh: src
                  (임곗값을 초과할 경우 thresh, 아닐 경우 변형 없음)
cv2.THRESH_TOZERO  dst = (src > thresh) ? src: 0
                   (임곗값을 초과할 경우 0)
cv2.THRESH_TOZERO_INV  dst = (src > thresh) ? 0: src
                       (임곗값을 초과할 경우 0, 아닐 경우 변형 없음)
cv2.THRESH_MASK  검은색 이미지로 변경(마스크용)
cv2.THRESH_OTSU  오츠 알고리즘 적용(단일 채널 이미지에만 적용 가능)
cv2.THRESH_TRIANGLE  삼각형(Triangle) 알고리즘 적용(단일 채널 이미지에만 적용 가능)
'''