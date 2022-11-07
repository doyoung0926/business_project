# OpenCV 강좌 : 제 11강 - 역상

# 역상(Reverse Image)

'''
역상(Reverse Image)은 영상이나 이미지를 반전 된 색상으로 변환하기 위해서 사용함
픽셀 단위마다 비트 연산(Bitwise Operation)을 적용하는데, 그중 Not 연산을 적용함
NOT 연산은 각 자릿수의 값을 반대로 바꾸는 연산
만약 153의 값을 갖는 픽셀에 NOT 연산을 적용한다면 102의 값으로 변경됨
153은 0b10011001의 값을 가지며, 102는 0b01100110의 값을 갖는다.
즉, 10 진수의 픽셀값을 2 진수의 값으로 변경한 다음, 각 자릿수의 값을 반대로 바꾸게 됨
1은 0이 되며, 0은 1로 변경 됨
'''

# 메인 코드

import os

os.chdir(r'C:\Users\USER\Desktop\vs코드\wkit_vs\OpenCV_study')

import cv2

src = cv2.imread('./data/test.jpg', cv2.IMREAD_COLOR)
dst = cv2.bitwise_not(src)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

# 세부 코드

# dst = cv2.bitwise_not(src)
'''
NOT 연산 함수(cv2.bitwise_not)로 이미지에 NOT 연산을 적용할 수 있다.
dst = cv2.bitwise_not(src, mask)는 입력 이미지(src), 마스크(mask)로 출력 이미지(dst)을 생성
마스크는 NOT 연산을 적용할 특정 영역을 의미.
마스크 배열이 포함되어 있다면, 해당 영역만 반전 연산을 적용함

Tip: not 연산 이외에도 and, or, xor 연산이 존재함
'''