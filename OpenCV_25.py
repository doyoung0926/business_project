# OpenCV 강좌 : 제 25강 모먼트

# 모멘트(Moments)


'''
윤곽선(contour)이나 이미지(array)의 0차 모멘트부터 3차 모멘트까지 계산한느 알고리즘이다.
공간 모멘트(sparical moments), 중심 모멘트(central moments), 정규화된 중심 모멘트(normalized central moments),
질량 중심(mass center) 등을 계산할 수 있다.
'''

# 메인 코드   # 왜 안될까

import os

os.chdir(r'C:\Users\USER\Desktop\vs코드\wkit_vs\OpenCV_study')

import cv2

src = cv2.imread('./data/test.jpg')
dst = src.copy()

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

for i in contours:
    M = cv2.moments(i)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    cv2.circle(dst, (cX, cY), 3, (255,0,0), -1)
    cv2.drawContours(dst, [i], 0, (0,0,255),2)

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 세부 코드

# for i in contours:
#     M = cv2.moments(i, False)
#     cX = int(M['m10'] / M['m00'])
#     cY = int(M['m01'] / M['m00'])

'''
cv2.moments()를 활용해 윤곽선에서 모멘트를 계산한다.
cv2.moments(배열, 이진화 이미지)을 의미함.
배열은 윤곽선 검출 함수에서 반환되는 구조 또는 이미지를 사용한다.
이진화 이미지는 입력된 배열 매개변수가 이미지일 경우, 이미지의 픽셀 값들을 이진화 처리할지 결정한다.
이진화 이미지 매개변수에 참 값을 할당한다면 이미지의 픽셀 값이 0이 아닌 값은 모두 1의 값으로 변경해 모멘트를 계산함
모멘트 함수를 통해 면적, 평균, 분산 등을 간단하게 구할 수 있다.
중심점을 구하는 공식은 다음과 같다.
x바 = m_10/m_00, y바 = m_01/m_00
위의 공식을 활용해 무게 중심(중심점)을 계산할 수 있다.
'''