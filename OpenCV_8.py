# OpenCV 강좌 : 제 8강 - 크기 조절

# 크기 조절(Resize)

'''
영상이나 이미지의 이미지를 확대하거나 축소하는 연산에서 확인할 수 있듯이 이미지의 크기를 변형하는 것은 단순한 연산이 아님
이미지를 확대하는 경우에는 픽셀에 대한 보간법,
이미지를 축소하는 경우에는 픽셀에 대한 병합법이 수행됨
이미지 피라미드는 2배로 확대하거나 축소하는 경우만 가능하므로, 원하는 크기로 변환하기 위해서 이미지 크기 조절 함수를 사용함
이미지 크기를 조절하는 방법은 크게 두가지 방법
첫 번째 방법은 이미지의 크기를 사용자가 요구하는 절대 크기로 변경하는 방법
즉, 임의의 크기(640*480 이나 123*456 등의 이미지 크기)로 변환하는 것을 의미
두 번째 방법은 이미지의 크기를 비율에 맞게 상대 크기로 변경하는 방법
입력 이미지의 크기와 비례하도록 너비와 높이가 계산됨
'''

# 메인 코드

import os

os.chdir(r'C:\Users\USER\Desktop\vs코드\wkit_vs\OpenCV_study')
import cv2

src = cv2.imread('./data/test.jpg', cv2.IMREAD_COLOR)

dst = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)
dst2 = cv2.resize(src, dsize=(0,0), fx=0.3, fy=0.7, interpolation=cv2.INTER_LINEAR)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()

# 세부 코드

# dst = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)
# dst2 = cv2.resize(src, dsize=(0, 0), fx=0.3, fy=0.7, interpolation=cv2.INTER_LINEAR)

'''
이미지 크기 조절 함수(cv2.resize)로 이미지의 크기를 변경할 수 있다.
dst = cv2.resize(src, dstSize, fx, fy, interpolation)는
입력 이미지(src), 절대 크기(dstSize), 상대 크기(fx, fy), 보간법(interpolation)으로
출력 이미지(dst)을 생성함
절대 크기는 튜플(Tuple) 형식으로 (너비, 높이)의 값을 할당해 사용함
절대 크기는 다음과 같은 수식을 통해 계산됨
dsize.width = round(fx * src.cols)
dsize.height = round(fy * src.rows)
상대 크기는 절대 크기에 (0,0)을 할당한 다음, 상대 크기의 값을 할당해 사용함
절대 크기에 (0,0)을 할당하는 이유로는 fx와 fy에서 계산된 크기가 dsize에 할당되기 때문
그러므로, 상대 크기로 이미지를 변경하기 위해서는 절대 크기에 0의 값을 할당해 사용함
상대 크기는 다음과 같은 수식을 통해 계산됨
fx = dsize.width/src.cols
fy = dsize.height/src.rows
보간법은 이미지의 크기를 변경하는 경우, 변형된 이미지의 픽셀은 추정해서 값을 할당해야 함
이미지의 비율을 변경하면 존재하지 않는 영역에 새로운 픽셀값을 매핑하거나 존재하는 픽세들을 압축해서 새로운 값을 할당해야 함
이를 이미지 상에 존재하는 픽셀 데이터(x_i,y_i)들에 대해 근사 함수 f(x,y)를 적용해서 새로운 픽셀 값을 구하는 것을 이해할 수 있다.
그러므로, 추정해야 하는 픽셀은 보간법을 이용하여 픽셀들의 값을 할당함
'''

# 추가 정보
'''
- interpolation 속성
cv2.INTER_NEAREST 이웃 보간법
cv2.INTER_LINEAR 쌍 선형 보간법
cv2.INTER_LINEAR_EXACT 비트 쌍 선형 보간법
cv2.INTER_CUBIC 바이큐빅 보간법
cv2.INTER_AREA 영역 보간법
cv2.INTER_LANCZOS4 Lanszos 보간법

일반적으로 쌍 선형 보간법이 가장 많이 사용됨
이미지를 확대하는 경우, 바이큐빅 보간법이나 쌍 선형 보간법을 가장 많이 사용
이미지를 축소하는 경우, 영역 보간법을 가장 많이 사용함
영역 보간법에서 이미지를 확대하는 경우, 이웃 보간법과 비슷한 결과를 반환함
'''