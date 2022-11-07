# OpenCV 강좌 : 제 23강 - 코너 검출

# 코너 검출(Good Features To Track)

'''
영상이나 이미지에서 코너를 검출하는 알고리즘이다.
코너 검출 알고리즘은 정확하게는 트래킹(Tracking) 하기 좋은 지점(특징)을 코너라 부른다.
꼭짓점은 트래킹하기 좋은 지점이 되어 다각형이나 객체의 꼭짓점을 검출하는 데 사용한다.
'''
 
# 메인 코드  #  왜 안돼

import os

os.chdir(r'C:\Users\USER\Desktop\vs코드\wkit_vs\OpenCV_study')

import cv2

src = cv2.imread('./data/test2.jpg')
dst = src.copy()

gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
corners = cv2.goodFeaturesToTrack(gray, 100,0.01,5,blockSize=3, useHarrisDetector=True, k=0.03)

for i in corners:
    cv2.circle(dst, tuple(i[0]), 3, (0,0,255),2)

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 세부 코드

# corners = cv2.goodFeaturesToTrack(gray, 100,0.01,5,blockSize=3, useHarrisDetector=True, k=0.03)

'''
cv2.goodFeaturesToTrack()를 활용해 윤곽선들의 이미지에서 코너를 검출한다.
cv2.goodFeaturesToTrack(입력 이미지, 코너 최댓값, 코너 품질, 최소 거리, 마스크, 블록 크기, 해리스 코너 검출기 유/무, 해리스 코너 계수)을 의미함
입력 이미지는 8비트 또는 32비트의 단일 채널 이미지를 사용함
코너 최댓값은 검출할 최대 코너의 수를 제한함
코너 최댓값보다 낮은 개수만 반환함
코너 품질은 반환할 코너의 최소 품질을 설정함
코너 품질은 0.0 ~ 1.0 사이의 값으로 할당할 수 있으며,
일반적으로 0.01 ~ 0.10 사이의 값을 사용한다.
최소 거리는 검출된 코너들의 최소 근접 거리를 나타내며, 설정된 최소 거리 이상의 값만 검출한다.
마스크는 입력 이미지와 같은 차원을 사용하며, 마스크 요솟값이 0인 곳은 코너로 계산하지 않는다.
블록 크기는 코너를 계산할 때, 고려하는 코너 주변 영역의 크기를 의미함
해리스 코너 검출기 유/무는 해리스 코너 검출 방법 사용 여부를 설정함
해리스 코너 계수는 해리스 알고리즘을 사용할 때 할당하며 해리스 대각합의 감도 계수를 의미함

Tip: 코너 품질에서 가장 좋은 코너의 강도가 1000이고,
코너 품질이 0.01이라면 10 이하의 코너 강도를 갖는 코너들은 검출하지 않는다.
Tip: 최소 거리의 값이 5일 경우, 거리가 5 이하인 코너점은 검출하지 않는다.
'''

# for i in corners:
#     cv2.circle(dst, tuple(i[0]), 3, (0,0,255),2)

'''
코너 검출 함수를 통해 corners가 반환되며, 이 배열안에 코너들의 좌표가 저장돼 있다.
반복문을 활용해 dst에 빨간색 원으로 지점을 표시함
'''