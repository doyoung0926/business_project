# OpenCV 강좌 : 제 9강 - 자르기

# 자르기(Slice)

'''
자르기(Slice)는 영상이나 이미지에서 특정 영역을 잘라내는 연산을 의미함
특정 영역을 잘라내는 것을 관심 영역(Region Of Interest, ROI)이라 하여,
이미지 상에서 관심 있는 영역을 의미함
이미지를 처리할 때 객체를 탐지하거나 검출하는 영역을 명확하게 관심 영역이라 볼 수 있다.
관심 영역에만 알고리즘을 적용한다면, 불필요한 연산이 줄어들고 정확도가 늘어나는 효과를 얻을 수 있다.
'''

# # 메인 코드 (1)

# import os

# os.chdir(r'C:\Users\USER\Desktop\vs코드\wkit_vs\OpenCV_study')
# import cv2

# src = cv2.imread('./data/test.jpg', cv2.IMREAD_COLOR)
# dst = src[100:600, 200:700].copy()

# cv2.imshow('src', src)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 세부 코드

# dst = src[100:600, 200:700].copy()
'''
OpenCV의 이미지는 numpy 배열 형식과 동일
src 이미지에 src[높이(행), 너비(열)]로 관심 영역을 설정
리스트(List)나 배열(Array)의 특정 영역을 자르는 방식과 동일
이미지를 자르거나 복사할 때, dst = src의 형태로 사용할 경우,
얕은 복사(shallow copy)가 되어 원본도 영향을 받게 됨
그러므로, *.copy()를 이용해 깊은 복사(deep copy)를 진행함
'''

# 메인 코드 (2)  # 왜 오류가 뜰까요?

import os

os.chdir(r'C:\Users\USER\Desktop\vs코드\wkit_vs\OpenCV_study')
import cv2

src = cv2.imread('./data/test.jpg', cv2.IMREAD_COLOR)   

# print(src.shape)      #
dst = src.copy() 
roi = src[100:400, 150:450]   # src.shape 넘게 잡으면 안됨!!
dst[0:300, 0:300] = roi

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

# 세부 코드

# dst = src.copy()

'''
dst 이미지를 생성할 때, dst = src.copy()가 아닌 dst = src를 적용한다면 깊은 복사가 적용되지 않습니다.
얕은 복사로 이미지를 복사할 경우, dst 이미지와 src 이미지는 동일한 결과로 반환됨.
'''

# roi = src[100:600, 200:700]
# dst[0:500, 0:500] = roi

'''
roi 이미지를 생성하여 src[높이(행), 너비(열)]로 관심 영역을 설정함
이후, dst[높이(행), 너비(열)] = roi를 이용하여 dst 이미지에 해당 영역을 붙여넣을 수 있다.
잘라낸 이미지와 붙여넣을 이미지의 크기가 다르다면 이미지를 붙여넣을 수 없다.
'''