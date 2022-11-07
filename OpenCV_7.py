# OpenCV 강좌 : 제 7강 - 확대 & 축소

# 이미지 피라미드(Image Pytamid)

'''
입력 이미지는 항상 동일한 크기가 아니며 작거나 너무 클 수도 있다.
만약 알고리즘에서 요구하는 해상도가 있다면 입력 이미지의 크기를 변경해 영상 처리를 진행해야 함.
또한, 검출하려는 객체가 너무 작거나 입력 이미지가 너무 큰 경우 입력 이미지 자체를 변환해서 영상 처리를 진행할 수도 있다.
이미지 확대와 축소는 이미지 피라미드(Image pyramid)를 활용해 이미지의 크기를 원하는 단계까지 샘플링하는 작업이다.
이미지 피라미드의 의미는 이미지의 크기를 확대하거나 축소했을 때 이미지들의 형태가 피라미드와 같이 표현됨.
원본 이미지에서 크기를 확대하는 것을 업 샘플링이라 하며 하위 단계의 이미지를 생성하게 됨
반대로 원본 이미지에서 크기를 축소하는 것을 다운 샘플링이라 하며, 상위 단계의 이미지를 생성하게 됨
이미지 피라미드로는 가우시안 피라미드(Gaussian Pyramid)와 라플라시안 피라미드(Laplacian pyramid)를 활용함
'''

# 메인 코드

import os

os.chdir(r'C:\Users\USER\Desktop\vs코드\wkit_vs\OpenCV_study')
import cv2

src = cv2.imread('./data/test.jpg', cv2.IMREAD_COLOR)
height, width, channel = src.shape

dst = cv2.pyrUp(src, dstsize=(width * 2, height * 2), borderType=cv2.BORDER_DEFAULT)
dst2 = cv2.pyrDown(src)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()


# 세부 코드

# height, width, channel = src.shape

'''
height, width, channel = src.shape를 이용하여 해당 이미지의 높이, 너비, 채널의 값을 저장함
너비와 높이를 이용하여 출력 이미지 크기를 설정할 수 있다.
'''

# dst = cv2.pyrUp(src, dstsize=(width * 2, height *2), borderType=cv2.BORDER_DEFAULT)
'''
이미지 확대 함수(cv2.pyrUP)로 이미지를 2배 확대할 수 있다.
dst = cv2.pyrUP(src, dstSize, borderType)는 
입력 이미지(src), 출력 이미지 크기(dstSize), 테두리 외삽법(borderType)으로
출력 이미지(dst)을 생성함

출력 이미지 크기(dstSize)는 매우 세밀한 크기 조정을 필요로 할때 사용함

다음과 같은 조건을 만족하는 출력 이미지 크기만 사용할 수 있다.
|dstSize,width - src.cols*2| <= (dstSize.width mod 2)
|dstSize.height - src.rows*2| <= (dstSize.height mod 2)

테두리 외삽법(borderType)은 이미지를 확대하거나 축소할 경우, 이미지 영역 밖의 픽셀은 추정해 값을 할당해야 함
그러므로, 테두리 외삽법은 이미지 밖의 픽셀을 외삽하는 데 사용되는 테두리 모드로, 외삽 방식을 설정함
'''

# dst2 = cv2.pyrDown(src)

'''
이미지 축소 함수(cv2.pyrUp)로 이미지를 2배 축소할 수 있다.
dst = cv2.pyrDown(src, dstSize, borderType)는
입력 이미지(src), 출력 이미지 크기(dstSize), 테두리 외삽법(borderType)으로
출력 이미지(dst)을 생성함

출력 이미지 크기(dstSize)는 이미지 확대 함수에서 사용되는 방법과 동일하며,
다음과 같은 조건을 만족하는 출력 이미지 크기만 사용할 수 있다.
|dstSize.width*2 - src.cols| <= 2
|dstSize.height*2 - src.rows| <= 2
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

# 함수 추가 정보
'''
이미지 확대 함수는 BORDER_DEFAULT의 픽셀 외삽법만 사용할 수 있다.
이미지 축소 함수는 BORDER_CONSTANT의 픽셀 외삽법을 제외한 나머지 플래그만 사용할 수 있다.
'''