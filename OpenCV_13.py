# OpenCV 강좌 : 제 13강 - 흐림 효과

# 흐림 효과(Blur)
'''
흐림 효과(Blur)는 블러링(Blurring) 또는 스무딩(Smoothing)이라 불리며,
노이즈를 줄이거나 외부 영향을 최소화하는데 사용됨
흐림 효과는 영상이나 이미지를 번지게 하며,
해당 픽셀의 주변 값들과 비교하고 계산해서 픽셀들의 색상을 재조정합니다.
단순히 이미지를 흐리게 만드는 것뿐만 아니라
노이즈를 제거해서 연산 시 계산을 빠르고 정확하게 수행하는 데 도움을 준다.
또한, 이미지의 해상도를 변경하는 경우에도 사용되는데
이미지의 크기를 변경하면 존재하지 않는 데이터를 생성하거나
존재하는 데이터를 줄여야 하므로 샘플링된 이미지를 재구성할 때 사용된다.
'''

# 메인 코드

import os

os.chdir(r'C:\Users\USER\Desktop\vs코드\wkit_vs\OpenCV_study')

import cv2

src = cv2.imread('./data/test.jpg', cv2.IMREAD_COLOR)
dst = cv2.blur(src, (9, 9), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()


# 세부 코드
# dst = cv2.blur(src, (9, 9), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)

'''
단순 흐림 효과 함수(cv2.blur)로 입력 이미지에 흐림 효과를 적용할 수 있다.
단순 흐림 효과는 각 픽셀에 대해 커널을 적용해 모든 픽셀의 단순 평균을 구하는 연산이다.

dst = cv2.blur(src, ksize, anchor, borderType)는입력 이미지(src)를
커널 크기(ksize), 고정점(anchor), 테두리 외삽법(borderType)으로
흐림 효과를 적용한 결과 이미지(dst)를 반환한다. 

커널, 고정점 테두리 외삽법에 대한 내용은 다음과 같다.

- 커널(Kernel)

커널(kernel)은 이미지에서 (x,y)의 픽셀과 해당 픽셀 주변을 포함한 작은 크기의 공간을 의미
이 영역 각각의 특정한 수식이나 함수 등을 적용해 새로운 이미지를 얻는 알고리즘에서 사용됨
커널은 영역의 형태와 요소가 결합되는 방식을 정의하는 템플릿을 의미하기도 하며,
신호 처리 분야에서는 커널을 필터(filter)라고도 함

https://076923.github.io/assets/posts/Python/OpenCV/lecture-13/2.webp
위 이미지의 파란색 사각형 내부가 커널이 되며, 파란색 사각형 크기가 3x3이 된다.

- 고정점(Anchor Point)

고정점(Anchor Point)은 커널을 통해 컨벌루션된 값을 할당한 지점
여기서 컨벌루션(Convolution)이란
새로운 픽셀을 만들어 내기 위해 커널 크기의 화소 값을 이용해 어떤 시스템을 통과해 계산하는 것을 의미함.
커널 내에서 고정점은 하나의 지점만을 가지며,
이미지와 어떻게 정렬되는지를 나타냄
위 이미지의 빨간색 부분이 고정점이 되며,
빨간색 사각형의 위치는 파란색 사각형을 기준으로 (1,1)에 위치함

- 테두리 외삽법(Border Extrapolation)
테두리 외삽법(Border Extrapolation)은 컨벌루션을 적용할 때,
이미지 가장자리 부분의 처리 방식을 의미함
컨벌루션을 적용하면 이미지 가장자리 부분은 계산이 불가능한 데,
이 문제를 해결하기 위해 테두리의 이미지 바깥쪽에 가상의 픽셀을 만들어 처리함
가상 픽셀의 값을 0으로 처리하거나, 임의의 값을 할당하거나, 커널이 연산할 수 있는 부분부터 연산을 수행하기도 함
예를 들어 cv2.BORDER_DEFAULT는 gfedcb | abcdefgh | gfedcba 의 형태로 외삽을 진행하는데
abcdefgh는 픽셀의 값을 의미함
즉 테두리 부분이 a라면 테두리 밖의 부분은 이미지를 반사하듯이 표현되어,
gfedcb나 gfedcba의 형태로 할당됨
위 이미지의 굵은 선 바깥 부분에 대해 테두리 외삽법이 적용됨
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

