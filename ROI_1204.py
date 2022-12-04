# 마우스 이벤트 처리

import cv2
import numpy as np

img = np.zeros((384,384,3), np.uint8)
img = cv2.imread('test1.jpg')
cv2.imshow('mouse event', img)

pt_total = list()
def onMouse(event, x, y, flags, param):

    global pt_total
    global img

    if event == cv2.EVENT_LBUTTONDOWN:
        pt_total.append([x, y])

    elif event == cv2.EVENT_RBUTTONDOWN:

        pt_total = np.array(pt_total, np.int32)
        img = cv2.fillConvexPoly(img, pt_total, (0,0,0))
        cv2.imshow('mouse event', img)

        pt_total = list()


cv2.setMouseCallback('mouse event', onMouse)  # 마우스 콜백 함수를 GUI 윈도우에 등록

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cv2.destroyAllWindows()