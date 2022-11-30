# 모듈 로딩
import cv2
from yolov5facedetector.face_detector import YoloDetector
import pandas as pd
import re

def mosaic(url, yolo_type='yolov5n', target_size=480, gpu=0, min_face=0, text_name='test.txt'):
    '''
    frame_num, bboxes 파일에 적는 함수
    '''
    model = YoloDetector(yolo_type=yolo_type, target_size=target_size, gpu=gpu, min_face=min_face) 
    webcam = cv2.VideoCapture(url)


    # 동영상 파일 열기 성공 여부 확인
    if not webcam.isOpened():   
        print("Could not open webcam") 
        exit()
        
    f = open(text_name, 'w')  # 파일 열기
    f.write('frame_num\tbboxes\n')  # 파일 컬럼명
    f = open(text_name, 'a')  # 파일 열기

    while webcam.isOpened():
            
        status, frame = webcam.read()

        if not status:
            print("Could not read frame")
            break
        
        bboxes, confs, points = model.predict(frame)

        for idx in range(len(bboxes[0])):
            f.write('%-3d \t' %int(webcam.get(cv2.CAP_PROP_POS_FRAMES)))
            f.write('%-22s \n' %bboxes[0][idx])

    f.close()


def saved_mosaic(video_path, text_path, sigma=35):
    '''
    frame_num, bboxes 저장된 파일로 영상 모자이크 처리
    '''
    df = pd.read_csv(text_path, sep='\t')

    # bboxes 전처리 및 정수형 변환
    for idxx in df.index:
        df.bboxes[idxx] = re.sub('[^0-9]', ' ', df.bboxes[idxx]).split()
        for i in range(len(df.bboxes[idxx])):
            df.bboxes[idxx][i] = int(df.bboxes[idxx][i])

    # frame_num 정수형 변환
    df.frame_num = df.frame_num.astype('int')

    # frame_num, bboxes 리스트 형 변환
    frame_num = df.frame_num.to_list()
    bboxes = df.bboxes.to_list()

    webcam = cv2.VideoCapture(video_path)

    # 동영상 파일 열기 성공 여부 확인
    if not webcam.isOpened():
        print("Could not open webcam") 
        exit()

    while webcam.isOpened():
            
        status, frame = webcam.read()

        if not status:
            print("Could not read frame")
            break

        for idx, num in enumerate(frame_num):
            if num == int(webcam.get(cv2.CAP_PROP_POS_FRAMES)):
                (startX, startY)=bboxes[idx][0], bboxes[idx][1]
                (endX, endY)=bboxes[idx][2], bboxes[idx][3]

                face_region = frame[startY:endY, startX:endX]  # 관심영역(얼굴) 지정

                frame[startY:endY, startX:endX] = cv2.GaussianBlur(face_region, ksize=(0,0), sigmaX=sigma)  # 모자이크

        # display output
        cv2.imshow("Mosaic Video", frame)  # 윈도우 창에 이미지를 띄움
            
        # 'q' 키 입력 받으면 윈도우 창이 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

    # 동영상 파일을 닫고 메모리 해제
    webcam.release()

    # 모든 윈도우 창을 닫음
    cv2.destroyAllWindows()

video_path = 'people.mp4'
text_path = 'test.txt'

mosaic(video_path)
saved_mosaic(video_path, text_path)