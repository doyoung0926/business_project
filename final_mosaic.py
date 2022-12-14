# 모듈 로딩
import cv2
from yolov5facedetector.face_detector import YoloDetector

#url=r'C:\Users\USER\Desktop\aaaa.mp4'

url=r'C:\Users\USER\Desktop\vs코드\wkit_vs\study\people.mp4'



def video_save(webcam, file_name):
    '''
    영상 저장을 위한 객체 생성
    '''
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
    width = round(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = round(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fps = webcam.get(cv2.CAP_PROP_FPS) 
    out = cv2.VideoWriter(file_name, fourcc, fps, (width, height))  # 파일명 경로
    return out


def mosaic(url, yolo_type='yolov5n',  target_size=1080, gpu=0, min_face=0, conf_thres=0.3, iou_thres=0.5):
    model = YoloDetector(yolo_type=yolo_type, target_size=target_size, gpu=gpu, min_face=min_face) 
    webcam=cv2.VideoCapture(url)
    out=video_save(webcam, f"{url.split('.')[0]}_mosaic.avi")

    # 동영상 파일 열기 성공 여부 확인
    if not webcam.isOpened():
        print("Could not open webcam") 
        exit()
        
    # mosa=True일 때 영상의 기본 상태 : 모자이크
    mosa=True

    while webcam.isOpened():
        status, frame = webcam.read() 

        if frame is None: break

        bboxes, confs, points = model.predict(frame, conf_thres=conf_thres, iou_thres=iou_thres)

        if not status:
            print("Could not read frame")
            exit()

        key = cv2.waitKey(33)
        if key == 26:  # Ctrl + Z : 모자이크 켜짐
            mosa = True
        elif key == 24:  # Ctrl + X : 모자이크 꺼짐
            mosa = False


        for bbox in bboxes[0]:
            (startX, startY)=bbox[0], bbox[1]
            (endX, endY)=bbox[2], bbox[3]

            if mosa == True:
                face_region = frame[startY:endY, startX:endX]  # 관심영역(얼굴) 지정
                
                M = face_region.shape[0]
                N = face_region.shape[1]

                # 모자이크 처리
                try: 
                    face_region = cv2.resize(face_region, None, fx=0.05, fy=0.05, interpolation=cv2.INTER_AREA)
                    face_region = cv2.resize(face_region, (N, M), interpolation=cv2.INTER_AREA)
                    frame[startY:endY, startX:endX] = face_region  # 원본 이미지에 적용

                except:
                    pass

            

        # display output
        out.write(frame)   # 동영상 저장
        cv2.imshow("Mosaic Video", frame)  # 윈도우 창에 이미지를 띄움

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키 입력 받으면 윈도우 창이 종료
            break
    
    #동영상 파일을 닫고 메모리 해제
    out.release()  
    webcam.release()  

    # 모든 윈도우 창을 닫음
    cv2.destroyAllWindows() 


mosaic(url)