# 근거 있는 값 넣었지만 부자연스러움

# 모듈 로딩
import cv2
from yolov5facedetector.face_detector import YoloDetector
import time

#url=r'C:\Users\USER\Desktop\aaaa.mp4'

url=r'C:\Users\USER\Desktop\vs코드\wkit_vs\study\data\people.mp4'



def video_save(webcam, file_name):
    '''
    영상 저장을 위한 객체 생성
    '''
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = round(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = round(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = webcam.get(cv2.CAP_PROP_FPS) 
    out = cv2.VideoWriter(file_name, fourcc, fps, (width, height))  # 파일명 경로
    return fps, out


def mosaic(url, yolo_type='yolov5n',  target_size=480, gpu=0, min_face=0, conf_thres=0.3, iou_thres=0.5):
    model = YoloDetector(yolo_type=yolo_type, target_size=target_size, gpu=gpu, min_face=min_face) 
    webcam=cv2.VideoCapture(url)
    fps, out=video_save(webcam, f"{url.split('.')[0]}_mosaic.mp4")

    # 동영상 파일 열기 성공 여부 확인
    if not webcam.isOpened():
        print("Could not open webcam") 
        exit()
        
    # mosa=True일 때 영상의 기본 상태 : 모자이크
    mosa = True
    
    frame_cnt = 0

    fps_list=[1./fps] 

    while webcam.isOpened():
        status, frame = webcam.read()

        # 알고리즘 시작 지점
        start_time = time.time()

        if frame is None: break

        frame_value.append(frame)

        
        if (frame_cnt == 0) | (frame_cnt % fps_list[0] == 0):
            
            bboxes, _, _ = model.predict(frame, conf_thres=conf_thres, iou_thres=iou_thres)

        else: pass
        

    
        if not status:
            print("Could not read frame")
            exit()

        key = cv2.waitKey(1)
        if key == 26:  # Ctrl + Z : 모자이크 켜짐
            mosa = True
        elif key == 24:  # Ctrl + X : 모자이크 꺼짐
            mosa = False

        for bbox in bboxes[0]:
            (startX, startY)=bbox[0], bbox[1]
            (endX, endY)=bbox[2], bbox[3]

            if mosa == True:
                face_region = frame[startY:endY, startX:endX]  # 관심영역(얼굴) 지정
                
                # 모자이크
                '''
                cv2.GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None) -> dst

                • src: 입력 영상. 각 채널 별로 처리됨.
                • dst: 출력 영상. src와 같은 크기, 같은 타입.
                • ksize: 가우시안 커널 크기. (0, 0)을 지정하면 sigma 값에 의해 자동 결정됨
                • sigmaX: x방향 sigma.
                • sigmaY: y방향 sigma. 0이면 sigmaX와 같게 설정.
                • borderType: 가장자리 픽셀 확장 방식.
                '''
                
                frame[startY:endY, startX:endX] = cv2.GaussianBlur(face_region, ksize=(0,0), sigmaX=7)
                

        # display output
        out.write(frame)   # 동영상 저장

        cv2.imshow("Mosaic Video", frame)  # 윈도우 창에 이미지를 띄움

        # 알고리즘 종료 시점
        # print('모자이크 처리에 걸리는 시간 \n▶ FPS', int(1./(time.time() - start_time)),\
        #     '\n▶ Time:',  time.time() - start_time, '\n')
        
        if frame_cnt == 0:
            fps_list[0] = round((time.time() - start_time) / (1./fps)) + 1

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키 입력 받으면 윈도우 창이 종료
            break

        frame_cnt += 1



    #동영상 파일을 닫고 메모리 해제
    out.release()  
    webcam.release()  


    # 모든 윈도우 창을 닫음
    cv2.destroyAllWindows() 

frame_value = []   # 프레임 담을 리스트
face_box = []   # 얼굴 박스 좌표 담을 리스트

mosaic(url)