# 스타트

# 모듈 로딩
import pafy
import cv2
from yolov5facedetector.face_detector import YoloDetector

url = 'https://www.youtube.com/watch?v=hoRAV8Wp2UM'

# url = 'https://www.you?v=hoRAV8Wp2UM'   # 잘못된 경로

# -------------------------------------------------------------
# 기    능 : 얼굴 모자이크 함수
# 함 수 명 : mojaic
# 파라미터 : best
# 반 환 값 : 없음
# -------------------------------------------------------------
def mojaic(url,yolo_type='yolov5n',  target_size=1080, gpu=0, min_face=0):
    
    try:
        video = pafy.new(url)
        best = video.getbest(preftype='mp4')
        uurl = best.url

    except Exception as e:
        uurl = ''
        print('오류 =>\n', e)
        cv2.destroyAllWindows()


    # ---------------------------------------------------------------------------------
    # YoloDetector 파라미터 
    # gpu : gpu number (int) or -1 or string for cpu.
    # min_face : minimal face size in pixels.
    # target_size : target size of smaller image axis (choose lower for faster work). e.g. 480, 720, 1080. None for original resolution.
    # ---------------------------------------------------------------------------------
    model = YoloDetector(yolo_type=yolo_type, target_size=target_size, gpu=gpu, min_face=min_face)  # yolov5n_state_dict.pt 자동 다운 됨

    # 경로 url의 비디오 정보를 반환
    
    webcam = cv2.VideoCapture(uurl)

    #재생할 파일의 넓이와 높이
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    width = round(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = webcam.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter('mosaic_chim.avi', fourcc, fps, (width, height))

    # 동영상 파일 열기 성공 여부 확인
    if not webcam.isOpened():
        print("Could not open webcam")  # 열려있지 않으면 문자열 출력
        exit()
        
    # 비디오 매 프레임 처리
    while webcam.isOpened():

        # -------------------------------------------------------------
        # webcam.read <- 프레임 읽기 메서드
        # status : 카메라의 상태가 저장. 정상 작동 True, 비정상 작동 False 반환
        # frame : 현재 시점의 프레임(numpy.ndarray) 저장
        # -------------------------------------------------------------

        status, frame = webcam.read() 

        # -------------------------------------------------------------
        # cv2.resize <- 이미지 크기 조절 함수 파라미터
        # 입력 이미지 : frame
        # 절대 크기 : dsize 
        # 상대 크기 : fx, fy
        # interpolation : 보간법
        # ---------------------------------------------------------------------------------

        # frame=cv2.resize(frame, dsize=(0,0), fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)

        # ---------------------------------------------------------------------------------
        # cv2.cvtcolor <- 색상 공간 변환 함수 파라미터
        # 입력 이미지 : frame
        # 색상 변환 코드. 원본 이미지 색상 공간2결과 이미지 색상 공간 : cv2.COLOR_BGR2RGB
        # ---------------------------------------------------------------------------------

        # frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        bboxes, confs, points = model.predict(frame)

        if not status:
            print("Could not read frame")
            exit()

        for bbox in bboxes[0]:

            (startX, startY)=bbox[0], bbox[1]
            (endX, endY)=bbox[2], bbox[3]
        
            # ---------------------------------------------------------------------------------
            # cv2.rectangle <- 사각형 그리기 함수 파라미터
            # 입력 이미지 : frame 
            # 좌측 상단 모서리 좌표 : (startX, startY)
            # 우측 하단 모서리 좌표 : (endX, endY)
            # 색상 : (0, 255, 0)
            # 두께 : 2
            # ---------------------------------------------------------------------------------

            # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # 모자이크 처리
            face_region = frame[startY:endY, startX:endX]
            
            M = face_region.shape[0]
            N = face_region.shape[1]

            try: 
                face_region = cv2.resize(face_region, None, fx=0.05, fy=0.05, interpolation=cv2.INTER_AREA)
                face_region = cv2.resize(face_region, (N, M), interpolation=cv2.INTER_AREA)
                frame[startY:endY, startX:endX] = face_region

            except:
                pass

            

        # display output

        # frame=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        cv2.imshow("Real-time face detection", frame)  # 윈도우 창에 이미지를 띄움

        

        # cv2.waitkey 키 입력 대기 함수
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키 입력 받으면 윈도우 창이 종료
            break
        
    out.release()
    webcam.release()  # 동영상 파일을 닫고 메모리 해제
    cv2.destroyAllWindows()   # 모든 윈도우 창을 닫음


mojaic(url) 