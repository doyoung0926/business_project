# 모듈 로딩
import cv2
from yolov5facedetector.face_detector import YoloDetector
import time
import math
import re
import os
from moviepy.editor import *
import ffmpeg


def video_save(webcam, file_name):
    '''
    영상 저장을 위한 객체 생성
    '''
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = round(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = webcam.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(file_name, fourcc, fps, (width, height))

    return fps, out


def write_information(f, webcam, bboxes, idx):
    '''
    영상 정보 파일쓰는 함수
    '''
    f.write('%-3d \t' %int(webcam.get(cv2.CAP_PROP_POS_FRAMES)))  # frame_num
    f.write('%-22s \t' %bboxes[0][idx])  # bounding box
    f.write("%12s \n" %time.strftime('%Y-%m-%d %H:%M:%S'))  # time


def concat(PATH, file_name, file_names):
    '''
    모자이크 영상에 원본 음성 합치는 함수
    '''
    if os.path.isfile(PATH+file_name+'.mp3'):
        
        # 모자이크 영상과 원본 음성 합치기
        videoclip = VideoFileClip(PATH+file_names)
        audioclip = AudioFileClip(PATH+file_name+'.mp3')

        videoclip.audio = audioclip
        videoclip.write_videofile(PATH+'ing_'+file_names)

    else:
        if os.path.isfile(PATH+'ing_'+file_names): os.remove(PATH+'ing_'+file_names)
        os.rename(PATH+file_names, PATH+'ing_'+file_names)


def compact_save(PATH, file_name, file_names, crf=25, outputPATH='./'):
    '''
    모자이크 용량 줄이는 함수
    '''
    # crf[quality] : 비트레이트 대신 화질 기준으로 인코딩할 때 쓰는 옵션. libx264 코덱 기준 사용 가능 범위 0-51, 0은 무손실, 디폴트는 23
    # - vsync : 비디오 동기화 방식.
    # 0, passthrough : 각 프레임은 타임스탬프와 함께 디먹서에서 먹서로 전달됩니다.
    # 1, cfr : 요청된 일정한 프레임 속도를 정확하게 달성하기 위해 프레임이 복제되고 삭제됩니다.
    # 2, vfr : 2개의 프레임이 동일한 타임스탬프를 갖지 않도록 프레임을 타임스탬프와 함께 전달하거나 드롭합니다.
    # drop : 패스스루이지만 모든 타임스탬프를 파괴하므로 muxer가 프레임 속도를 기반으로 새로운 타임스탬프를 생성합니다.
    # -1, auto : muxer 기능에 따라 1과 2 사이에서 선택합니다. 이것이 기본 방법입니다.
    ffmpeg.input(PATH+'ing_'+file_names).output(outputPATH+file_name+'_mosaic.mp4', crf=crf, vsync='vfr').run()


def mosaic(url, yolo_type='yolov5n', target_size=480, gpu=0, min_face=0, conf_thres=0.3, iou_thres=0.5, sigma=33, crf=25, outputPATH='C:/Users/USER/Desktop/', original_speed=1):
    '''
    모자이크 실행 함수
    '''
    model = YoloDetector(yolo_type=yolo_type, target_size=target_size, gpu=gpu, min_face=min_face) 
    webcam = cv2.VideoCapture(url)

    # 동영상 파일 열기 성공 여부 확인
    if not webcam.isOpened():
        print("Could not open webcam") 
        exit()
    
    # 원본 영상 주소 및 영상 이름 저장
    if '\\' in url:
        file_path = url.split('\\')
        file_name = re.sub('.mp4|.avi','',url.split('\\')[-1])
    else:
        file_path = url.split('/')
        file_name = re.sub('.mp4|.avi','',url.split('/')[-1])

    if len(file_path) == 1:
        PATH = './'
    else:
        PATH = '/'.join(file_path[:-1])+'/'

    file_names = f"{file_name}_nosound.mp4"

    fps, out = video_save(webcam, PATH+file_names)
    
    mosa = True  # mosa=True일 때 영상의 기본 상태 : 모자이크
    fps_value = 1  # 원본 fps와 맞추기 위해 나눌 수
    gauge_bar = -1   # 상황 바 초기화
    times = 0  # 중간에 정지할 시 시간 초기화

    f = open(PATH+file_name+'_mosaic.txt', 'w')  # 파일 열기
    f.write('frame_num\tbboxes\ttime\n')  # 컬럼명 쓰기
    f = open(PATH+file_name+'_mosaic.txt', 'a')  # 파일 열기

    while webcam.isOpened():
            
        status, frame = webcam.read()

        if not status:
            print("Could not read frame")
            break

        # 알고리즘 시작 지점
        start_time = time.time()

        # 프레임 드랍
        if (webcam.get(cv2.CAP_PROP_POS_FRAMES) == 1) | (webcam.get(cv2.CAP_PROP_POS_FRAMES) == 2) | (webcam.get(cv2.CAP_PROP_POS_FRAMES) % fps_value == 0):
            bboxes, confs, points = model.predict(frame, conf_thres=conf_thres, iou_thres=iou_thres)
            
            for idx in range(len(bboxes[0])):
                # 영상 정보 파일에 쓰기
                write_information(f, webcam, bboxes, idx)

        else: pass

        # 모자이크 ON & OFF 기능
        key = cv2.waitKey(1)
        if key == 26:  # Ctrl + Z : 모자이크 켜짐
            mosa = True
        elif key == 24:  # Ctrl + X : 모자이크 꺼짐
            mosa = False

        # 얼굴 모자이크
        for bbox in bboxes[0]:
            (startX, startY)=bbox[0], bbox[1]
            (endX, endY)=bbox[2], bbox[3]

            if mosa == True:
                face_region = frame[startY:endY, startX:endX]  # 관심영역(얼굴) 지정

                frame[startY:endY, startX:endX] = cv2.GaussianBlur(face_region, ksize=(0,0), sigmaX=sigma)  # 모자이크
        
        # display output
        out.write(frame)   # 동영상 저장

        cv2.putText(frame, f'{gauge_bar}%', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)  # 진행 상황 화면에 출력

        cv2.imshow("Mosaic Video", frame)  # 윈도우 창에 이미지를 띄움


        # 영상 프레임 드랍할지 안 할지 정하기
        if webcam.get(cv2.CAP_PROP_POS_FRAMES) == 2:
            if original_speed == True:
                fps_value = 1
            else:
                fps_value = math.ceil((time.time() - start_time)/(1./fps))


        # 진행 상황 바 출력
        if gauge_bar != round(webcam.get(cv2.CAP_PROP_POS_FRAMES)*100/webcam.get(cv2.CAP_PROP_FRAME_COUNT)):
            # print(' '*round(webcam.get(cv2.CAP_PROP_POS_FRAMES)*100/webcam.get(cv2.CAP_PROP_FRAME_COUNT))+'▽',\
            #     round(webcam.get(cv2.CAP_PROP_POS_FRAMES)*100/webcam.get(cv2.CAP_PROP_FRAME_COUNT)),'%')
            # print('[', end='')
            # print(round(webcam.get(cv2.CAP_PROP_POS_FRAMES)*100/webcam.get(cv2.CAP_PROP_FRAME_COUNT))*'■'+\
            #     (100-round(webcam.get(cv2.CAP_PROP_POS_FRAMES)*100/webcam.get(cv2.CAP_PROP_FRAME_COUNT)))*'□', end='')
            # print(']')
            gauge_bar = round(webcam.get(cv2.CAP_PROP_POS_FRAMES)*100/webcam.get(cv2.CAP_PROP_FRAME_COUNT))
            
        # 'q' 키 입력 받으면 윈도우 창이 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            times = round(webcam.get(cv2.CAP_PROP_POS_MSEC) // 1000)   # 중간에 끊을 시 시간 담기
            break

    f.close()   # 파일 닫기

    #동영상 파일을 닫고 메모리 해제
    out.release()
    webcam.release()

    # 모든 윈도우 창을 닫음
    cv2.destroyAllWindows()

    # 음성 파일 저장
    if os.path.isfile(PATH+file_name+'.mp3'): os.remove(PATH+file_name+'.mp3')  # 경로에 음성 파일 있으면 삭제

    try:
        if times != 0: 
            ffmpeg.input(PATH+file_name+'.mp4').output(PATH+file_name+'.mp3', t=times).run()
        else:
            ffmpeg.input(PATH+file_name+'.mp4').output(PATH+file_name+'.mp3').run()
    except:
        pass

    # 모자이크 영상, 원본 음성 합치는 함수
    try:
        concat(PATH, file_name, file_names)
    except: pass

    if os.path.isfile(outputPATH+file_name+'_mosaic.mp4'): os.remove(outputPATH+file_name+'_mosaic.mp4')  # 출력 경로에 최종 파일 있으면 삭제

    # 영상 용량 줄인 후 출력 경로로 내보내는 함수
    try:
        compact_save(PATH, file_name, file_names, crf, outputPATH)
    except: pass

    if os.path.isfile(PATH+file_name+'_mosaic.txt'): os.replace(PATH+file_name+'_mosaic.txt', outputPATH+file_name+'_mosaic.txt')  # 정보 파일 출력 경로로 이동시키기

    if os.path.isfile(PATH+file_name+'.mp3'): os.remove(PATH+file_name+'.mp3')  # 원본 음성 파일 삭제
    if os.path.isfile(PATH+file_names): os.remove(PATH+file_names)  # 압축x, 음성x 모자이크 파일 삭제
    if os.path.isfile(PATH+'ing_'+file_names): os.remove(PATH+'ing_'+file_names)  # 압축x, 음성o 모자이크 파일 삭제

url = 'people.mp4'

mosaic(url)