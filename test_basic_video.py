# test_basic_video.py - 기본 비디오 재생 테스트

import cv2
import os
import time
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_video_playback():
    """기본 비디오 재생 테스트"""
    video_files = [
        'C:/Users/HKIT/Videos/VS1/SGA5100180/SGA5100180S0097/video/SGA5100180S0097.mp4',
        'C:/Users/HKIT/Videos/VS1/SGA5100180/SGA5100180S0126/video/SGA5100180S0126.mp4',
        'C:/Users/HKIT/Videos/VS1/SGA5100180/SGA5100180S0135/video/SGA5100180S0135.mp4',
        'C:/Users/HKIT/Videos/VS1/SGA5100180/SGA5100180S0086/video/SGA5100180S0086.mp4',
        'C:/Users/HKIT/Videos/VS1/SGA5100180/SGA5100180S0101/video/SGA5100180S0101.mp4',
        'C:/Users/HKIT/Videos/VS1/SGA5100180/SGA5100180S0109/video/SGA5100180S0109.mp4'
    ]
    
    logger.info("=== 기본 비디오 재생 테스트 시작 ===")
    
    for i, video_path in enumerate(video_files):
        logger.info(f"테스트 중인 비디오 {i+1}/{len(video_files)}: {os.path.basename(video_path)}")
        
        if not os.path.exists(video_path):
            logger.error(f"비디오 파일이 존재하지 않음: {video_path}")
            continue
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"비디오 파일을 열 수 없음: {video_path}")
            continue
        
        # 비디오 정보 출력
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"비디오 정보: {width}x{height}, {fps:.2f}fps, {frame_count}프레임")
        
        # OpenCV 창 생성 테스트
        window_name = f"테스트 비디오 {i+1}"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        frame_num = 0
        max_frames_to_test = min(300, frame_count)  # 최대 300프레임 또는 전체 프레임 수
        start_time = time.time()
        max_test_duration = 30.0  # 최대 30초 테스트
        
        logger.info(f"최대 {max_frames_to_test}프레임 또는 {max_test_duration}초 동안 테스트")
        
        while frame_num < max_frames_to_test:
            # 타임아웃 체크
            if time.time() - start_time > max_test_duration:
                logger.info(f"테스트 시간 초과 ({max_test_duration}초) - 다음 비디오로 이동")
                break
            
            ret, frame = cap.read()
            if not ret:
                logger.info(f"비디오 {i+1} 재생 완료 (프레임 {frame_num})")
                break
            
            frame_num += 1
            
            # 기본 정보 오버레이
            cv2.putText(frame, f"Video {i+1}/{len(video_files)} - Frame {frame_num}/{max_frames_to_test}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"File: {os.path.basename(video_path)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'n' for next video, 'q' to quit", 
                       (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            elapsed_time = time.time() - start_time
            remaining_time = max_test_duration - elapsed_time
            cv2.putText(frame, f"Time: {elapsed_time:.1f}s / {max_test_duration}s", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(int(1000/fps)) & 0xFF
            if key == ord('q'):
                logger.info("사용자가 테스트 종료")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                logger.info("다음 비디오로 이동")
                break
        
        cap.release()
        cv2.destroyWindow(window_name)
    
    logger.info("=== 모든 비디오 테스트 완료 ===")
    cv2.destroyAllWindows()

def test_opencv_installation():
    """OpenCV 설치 상태 테스트"""
    logger.info("=== OpenCV 설치 상태 확인 ===")
    
    try:
        logger.info(f"OpenCV 버전: {cv2.__version__}")
        
        # 기본 창 생성 테스트
        test_image = cv2.imread("test_image.jpg")
        if test_image is None:
            # 테스트 이미지 생성
            import numpy as np
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(test_image, "OpenCV Test", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        cv2.imshow("OpenCV Installation Test", test_image)
        logger.info("OpenCV 창 표시 테스트 - 아무 키나 누르세요")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        logger.info("OpenCV 설치 상태: 정상")
        
    except Exception as e:
        logger.error(f"OpenCV 설치 문제: {e}")

if __name__ == "__main__":
    logger.info("DMS 비디오 재생 진단 도구 시작")
    
    # OpenCV 설치 상태 확인
    test_opencv_installation()
    
    # 기본 비디오 재생 테스트
    test_video_playback()
    
    logger.info("진단 완료")
