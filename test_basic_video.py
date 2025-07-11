# test_basic_video.py - 기본 비디오 재생 테스트 (크로스 플랫폼)

import cv2
import os
import time
import logging
import argparse
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_webcam(device_id=0):
    """웹캠 기본 기능 테스트"""
    logger.info(f"=== 웹캠 테스트 시작 (Device ID: {device_id}) ===")
    
    try:
        cap = cv2.VideoCapture(device_id)
        
        if not cap.isOpened():
            logger.error(f"웹캠을 열 수 없음: Device ID {device_id}")
            return False
        
        # 웹캠 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 웹캠 정보 출력
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"웹캠 정보: {width}x{height}, {fps:.2f}fps")
        
        window_name = "웹캠 기본 테스트"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        frame_count = 0
        start_time = time.time()
        
        logger.info("웹캠 테스트 중... 'q'를 눌러 종료하세요")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("웹캠에서 프레임을 읽을 수 없음")
                break
            
            frame_count += 1
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # 기본 정보 오버레이
            cv2.putText(frame, f"Webcam Test - Device {device_id}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count} | FPS: {current_fps:.1f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Resolution: {width}x{height}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit, 's' to save frame", 
                       (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("사용자가 웹캠 테스트 종료")
                break
            elif key == ord('s'):
                # 프레임 저장
                filename = f"webcam_test_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                logger.info(f"프레임 저장됨: {filename}")
        
        cap.release()
        cv2.destroyWindow(window_name)
        logger.info("웹캠 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"웹캠 테스트 중 오류: {e}")
        return False

def test_video_files(video_paths):
    """비디오 파일들 재생 테스트"""
    if not video_paths:
        logger.info("테스트할 비디오 파일이 없습니다.")
        return
        
    logger.info(f"=== 비디오 파일 테스트 시작 ({len(video_paths)}개 파일) ===")
    
    for i, video_path in enumerate(video_paths):
        logger.info(f"테스트 중인 비디오 {i+1}/{len(video_paths)}: {os.path.basename(video_path)}")
        
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
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"비디오 정보: {width}x{height}, {fps:.2f}fps, {frame_count}프레임, {duration:.1f}초")
        
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
            cv2.putText(frame, f"Video {i+1}/{len(video_paths)} - Frame {frame_num}/{max_frames_to_test}", 
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
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "OpenCV Test", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.putText(test_image, f"Version: {cv2.__version__}", (200, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(test_image, "Press any key to continue", (200, 320), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow("OpenCV Installation Test", test_image)
        logger.info("OpenCV 창 표시 테스트 - 아무 키나 누르세요")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        logger.info("OpenCV 설치 상태: 정상")
        
    except Exception as e:
        logger.error(f"OpenCV 설치 문제: {e}")

def find_available_cameras():
    """사용 가능한 웹캠 장치 찾기"""
    logger.info("=== 사용 가능한 카메라 장치 검색 ===")
    available_cameras = []
    
    # 0-9번 장치 확인
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append(i)
                logger.info(f"✅ 카메라 장치 {i} 사용 가능")
            cap.release()
    
    if not available_cameras:
        logger.warning("사용 가능한 카메라 장치를 찾을 수 없습니다.")
    else:
        logger.info(f"총 {len(available_cameras)}개의 카메라 장치 발견: {available_cameras}")
    
    return available_cameras

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DMS 기본 비디오 테스트 도구")
    parser.add_argument("--webcam", "-w", type=int, default=None, help="웹캠 장치 ID (예: 0, 1, 2)")
    parser.add_argument("--videos", "-v", nargs="+", help="테스트할 비디오 파일 경로들")
    parser.add_argument("--find-cameras", "-f", action="store_true", help="사용 가능한 카메라 장치 검색")
    parser.add_argument("--opencv-test", "-o", action="store_true", help="OpenCV 설치 테스트만 실행")
    
    args = parser.parse_args()
    
    logger.info("DMS 기본 비디오 테스트 도구 시작")
    
    # OpenCV 설치 상태 확인
    if args.opencv_test:
        test_opencv_installation()
        exit(0)
    
    # 카메라 검색
    if args.find_cameras:
        find_available_cameras()
        exit(0)
    
    # 기본 테스트 실행
    test_opencv_installation()
    
    # 웹캠 테스트
    if args.webcam is not None:
        test_webcam(args.webcam)
    elif args.videos:
        # 비디오 파일 테스트
        test_video_files(args.videos)
    else:
        # 기본: 사용 가능한 카메라 찾아서 테스트
        available_cameras = find_available_cameras()
        if available_cameras:
            logger.info(f"첫 번째 사용 가능한 카메라로 테스트: {available_cameras[0]}")
            test_webcam(available_cameras[0])
        else:
            logger.info("웹캠을 찾을 수 없어 기본 OpenCV 테스트만 실행됩니다.")
    
    logger.info("테스트 완료")
