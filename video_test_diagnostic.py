#!/usr/bin/env python3
"""
비디오 입력 시스템 진단 도구

이 스크립트는 비디오 파일 읽기 및 OpenCV 창 표시 기능을 
독립적으로 테스트하여 문제의 근본 원인을 파악합니다.
"""

import cv2
import os
import sys
import time
import logging
import threading
from typing import Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_opencv_installation():
    """OpenCV 설치 및 기본 기능 테스트"""
    logger.info("=== OpenCV 설치 테스트 ===")
    
    try:
        # OpenCV 버전 확인
        cv_version = cv2.__version__
        logger.info(f"OpenCV 버전: {cv_version}")
        
        # 지원되는 백엔드 확인
        backends = []
        backend_names = {
            cv2.CAP_FFMPEG: "FFMPEG",
            cv2.CAP_DSHOW: "DirectShow", 
            cv2.CAP_MSMF: "MediaFoundation",
            cv2.CAP_VFW: "Video for Windows"
        }
        
        for backend_id, name in backend_names.items():
            try:
                cap = cv2.VideoCapture()
                if cap.open(0, backend_id):
                    backends.append(name)
                    cap.release()
            except:
                pass
        
        logger.info(f"지원되는 백엔드: {', '.join(backends) if backends else '없음'}")
        
        # 기본 창 생성 테스트
        test_img = cv2.imread("test_image.jpg")
        if test_img is None:
            # 테스트 이미지 생성
            import numpy as np
            test_img = np.zeros((200, 300, 3), dtype=np.uint8)
            cv2.putText(test_img, "OpenCV Test", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.namedWindow("OpenCV Test", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("OpenCV Test", test_img)
        
        logger.info("OpenCV 창 생성 테스트 - 2초 후 자동 닫힘")
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        
        logger.info("✅ OpenCV 기본 기능 정상")
        return True
        
    except Exception as e:
        logger.error(f"❌ OpenCV 테스트 실패: {e}")
        return False


def test_video_file_properties(video_path: str):
    """비디오 파일 속성 상세 분석"""
    logger.info(f"=== 비디오 파일 분석: {os.path.basename(video_path)} ===")
    
    try:
        # 파일 존재 및 크기 확인
        if not os.path.exists(video_path):
            logger.error(f"❌ 파일이 존재하지 않음: {video_path}")
            return False
            
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        logger.info(f"파일 크기: {file_size:.2f} MB")
        
        # 다양한 백엔드로 열기 시도
        backends_to_test = [
            (cv2.CAP_FFMPEG, "FFMPEG"),
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "MediaFoundation"),
            (-1, "Default")  # 기본 백엔드
        ]
        
        successful_backend = None
        
        for backend_id, backend_name in backends_to_test:
            logger.info(f"--- {backend_name} 백엔드 테스트 ---")
            
            try:
                if backend_id == -1:
                    cap = cv2.VideoCapture(video_path)
                else:
                    cap = cv2.VideoCapture(video_path, backend_id)
                
                if cap.isOpened():
                    # 비디오 속성 읽기
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    logger.info(f"해상도: {width}x{height}")
                    logger.info(f"FPS: {fps:.2f}")
                    logger.info(f"총 프레임: {frame_count}")
                    logger.info(f"재생시간: {duration:.1f}초")
                    
                    # 코덱 정보
                    try:
                        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                        logger.info(f"코덱: {codec}")
                    except:
                        logger.info("코덱 정보 읽기 실패")
                    
                    # 첫 번째 프레임 읽기 테스트
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        logger.info(f"✅ 첫 프레임 읽기 성공: {frame.shape}")
                        successful_backend = (backend_id, backend_name, cap)
                        break
                    else:
                        logger.warning(f"❌ 첫 프레임 읽기 실패")
                        cap.release()
                else:
                    logger.warning(f"❌ {backend_name} 백엔드로 열기 실패")
                    
            except Exception as e:
                logger.warning(f"❌ {backend_name} 백엔드 오류: {e}")
        
        if successful_backend:
            logger.info(f"✅ 최적 백엔드: {successful_backend[1]}")
            return successful_backend
        else:
            logger.error("❌ 모든 백엔드로 파일 열기 실패")
            return False
            
    except Exception as e:
        logger.error(f"❌ 비디오 파일 분석 실패: {e}")
        return False


def test_video_playback(video_path: str, max_frames: int = 100):
    """실제 비디오 재생 테스트"""
    logger.info(f"=== 비디오 재생 테스트: {max_frames}프레임 ===")
    
    try:
        # 최적 백엔드로 열기
        backend_info = test_video_file_properties(video_path)
        if not backend_info:
            return False
            
        backend_id, backend_name, cap = backend_info
        logger.info(f"사용 백엔드: {backend_name}")
        
        # 재생 테스트
        frame_count = 0
        successful_reads = 0
        failed_reads = 0
        
        start_time = time.time()
        
        logger.info("비디오 재생 시작 (ESC 키로 중단)")
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                failed_reads += 1
                logger.warning(f"프레임 {frame_count} 읽기 실패")
                
                if failed_reads >= 10:
                    logger.error("연속 실패 10회 - 재생 중단")
                    break
                continue
            
            successful_reads += 1
            frame_count += 1
            
            # 프레임 정보 표시
            if frame is not None:
                # 정보 오버레이
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Backend: {backend_name}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, "Press ESC to quit", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Video Test", frame)
                
                # 키 입력 확인 (ESC = 27)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC
                    logger.info("사용자가 ESC로 중단")
                    break
                    
                # 진행상황 로깅 (10프레임마다)
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    logger.info(f"프레임 {frame_count}/{max_frames} 처리됨 (평균 FPS: {fps:.1f})")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 결과 요약
        total_time = time.time() - start_time
        success_rate = successful_reads / frame_count if frame_count > 0 else 0
        avg_fps = successful_reads / total_time if total_time > 0 else 0
        
        logger.info(f"=== 재생 테스트 결과 ===")
        logger.info(f"총 처리 프레임: {frame_count}")
        logger.info(f"성공한 읽기: {successful_reads}")
        logger.info(f"실패한 읽기: {failed_reads}")
        logger.info(f"성공률: {success_rate:.1%}")
        logger.info(f"평균 FPS: {avg_fps:.1f}")
        logger.info(f"총 소요시간: {total_time:.1f}초")
        
        if success_rate > 0.9:
            logger.info("✅ 비디오 재생 테스트 성공")
            return True
        else:
            logger.warning("⚠️ 비디오 재생에 문제가 있음")
            return False
            
    except Exception as e:
        logger.error(f"❌ 비디오 재생 테스트 실패: {e}")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        return False


def test_threading_video_reader(video_path: str):
    """스레드 기반 비디오 읽기 테스트 (실제 DMS와 유사)"""
    logger.info("=== 스레드 기반 비디오 읽기 테스트 ===")
    
    class ThreadedVideoReader:
        def __init__(self, video_path):
            self.video_path = video_path
            self.cap = None
            self.current_frame = None
            self.frame_lock = threading.Lock()
            self.stopped = True
            self.thread = None
            
        def start(self):
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                logger.error("VideoCapture 열기 실패")
                return False
                
            self.stopped = False
            self.thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.thread.start()
            logger.info("리더 스레드 시작됨")
            return True
            
        def _reader_loop(self):
            frame_count = 0
            while not self.stopped:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"프레임 {frame_count} 읽기 실패 - 스레드 종료")
                    self.stopped = True
                    break
                    
                with self.frame_lock:
                    self.current_frame = frame.copy()
                    
                frame_count += 1
                if frame_count % 30 == 0:
                    logger.info(f"스레드에서 {frame_count} 프레임 처리됨")
                    
                time.sleep(1/30)  # 30 FPS 시뮬레이션
                
        def get_frame(self):
            with self.frame_lock:
                return self.current_frame
                
        def stop(self):
            self.stopped = True
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1)
            if self.cap:
                self.cap.release()
                
        def is_running(self):
            return not self.stopped and self.thread and self.thread.is_alive()
    
    try:
        reader = ThreadedVideoReader(video_path)
        
        if not reader.start():
            return False
            
        # 첫 프레임 대기
        logger.info("첫 프레임 대기 중...")
        wait_time = 0
        while wait_time < 5.0:  # 최대 5초 대기
            frame = reader.get_frame()
            if frame is not None:
                logger.info(f"✅ 첫 프레임 수신: {frame.shape}")
                break
            time.sleep(0.1)
            wait_time += 0.1
        
        if frame is None:
            logger.error("❌ 첫 프레임 수신 실패")
            reader.stop()
            return False
        
        # 30초간 프레임 읽기 테스트
        logger.info("30초간 프레임 읽기 테스트 시작")
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 30.0:
            frame = reader.get_frame()
            if frame is not None:
                frame_count += 1
                
                # 5초마다 프레임 표시
                if frame_count % 150 == 0:  # 30fps * 5초
                    cv2.putText(frame, f"Threaded Frame: {frame_count}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Threaded Video Test", frame)
                    cv2.waitKey(1)
                    logger.info(f"스레드 테스트: {frame_count} 프레임 처리됨")
            
            if not reader.is_running():
                logger.warning("리더 스레드가 중단됨")
                break
                
            time.sleep(1/30)
        
        reader.stop()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        logger.info(f"=== 스레드 테스트 결과 ===")
        logger.info(f"처리된 프레임: {frame_count}")
        logger.info(f"평균 FPS: {avg_fps:.1f}")
        logger.info(f"테스트 시간: {total_time:.1f}초")
        
        if frame_count > 100 and avg_fps > 10:
            logger.info("✅ 스레드 기반 비디오 읽기 성공")
            return True
        else:
            logger.warning("⚠️ 스레드 기반 비디오 읽기에 문제")
            return False
            
    except Exception as e:
        logger.error(f"❌ 스레드 테스트 실패: {e}")
        if 'reader' in locals():
            reader.stop()
        cv2.destroyAllWindows()
        return False


def main():
    """메인 진단 루틴"""
    logger.info("🔍 DMS 비디오 입력 시스템 진단 시작")
    logger.info("=" * 60)
    
    # 테스트할 비디오 파일 (실제 DMS에서 사용하는 파일)
    test_videos = [
        "C:/Users/HKIT/Videos/VS1/SGA5100180/SGA5100180S0097/video/SGA5100180S0097.mp4",
        "C:/Users/HKIT/Videos/VS1/SGA5100180/SGA5100180S0126/video/SGA5100180S0126.mp4"
    ]
    
    results = {}
    
    # 1. OpenCV 기본 기능 테스트
    results['opencv_basic'] = test_opencv_installation()
    
    # 2. 비디오 파일 테스트
    for video_path in test_videos:
        if os.path.exists(video_path):
            logger.info(f"\n{'='*60}")
            logger.info(f"비디오 파일 테스트: {os.path.basename(video_path)}")
            
            # 파일 속성 분석
            file_analysis = test_video_file_properties(video_path)
            results[f'file_analysis_{os.path.basename(video_path)}'] = bool(file_analysis)
            
            if file_analysis:
                # 실제 재생 테스트
                playback_result = test_video_playback(video_path, max_frames=50)
                results[f'playback_{os.path.basename(video_path)}'] = playback_result
                
                # 스레드 기반 읽기 테스트
                thread_result = test_threading_video_reader(video_path)
                results[f'threading_{os.path.basename(video_path)}'] = thread_result
            
            break  # 첫 번째 파일만 테스트
        else:
            logger.warning(f"비디오 파일이 존재하지 않음: {video_path}")
    
    # 결과 요약
    logger.info(f"\n{'='*60}")
    logger.info("🔍 진단 결과 요약")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ 성공" if result else "❌ 실패"
        logger.info(f"{test_name}: {status}")
    
    # 종합 판정
    success_count = sum(results.values())
    total_count = len(results)
    
    if success_count == total_count:
        logger.info("\n🎉 모든 테스트 통과 - 비디오 시스템 정상")
        logger.info("DMS 시스템의 다른 부분에 문제가 있을 수 있습니다.")
    elif success_count > total_count // 2:
        logger.info(f"\n⚠️ 부분적 성공 ({success_count}/{total_count}) - 일부 기능에 문제")
        logger.info("비디오 코덱이나 백엔드 호환성 문제일 수 있습니다.")
    else:
        logger.info(f"\n❌ 대부분 실패 ({success_count}/{total_count}) - 심각한 문제")
        logger.info("OpenCV 재설치나 비디오 파일 확인이 필요합니다.")
    
    logger.info("\n진단 완료. 추가 도움이 필요하면 결과를 개발자에게 전달하세요.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n사용자가 진단을 중단했습니다.")
    except Exception as e:
        logger.error(f"진단 중 예상치 못한 오류: {e}")
    finally:
        cv2.destroyAllWindows()
        logger.info("진단 도구 종료.")
