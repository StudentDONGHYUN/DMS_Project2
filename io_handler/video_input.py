"""
Video Input Manager - Enhanced System
통합 시스템과 호환되는 비디오 입력 관리자
기존 코드와 개선된 코드를 통합하여 호환성과 성능을 모두 확보
"""

import cv2
import asyncio
import threading
import time
import logging
import os
import contextlib
from typing import Optional, List, Dict, Any
from pathlib import Path
try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def video_capture_context(source):
    """VideoCapture 객체를 안전하게 관리하는 Context Manager"""
    cap = None
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"VideoCapture 열기 실패: {source}")
        yield cap
    except Exception as e:
        logger.error(f"VideoCapture Context Manager 오류: {e}")
        raise
    finally:
        if cap is not None:
            try:
                cap.release()
                logger.debug("VideoCapture 리소스 정리 완료")
            except Exception as e:
                logger.warning(f"VideoCapture 정리 중 오류: {e}")

@contextlib.asynccontextmanager
async def async_video_capture_context(source):
    """비동기 VideoCapture Context Manager"""
    cap = None
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Async VideoCapture 열기 실패: {source}")
        yield cap
    except Exception as e:
        logger.error(f"Async VideoCapture Context Manager 오류: {e}")
        raise
    finally:
        if cap is not None:
            try:
                cap.release()
                logger.debug("Async VideoCapture 리소스 정리 완료")
            except Exception as e:
                logger.warning(f"Async VideoCapture 정리 중 오류: {e}")


class ContinuityManager:
    """운전자 연속성 관리"""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.is_same_driver = True

    def set_driver_continuity(self, is_same: bool):
        self.is_same_driver = is_same

    def should_skip_calibration(self) -> bool:
        return self.is_same_driver

    def save_calibration_data(self, data: Dict):
        logger.debug(f"캘리브레이션 데이터 저장: {self.user_id}")

    def get_shared_calibration_data(self) -> Optional[Dict]:
        return None


class VideoInputManager:
    """통합 비디오 입력 관리자 - 기존 코드와 개선된 코드의 통합"""

    def __init__(
        self,
        source_type: str = "webcam",
        webcam_id: int = 0,
        video_files: List[str] = None,
        enable_calibration: bool = True,
        input_source=None  # Legacy compatibility
    ):
        """
        비디오 입력 관리자 초기화
        
        Args:
            source_type: 입력 소스 타입 ("webcam" 또는 "video")
            webcam_id: 웹캠 ID
            video_files: 비디오 파일 목록
            enable_calibration: 캘리브레이션 활성화 여부
            input_source: Legacy compatibility parameter
        """
        # Legacy compatibility
        if input_source is not None:
            if isinstance(input_source, int):
                self.source_type = "webcam"
                self.webcam_id = input_source
            el            elif isinstance(input_source, (str, list)):
                self.source_type = "video"
                self.video_files = [input_source] if isinstance(input_source, str) else list(input_source)
                self.webcam_id = 0
            else:
                self.source_type = source_type
                self.webcam_id = webcam_id
                self.video_files = video_files if video_files is not None else []
        else:
            self.source_type = source_type
            self.webcam_id = webcam_id
            self.video_files = video_files or []
        
        self.enable_calibration = enable_calibration
        
        # Video capture
        self.cap = None
        self.is_initialized = False
        self._is_running = False
        
        # Frame management
        self.current_frame = None
        self.frame_count = 0
        self.last_frame_time = 0.0
        self.fps = 30.0
        
        # Threading
        self.capture_thread = None
        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        
        # Video file management
        self.current_video_index = 0
        self.video_changed = False
        self.playback_speed = 1.0
        
        # Error handling
        self.last_error = None
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10

    async def start(self) -> bool:
        """비디오 입력 시작"""
        try:
            if self._is_running:
                logger.warning("Video input is already running")
                return True
            
            success = await self._initialize_capture()
            if success:
                self._is_running = True
                logger.info(f"Video input started - Type: {self.source_type}")
            
            return success
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to start video input: {e}")
            return False

    def initialize(self) -> bool:
        """동기 초기화 (레거시 호환성)"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.start())
        except Exception as e:
            logger.error(f"Sync initialization failed: {e}")
            return False

    async def _initialize_capture(self) -> bool:
        """비디오 캡처 초기화"""
        try:
            if self.source_type == "webcam":
                return await self._initialize_webcam()
            elif self.source_type == "video":
                return await self._initialize_video()
            else:
                raise ValueError(f"Unsupported source type: {self.source_type}")
                
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to initialize capture: {e}")
            return False

    async def _initialize_webcam(self) -> bool:
        """웹캠 초기화"""
        try:
            self.cap = cv2.VideoCapture(self.webcam_id)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open webcam {self.webcam_id}")
            
            # Optimize webcam settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test first frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to read first frame from webcam")
            
            self.current_frame = frame
            self.is_initialized = True
            
            # Start capture thread
            self.stop_event.clear()
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            return await self._wait_for_first_frame()
            
        except Exception as e:
            logger.error(f"Webcam initialization failed: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    async def _initialize_video(self) -> bool:
        """비디오 파일 초기화"""
        try:
            if not self.video_files:
                raise ValueError("No video files provided")
            
            # Try to open first video
            for i, video_file in enumerate(self.video_files):
                if await self._try_open_video(video_file, i):
                    self.current_video_index = i
                    break
            else:
                raise RuntimeError("Failed to open any video file")
            
            self.is_initialized = True
            
            # Start capture thread
            self.stop_event.clear()
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            return await self._wait_for_first_frame()
            
        except Exception as e:
            logger.error(f"Video initialization failed: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    async def _try_open_video(self, video_file: str, index: int) -> bool:
        """비디오 파일 열기 시도"""
        try:
            if not os.path.exists(video_file):
                logger.warning(f"Video file not found: {video_file}")
                return False
            
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                logger.warning(f"Failed to open video: {video_file}")
                return False
            
            # Test first frame
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning(f"Failed to read first frame from: {video_file}")
                cap.release()
                return False
            
            # Success - store the capture and frame
            if self.cap:
                self.cap.release()
            
            self.cap = cap
            self.current_frame = frame
            self.video_changed = True
            
            logger.info(f"Successfully opened video: {video_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening video {video_file}: {e}")
            return False

    async def _wait_for_first_frame(self, timeout: float = 5.0) -> bool:
        """첫 번째 프레임 대기"""
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                with self.frame_lock:
                    if self.current_frame is not None:
                        return True
                
                await asyncio.sleep(0.1)
            
            logger.error("Timeout waiting for first frame")
            return False
            
        except Exception as e:
            logger.error(f"Error waiting for first frame: {e}")
            return False

    def _capture_loop(self):
        """프레임 캡처 루프"""
        logger.info("Capture loop started")
        
        while not self.stop_event.is_set():
            try:
                if not self.cap or not self.cap.isOpened():
                    if self.source_type == "video" and self._try_next_video():
                        continue
                    else:
                        break
                
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    with self.frame_lock:
                        self.current_frame = frame
                        self.frame_count += 1
                        self.last_frame_time = time.time()
                    
                    self.consecutive_failures = 0
                    
                    # Frame rate control for video files
                    if self.source_type == "video":
                        sleep_time = (1.0 / self.fps) / self.playback_speed
                        time.sleep(max(0.001, sleep_time))
                
                else:
                    # End of video file or error
                    if self.source_type == "video":
                        if not self._try_next_video():
                            break
                    else:
                        self.consecutive_failures += 1
                        if self.consecutive_failures >= self.max_consecutive_failures:
                            logger.error("Too many consecutive failures, stopping capture")
                            break
                        time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_consecutive_failures:
                    break
                time.sleep(0.1)
        
        logger.info("Capture loop ended")

    def _try_next_video(self) -> bool:
        """다음 비디오 파일로 전환"""
        try:
            if not self.video_files or len(self.video_files) <= 1:
                return False
            
            self.current_video_index = (self.current_video_index + 1) % len(self.video_files)
            next_video = self.video_files[self.current_video_index]
            
            return self._try_open_video_sync(next_video, self.current_video_index)
            
        except Exception as e:
            logger.error(f"Error switching to next video: {e}")
            return False

    def _try_open_video_sync(self, video_file: str, index: int) -> bool:
        """비디오 파일 열기 (동기 버전)"""
        try:
            if not os.path.exists(video_file):
                return False
            
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                return False
            
            if self.cap:
                self.cap.release()
            
            self.cap = cap
            self.video_changed = True
            logger.info(f"Switched to video: {video_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening video sync {video_file}: {e}")
            return False

    async def get_frame_async(self) -> Optional[np.ndarray]:
        """현재 프레임 반환 (비동기 버전)"""
        try:
            with self.frame_lock:
                if self.current_frame is not None:
                    return self.current_frame.copy()
                return None
                
        except Exception as e:
            logger.error(f"Error getting frame async: {e}")
            return None

    def get_frame(self) -> Optional[np.ndarray]:
        """현재 프레임 반환 (동기 버전)"""
        try:
            with self.frame_lock:
                if self.current_frame is not None:
                    return self.current_frame.copy()
                return None
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None

    # Legacy compatibility - remove duplicate method
    # get_frame is handled by get_frame_sync for backward compatibility

    def is_active(self) -> bool:
        """비디오 입력 활성 상태 확인"""
        return self._is_running and self.cap is not None and self.cap.isOpened()

    def get_performance_info(self) -> Dict[str, Any]:
        """성능 정보 반환"""
        return {
            'frame_count': self.frame_count,
            'fps': self.fps,
            'playback_speed': self.playback_speed,
            'source_type': self.source_type,
            'is_active': self.is_active(),
            'consecutive_failures': self.consecutive_failures,
            'current_video_index': self.current_video_index if self.source_type == "video" else None
        }

    def set_playback_speed(self, speed: float):
        """재생 속도 설정"""
        self.playback_speed = max(0.1, min(5.0, speed))

    def has_video_changed(self) -> bool:
        """비디오 변경 여부 확인"""
        if self.video_changed:
            self.video_changed = False
            return True
        return False

    async def stop(self):
        """비디오 입력 중지"""
        try:
            self._is_running = False
            self.stop_event.set()
            
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            self.is_initialized = False
            logger.info("Video input stopped")
            
        except Exception as e:
            logger.error(f"Error stopping video input: {e}")

    def release(self):
        """리소스 해제 (레거시 호환성)"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.stop())
        except Exception as e:
            logger.error(f"Error in sync release: {e}")

    def get_error(self) -> Optional[str]:
        """마지막 에러 반환"""
        return self.last_error

    def __enter__(self):
        """Context manager 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.release()

    # Legacy compatibility methods
    def is_running(self) -> bool:
        """Legacy compatibility for is_running"""
        return self.is_active()

    def get_playback_info(self) -> Dict[str, Any]:
        """Legacy compatibility for playback info"""
        return self.get_performance_info()


# Legacy compatibility class
class MultiVideoCalibrationManager:
    """레거시 호환성을 위한 더미 클래스"""
    def __init__(self, *args, **kwargs):
        logger.warning("MultiVideoCalibrationManager is deprecated")
    
    def should_skip_calibration(self) -> bool:
        return False
