"""
Video Input Manager v2
통합 시스템과 호환되는 비디오 입력 관리자
기존 코드와 개선된 코드를 통합하여 호환성과 성능을 모두 확보
"""

import cv2
import asyncio
import threading
import time
import logging
import os
from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class VideoInputManager:
    """통합 비디오 입력 관리자 - 기존 코드와 개선된 코드의 통합"""

    def __init__(
        self,
        source_type: str = "webcam",
        webcam_id: int = 0,
        video_files: List[str] = None,
        enable_calibration: bool = True
    ):
        """
        비디오 입력 관리자 초기화
        
        Args:
            source_type: 입력 소스 타입 ("webcam" 또는 "video")
            webcam_id: 웹캠 ID
            video_files: 비디오 파일 목록
            enable_calibration: 캘리브레이션 활성화 여부
        """
        self.source_type = source_type
        self.webcam_id = webcam_id
        self.video_files = video_files or []
        self.enable_calibration = enable_calibration
        
        # Video capture
        self.cap = None
        self.is_initialized = False
        self.is_running = False
        
        # Frame management
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        
        # Video playback
        self.current_video_index = 0
        self.playback_speed = 1.0
        self.video_changed = False
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 30.0
        
        # Error handling
        self.last_error = None
        
        logger.info(f"Video Input Manager initialized - Type: {source_type}")

    async def start(self) -> bool:
        """비디오 입력 시작"""
        try:
            logger.info("Starting Video Input Manager...")
            
            # Initialize video capture
            if not await self._initialize_capture():
                return False
            
            # Start capture thread
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            # Wait for first frame
            if not await self._wait_for_first_frame():
                return False
            
            self.is_initialized = True
            logger.info("Video Input Manager started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Video Input Manager startup failed: {e}")
            self.last_error = str(e)
            return False

    async def _initialize_capture(self) -> bool:
        """비디오 캡처 초기화"""
        try:
            if self.source_type == "webcam":
                return await self._initialize_webcam()
            elif self.source_type == "video":
                return await self._initialize_video()
            else:
                logger.error(f"Unsupported source type: {self.source_type}")
                return False
                
        except Exception as e:
            logger.error(f"Capture initialization failed: {e}")
            return False

    async def _initialize_webcam(self) -> bool:
        """웹캠 초기화"""
        try:
            logger.info(f"Initializing webcam {self.webcam_id}")
            
            self.cap = cv2.VideoCapture(self.webcam_id)
            if not self.cap.isOpened():
                logger.error(f"Failed to open webcam {self.webcam_id}")
                return False
            
            # Set webcam properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Get actual properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Webcam initialized: {width}x{height} @ {fps:.1f} FPS")
            return True
            
        except Exception as e:
            logger.error(f"Webcam initialization failed: {e}")
            return False

    async def _initialize_video(self) -> bool:
        """비디오 파일 초기화"""
        try:
            if not self.video_files:
                logger.error("No video files provided")
                return False
            
            logger.info(f"Initializing video files: {len(self.video_files)} files")
            
            # Try to open the first video file
            for i, video_file in enumerate(self.video_files):
                if await self._try_open_video(video_file, i):
                    return True
            
            logger.error("Failed to open any video file")
            return False
            
        except Exception as e:
            logger.error(f"Video initialization failed: {e}")
            return False

    async def _try_open_video(self, video_file: str, index: int) -> bool:
        """비디오 파일 열기 시도"""
        try:
            if not os.path.exists(video_file):
                logger.warning(f"Video file not found: {video_file}")
                return False
            
            self.cap = cv2.VideoCapture(video_file)
            if not self.cap.isOpened():
                logger.warning(f"Failed to open video file: {video_file}")
                return False
            
            # Get video properties
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.current_video_index = index
            self.fps = fps
            
            logger.info(f"Video opened: {os.path.basename(video_file)} - {width}x{height} @ {fps:.1f} FPS ({total_frames} frames)")
            return True
            
        except Exception as e:
            logger.error(f"Error opening video file {video_file}: {e}")
            return False

    async def _wait_for_first_frame(self, timeout: float = 5.0) -> bool:
        """첫 번째 프레임 대기"""
        try:
            logger.info("Waiting for first frame...")
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                with self.frame_lock:
                    if self.current_frame is not None:
                        logger.info("✅ First frame received")
                        return True
                
                await asyncio.sleep(0.1)
            
            logger.error("Timeout waiting for first frame")
            return False
            
        except Exception as e:
            logger.error(f"Error waiting for first frame: {e}")
            return False

    def _capture_loop(self):
        """캡처 루프 (별도 스레드에서 실행)"""
        try:
            logger.info("Capture loop started")
            
            while self.is_running:
                if self.cap is None or not self.cap.isOpened():
                    break
                
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    if self.source_type == "video":
                        # Try next video file
                        if not self._try_next_video():
                            break
                    else:
                        # Webcam error
                        logger.warning("Failed to read frame from webcam")
                        break
                
                # Update current frame
                with self.frame_lock:
                    self.current_frame = frame.copy()
                    self.frame_count += 1
                
                # Control playback speed
                if self.source_type == "video":
                    time.sleep(1.0 / (self.fps * self.playback_speed))
                
        except Exception as e:
            logger.error(f"Capture loop error: {e}")
        finally:
            logger.info("Capture loop ended")

    def _try_next_video(self) -> bool:
        """다음 비디오 파일 시도"""
        try:
            self.current_video_index += 1
            
            if self.current_video_index >= len(self.video_files):
                logger.info("All video files completed")
                return False
            
            # Close current video
            if self.cap:
                self.cap.release()
            
            # Open next video
            next_video = self.video_files[self.current_video_index]
            if self._try_open_video_sync(next_video, self.current_video_index):
                self.video_changed = True
                logger.info(f"Switched to next video: {os.path.basename(next_video)}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error switching to next video: {e}")
            return False

    def _try_open_video_sync(self, video_file: str, index: int) -> bool:
        """동기 비디오 파일 열기"""
        try:
            if not os.path.exists(video_file):
                return False
            
            self.cap = cv2.VideoCapture(video_file)
            if not self.cap.isOpened():
                return False
            
            self.current_video_index = index
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            return True
            
        except Exception:
            return False

    async def get_frame(self) -> Optional[np.ndarray]:
        """현재 프레임 반환"""
        try:
            with self.frame_lock:
                if self.current_frame is not None:
                    return self.current_frame.copy()
                return None
                
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None

    def is_active(self) -> bool:
        """입력 소스 활성 상태 확인"""
        return self.is_running and self.is_initialized and self.cap is not None and self.cap.isOpened()

    def get_performance_info(self) -> Dict[str, Any]:
        """성능 정보 반환"""
        elapsed_time = time.time() - self.start_time
        current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0.0
        
        return {
            "source_type": self.source_type,
            "frame_count": self.frame_count,
            "fps": current_fps,
            "elapsed_time": elapsed_time,
            "is_active": self.is_active(),
            "current_video_index": self.current_video_index if self.source_type == "video" else None,
            "total_videos": len(self.video_files) if self.source_type == "video" else None
        }

    def set_playback_speed(self, speed: float):
        """재생 속도 설정"""
        self.playback_speed = max(0.1, min(5.0, speed))
        logger.info(f"Playback speed set to {self.playback_speed}x")

    def has_video_changed(self) -> bool:
        """비디오 변경 여부 확인"""
        if self.video_changed:
            self.video_changed = False
            return True
        return False

    async def stop(self):
        """비디오 입력 중지"""
        try:
            logger.info("Stopping Video Input Manager...")
            
            self.is_running = False
            
            # Wait for capture thread to finish
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=5.0)
            
            # Release video capture
            if self.cap:
                self.cap.release()
                self.cap = None
            
            self.is_initialized = False
            logger.info("Video Input Manager stopped successfully")
            
        except Exception as e:
            logger.error(f"Video Input Manager shutdown error: {e}")

    def get_error(self) -> Optional[str]:
        """마지막 오류 반환"""
        return self.last_error

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        asyncio.run(self.stop())