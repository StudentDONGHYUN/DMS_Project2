"""
MediaPipe Manager - Enhanced System
통합 시스템과 호환되는 MediaPipe 관리자
기존 코드와 개선된 코드를 통합하여 호환성과 성능을 모두 확보
"""

import cv2
import mediapipe as mp
import time
import logging
import asyncio
import numpy as np
from typing import Dict, Any, Optional
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

logger = logging.getLogger(__name__)


class MediaPipeManager:
    """통합 MediaPipe 관리자 - 기존 코드와 개선된 코드의 통합"""

    def __init__(
        self,
        enable_face: bool = True,
        enable_pose: bool = True,
        enable_hand: bool = True,
        enable_object: bool = True,
        models_dir: str = "models"
    ):
        """
        MediaPipe 관리자 초기화
        
        Args:
            enable_face: 얼굴 랜드마크 활성화
            enable_pose: 포즈 랜드마크 활성화
            enable_hand: 손 랜드마크 활성화
            enable_object: 객체 감지 활성화
            models_dir: 모델 디렉토리 경로
        """
        self.enable_face = enable_face
        self.enable_pose = enable_pose
        self.enable_hand = enable_hand
        self.enable_object = enable_object
        self.models_dir = Path(models_dir)
        
        # MediaPipe components
        self.face_landmarker = None
        self.pose_landmarker = None
        self.hand_landmarker = None
        self.object_detector = None
        
        # Results storage
        self.latest_results = {
            'face': None,
            'pose': None,
            'hand': None,
            'object': None,
            'timestamp': time.time()
        }
        
        # Error tracking
        self.error_count = 0
        self.max_errors = 10
        
        # Performance tracking
        self.processing_times = []
        self.frame_count = 0
        
        # Initialize components
        self._initialize_components()
        
        logger.info("MediaPipe Manager initialized")

    def _initialize_components(self):
        """MediaPipe 컴포넌트 초기화"""
        try:
            # Create models directory if it doesn't exist
            self.models_dir.mkdir(exist_ok=True)
            
            if self.enable_face:
                self._initialize_face_landmarker()
            
            if self.enable_pose:
                self._initialize_pose_landmarker()
            
            if self.enable_hand:
                self._initialize_hand_landmarker()
            
            if self.enable_object:
                self._initialize_object_detector()
                
        except Exception as e:
            logger.error(f"Error initializing MediaPipe components: {e}")
            self.error_count += 1

    def _initialize_face_landmarker(self):
        """얼굴 랜드마크 초기화"""
        try:
            base_options = python.BaseOptions(
                model_asset_path=str(self.models_dir / 'face_landmarker.task')
            )
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.LIVE_STREAM,
                result_callback=self._on_face_result,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
            logger.info("Face landmarker initialized")
        except Exception as e:
            logger.warning(f"Face landmarker initialization failed: {e}")
            self.face_landmarker = None

    def _initialize_pose_landmarker(self):
        """포즈 랜드마크 초기화"""
        try:
            base_options = python.BaseOptions(
                model_asset_path=str(self.models_dir / 'pose_landmarker.task')
            )
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.LIVE_STREAM,
                result_callback=self._on_pose_result,
                num_poses=1
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
            logger.info("Pose landmarker initialized")
        except Exception as e:
            logger.warning(f"Pose landmarker initialization failed: {e}")
            self.pose_landmarker = None

    def _initialize_hand_landmarker(self):
        """손 랜드마크 초기화"""
        try:
            base_options = python.BaseOptions(
                model_asset_path=str(self.models_dir / 'hand_landmarker.task')
            )
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.LIVE_STREAM,
                result_callback=self._on_hand_result,
                num_hands=2
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
            logger.info("Hand landmarker initialized")
        except Exception as e:
            logger.warning(f"Hand landmarker initialization failed: {e}")
            self.hand_landmarker = None

    def _initialize_object_detector(self):
        """객체 감지기 초기화"""
        try:
            base_options = python.BaseOptions(
                model_asset_path=str(self.models_dir / 'object_detector.tflite')
            )
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.LIVE_STREAM,
                result_callback=self._on_object_result,
                score_threshold=0.5,
                max_results=10
            )
            self.object_detector = vision.ObjectDetector.create_from_options(options)
            logger.info("Object detector initialized")
        except Exception as e:
            logger.warning(f"Object detector initialization failed: {e}")
            self.object_detector = None

    def _on_face_result(self, result, output_image, timestamp_ms):
        """얼굴 랜드마크 결과 콜백"""
        try:
            self.latest_results['face'] = result
            self.latest_results['timestamp'] = time.time()
        except Exception as e:
            logger.error(f"Face result callback error: {e}")
            self.error_count += 1

    def _on_pose_result(self, result, output_image, timestamp_ms):
        """포즈 랜드마크 결과 콜백"""
        try:
            self.latest_results['pose'] = result
            self.latest_results['timestamp'] = time.time()
        except Exception as e:
            logger.error(f"Pose result callback error: {e}")
            self.error_count += 1

    def _on_hand_result(self, result, output_image, timestamp_ms):
        """손 랜드마크 결과 콜백"""
        try:
            self.latest_results['hand'] = result
            self.latest_results['timestamp'] = time.time()
        except Exception as e:
            logger.error(f"Hand result callback error: {e}")
            self.error_count += 1

    def _on_object_result(self, result, output_image, timestamp_ms):
        """객체 감지 결과 콜백"""
        try:
            self.latest_results['object'] = result
            self.latest_results['timestamp'] = time.time()
        except Exception as e:
            logger.error(f"Object result callback error: {e}")
            self.error_count += 1

    async def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """프레임 처리"""
        try:
            start_time = time.time()
            
            # Convert frame to MediaPipe format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(time.time() * 1000)
            
            # Process with each active component
            if self.face_landmarker and self.enable_face:
                try:
                    self.face_landmarker.detect_async(mp_image, timestamp_ms)
                except Exception as e:
                    logger.error(f"Face detection error: {e}")
                    self.error_count += 1
            
            if self.pose_landmarker and self.enable_pose:
                try:
                    self.pose_landmarker.detect_async(mp_image, timestamp_ms)
                except Exception as e:
                    logger.error(f"Pose detection error: {e}")
                    self.error_count += 1
            
            if self.hand_landmarker and self.enable_hand:
                try:
                    self.hand_landmarker.detect_async(mp_image, timestamp_ms)
                except Exception as e:
                    logger.error(f"Hand detection error: {e}")
                    self.error_count += 1
            
            if self.object_detector and self.enable_object:
                try:
                    self.object_detector.detect_async(mp_image, timestamp_ms)
                except Exception as e:
                    logger.error(f"Object detection error: {e}")
                    self.error_count += 1
            
            # Track performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            self.frame_count += 1
            
            return self.get_results()
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self.error_count += 1
            return self.get_fallback_results()

    def get_results(self) -> Dict[str, Any]:
        """최신 결과 반환"""
        return {
            'face_result': self.latest_results['face'],
            'pose_result': self.latest_results['pose'],
            'hand_result': self.latest_results['hand'],
            'object_result': self.latest_results['object'],
            'timestamp': self.latest_results['timestamp'],
            'frame_count': self.frame_count
        }

    def get_fallback_results(self) -> Dict[str, Any]:
        """폴백 결과 반환"""
        return {
            'face_result': None,
            'pose_result': None,
            'hand_result': None,
            'object_result': None,
            'timestamp': time.time(),
            'frame_count': self.frame_count,
            'error': True
        }

    def get_performance_info(self) -> Dict[str, Any]:
        """성능 정보 반환"""
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            'frame_count': self.frame_count,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.frame_count),
            'is_healthy': self.is_healthy()
        }

    def is_healthy(self) -> bool:
        """시스템 상태 확인"""
        return self.error_count < self.max_errors

    def get_active_tasks(self) -> list:
        """활성 작업 목록 반환"""
        active = []
        if self.enable_face and self.face_landmarker:
            active.append('face')
        if self.enable_pose and self.pose_landmarker:
            active.append('pose')
        if self.enable_hand and self.hand_landmarker:
            active.append('hand')
        if self.enable_object and self.object_detector:
            active.append('object')
        return active

    def close(self):
        """리소스 정리"""
        try:
            if self.face_landmarker:
                self.face_landmarker.close()
                self.face_landmarker = None
            
            if self.pose_landmarker:
                self.pose_landmarker.close()
                self.pose_landmarker = None
            
            if self.hand_landmarker:
                self.hand_landmarker.close()
                self.hand_landmarker = None
            
            if self.object_detector:
                self.object_detector.close()
                self.object_detector = None
            
            logger.info("MediaPipe Manager closed")
            
        except Exception as e:
            logger.error(f"Error closing MediaPipe Manager: {e}")

    def __enter__(self):
        """Context manager 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.close()

    # Legacy compatibility methods
    def process_frame_sync(self, frame):
        """Legacy synchronous processing"""
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.process_frame(frame))
        except Exception as e:
            logger.error(f"Sync processing error: {e}")
            return self.get_fallback_results()


# Legacy compatibility class
class EnhancedMediaPipeManager(MediaPipeManager):
    """Legacy compatibility class"""
    def __init__(self, *args, **kwargs):
        logger.warning("EnhancedMediaPipeManager is deprecated, use MediaPipeManager instead")
        super().__init__(*args, **kwargs)