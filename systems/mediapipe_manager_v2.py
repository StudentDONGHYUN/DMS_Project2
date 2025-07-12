"""
MediaPipe Manager v2
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
        self.last_results = {
            "face": None,
            "pose": None,
            "hand": None,
            "object": None
        }
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_timestamp = -1
        
        # Task health
        self.task_health = {
            "face": False,
            "pose": False,
            "hand": False,
            "object": False
        }
        
        # Initialize components
        self._initialize_components()
        
        logger.info("MediaPipe Manager initialized")

    def _initialize_components(self):
        """MediaPipe 컴포넌트 초기화"""
        logger.info("Initializing MediaPipe components...")
        
        # Face Landmarker
        if self.enable_face:
            self._initialize_face_landmarker()
        
        # Pose Landmarker
        if self.enable_pose:
            self._initialize_pose_landmarker()
        
        # Hand Landmarker
        if self.enable_hand:
            self._initialize_hand_landmarker()
        
        # Object Detector
        if self.enable_object:
            self._initialize_object_detector()
        
        logger.info("MediaPipe components initialization completed")

    def _initialize_face_landmarker(self):
        """얼굴 랜드마커 초기화"""
        try:
            model_path = self.models_dir / "face_landmarker.task"
            if not model_path.exists():
                logger.warning(f"Face landmarker model not found: {model_path}")
                return
            
            face_options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                result_callback=self._on_face_result
            )
            
            self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
            self.task_health["face"] = True
            logger.info("✅ Face Landmarker initialized")
            
        except Exception as e:
            logger.error(f"❌ Face Landmarker initialization failed: {e}")

    def _initialize_pose_landmarker(self):
        """포즈 랜드마커 초기화"""
        try:
            model_path = self.models_dir / "pose_landmarker_full.task"
            if not model_path.exists():
                logger.warning(f"Pose landmarker model not found: {model_path}")
                return
            
            pose_options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_poses=1,
                result_callback=self._on_pose_result
            )
            
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
            self.task_health["pose"] = True
            logger.info("✅ Pose Landmarker initialized")
            
        except Exception as e:
            logger.error(f"❌ Pose Landmarker initialization failed: {e}")

    def _initialize_hand_landmarker(self):
        """손 랜드마커 초기화"""
        try:
            model_path = self.models_dir / "hand_landmarker.task"
            if not model_path.exists():
                logger.warning(f"Hand landmarker model not found: {model_path}")
                return
            
            hand_options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_hands=2,
                result_callback=self._on_hand_result
            )
            
            self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
            self.task_health["hand"] = True
            logger.info("✅ Hand Landmarker initialized")
            
        except Exception as e:
            logger.error(f"❌ Hand Landmarker initialization failed: {e}")

    def _initialize_object_detector(self):
        """객체 감지기 초기화"""
        try:
            model_path = self.models_dir / "efficientdet_lite0.tflite"
            if not model_path.exists():
                logger.warning(f"Object detector model not found: {model_path}")
                return
            
            object_options = vision.ObjectDetectorOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.LIVE_STREAM,
                max_results=5,
                score_threshold=0.3,
                result_callback=self._on_object_result
            )
            
            self.object_detector = vision.ObjectDetector.create_from_options(object_options)
            self.task_health["object"] = True
            logger.info("✅ Object Detector initialized")
            
        except Exception as e:
            logger.error(f"❌ Object Detector initialization failed: {e}")

    def _on_face_result(self, result, output_image, timestamp_ms):
        """얼굴 결과 콜백"""
        try:
            self.last_results["face"] = {
                "landmarks": result.face_landmarks,
                "blendshapes": result.face_blendshapes,
                "transformation_matrixes": result.facial_transformation_matrixes,
                "timestamp": timestamp_ms
            }
        except Exception as e:
            logger.error(f"Face result processing error: {e}")

    def _on_pose_result(self, result, output_image, timestamp_ms):
        """포즈 결과 콜백"""
        try:
            self.last_results["pose"] = {
                "landmarks": result.pose_landmarks,
                "world_landmarks": result.pose_world_landmarks,
                "timestamp": timestamp_ms
            }
        except Exception as e:
            logger.error(f"Pose result processing error: {e}")

    def _on_hand_result(self, result, output_image, timestamp_ms):
        """손 결과 콜백"""
        try:
            self.last_results["hand"] = {
                "landmarks": result.hand_landmarks,
                "world_landmarks": result.hand_world_landmarks,
                "handedness": result.handedness,
                "timestamp": timestamp_ms
            }
        except Exception as e:
            logger.error(f"Hand result processing error: {e}")

    def _on_object_result(self, result, output_image, timestamp_ms):
        """객체 결과 콜백"""
        try:
            self.last_results["object"] = {
                "detections": result.detections,
                "timestamp": timestamp_ms
            }
        except Exception as e:
            logger.error(f"Object result processing error: {e}")

    async def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        프레임 처리
        
        Args:
            frame: 입력 프레임
            
        Returns:
            처리된 결과 딕셔너리
        """
        try:
            # Update frame count
            self.frame_count += 1
            
            # Generate timestamp
            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= self.last_timestamp:
                timestamp_ms = self.last_timestamp + 1
            self.last_timestamp = timestamp_ms
            
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Process with active components
            if self.enable_face and self.face_landmarker:
                self.face_landmarker.detect_async(mp_image, timestamp_ms)
            
            if self.enable_pose and self.pose_landmarker:
                self.pose_landmarker.detect_async(mp_image, timestamp_ms)
            
            if self.enable_hand and self.hand_landmarker:
                self.hand_landmarker.detect_async(mp_image, timestamp_ms)
            
            if self.enable_object and self.object_detector:
                self.object_detector.detect_async(mp_image, timestamp_ms)
            
            # Return current results
            return self.get_results()
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return self.get_fallback_results()

    def get_results(self) -> Dict[str, Any]:
        """현재 결과 반환"""
        return {
            "face": self.last_results["face"],
            "pose": self.last_results["pose"],
            "hand": self.last_results["hand"],
            "object": self.last_results["object"],
            "frame_count": self.frame_count,
            "task_health": self.task_health.copy()
        }

    def get_fallback_results(self) -> Dict[str, Any]:
        """폴백 결과 반환"""
        return {
            "face": None,
            "pose": None,
            "hand": None,
            "object": None,
            "frame_count": self.frame_count,
            "task_health": self.task_health.copy()
        }

    def get_performance_info(self) -> Dict[str, Any]:
        """성능 정보 반환"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0.0
        
        return {
            "frame_count": self.frame_count,
            "fps": fps,
            "elapsed_time": elapsed_time,
            "task_health": self.task_health.copy()
        }

    def is_healthy(self) -> bool:
        """시스템 건강 상태 확인"""
        return any(self.task_health.values())

    def get_active_tasks(self) -> list:
        """활성 태스크 목록 반환"""
        return [task for task, active in self.task_health.items() if active]

    def close(self):
        """리소스 정리"""
        logger.info("Closing MediaPipe Manager...")
        
        try:
            # Close all components
            if self.face_landmarker:
                self.face_landmarker.close()
            
            if self.pose_landmarker:
                self.pose_landmarker.close()
            
            if self.hand_landmarker:
                self.hand_landmarker.close()
            
            if self.object_detector:
                self.object_detector.close()
            
            logger.info("MediaPipe Manager closed successfully")
            
        except Exception as e:
            logger.error(f"MediaPipe Manager close error: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()