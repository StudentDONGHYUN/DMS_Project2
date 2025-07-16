import cv2
import mediapipe as mp
import time
import logging
from typing import List
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from systems.performance import PerformanceOptimizer

logger = logging.getLogger(__name__)

class EnhancedMediaPipeManager:
    """향상된 MediaPipe 관리자"""

    def __init__(self, analysis_engine):
        self.analysis_engine = analysis_engine
        self.fps_avg_frame_count = 10
        self.counter, self.start_time, self.current_fps = 0, time.time(), 0.0
        self.face_landmarker, self.pose_landmarker, self.hand_landmarker, self.object_detector = None, None, None, None
        self.task_health = {"face": False, "pose": False, "hand": False, "object": False}
        self.active_tasks = ["face", "pose"]
        self.performance_optimizer = PerformanceOptimizer()
        self._initialize_tasks()

    def _calculate_fps(self):
        if self.counter > 0 and self.counter % self.fps_avg_frame_count == 0:
            self.current_fps = self.fps_avg_frame_count / (time.time() - self.start_time)
            self.start_time = time.time()
        self.counter += 1

    def _initialize_tasks(self):
        try:
            options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path="models/face_landmarker.task"),
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                result_callback=self.analysis_engine.on_face_result,
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
            self.task_health["face"] = True
            logger.info("✅ FaceLandmarker 초기화 완료 (블렌드셰이프 활성화)")
        except Exception as e:
            logger.error(f"❌ FaceLandmarker 초기화 실패: {e}")
        try:
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path="models/pose_landmarker_full.task"),
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_poses=1,
                min_pose_detection_confidence=0.7,
                min_pose_presence_confidence=0.9,
                min_tracking_confidence=0.8,
                result_callback=self.analysis_engine.on_pose_result,
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
            self.task_health["pose"] = True
            logger.info("✅ PoseLandmarker 초기화 완료 (33 키포인트)")
        except Exception as e:
            logger.error(f"❌ PoseLandmarker 초기화 실패: {e}")
        try:
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path="models/hand_landmarker.task"),
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_hands=2,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.7,
                result_callback=self.analysis_engine.on_hand_result,
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
            self.task_health["hand"] = True
            logger.info("✅ HandLandmarker 초기화 완료")
        except Exception as e:
            logger.error(f"❌ HandLandmarker 초기화 실패: {e}")
        try:
            options = vision.ObjectDetectorOptions(
                base_options=python.BaseOptions(model_asset_path="models/efficientdet_lite0.tflite"),
                running_mode=vision.RunningMode.LIVE_STREAM,
                max_results=5,
                score_threshold=0.3,
                category_allowlist=["cell phone", "cup", "bottle", "sandwich", "book", "laptop"],
                result_callback=self.analysis_engine.on_object_result,
            )
            self.object_detector = vision.ObjectDetector.create_from_options(options)
            self.task_health["object"] = True
            logger.info("✅ ObjectDetector 초기화 완료 (주의산만 객체 감지)")
        except Exception as e:
            logger.error(f"❌ ObjectDetector 초기화 실패: {e}")

    def update_active_tasks(self, required_tasks: List[str]):
        self.active_tasks = required_tasks

    def run_tasks(self, frame):
        start_time = time.time()
        self._calculate_fps()
        timestamp_ms = int(time.time() * 1000)
        self.analysis_engine.frame_buffer[timestamp_ms] = frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if "face" in self.active_tasks and self.task_health["face"]:
            self.face_landmarker.detect_async(mp_image, timestamp_ms)
        if "pose" in self.active_tasks and self.task_health["pose"]:
            self.pose_landmarker.detect_async(mp_image, timestamp_ms)
        if "hand" in self.active_tasks and self.task_health["hand"]:
            self.hand_landmarker.detect_async(mp_image, timestamp_ms)
        if "object" in self.active_tasks and self.task_health["object"]:
            self.object_detector.detect_async(mp_image, timestamp_ms)
        processing_time = (time.time() - start_time) * 1000
        self.performance_optimizer.log_performance(processing_time, self.current_fps)

    def get_system_health(self) -> dict:
        healthy = sum(self.task_health.values())
        total = len(self.task_health)
        return {
            "healthy_tasks": healthy,
            "total_tasks": total,
            "task_status": self.task_health.copy(),
            "overall_health": healthy / total if total > 0 else 0,
            "performance_status": self.performance_optimizer.get_optimization_status(),
        }

    def close(self):
        logger.info("MediaPipe Task 리소스 정리 시작...")
        for task in [self.face_landmarker, self.pose_landmarker, self.hand_landmarker, self.object_detector]:
            if task and hasattr(task, "close"):
                try:
                    task.close()
                except Exception as e:
                    logger.error(f"Task 정리 중 오류 발생: {e}")
        logger.info("MediaPipe Task 리소스 정리 완료.")
