import cv2
import mediapipe as mp
import time
import logging
import threading
import queue
import asyncio
import numpy as np
from typing import List
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from systems.performance import PerformanceOptimizer

logger = logging.getLogger(__name__)

class EnhancedMediaPipeManager:
    """향상된 MediaPipe 관리자 (큐 및 스레드 기반 콜백 처리)"""

    def __init__(self, analysis_engine):
        self.analysis_engine = analysis_engine
        self.fps_avg_frame_count = 10
        self.counter, self.start_time, self.current_fps = 0, time.time(), 0.0
        self.face_landmarker, self.pose_landmarker, self.hand_landmarker, self.object_detector = None, None, None, None
        self.task_health = {"face": False, "pose": False, "hand": False, "object": False}
        self.active_tasks = ["face", "pose", "hand", "object"]
        self.performance_optimizer = PerformanceOptimizer()
        self.last_timestamp = -1
        
        # 결과 저장을 위한 속성
        self.last_face_result = None
        self.last_pose_result = None
        self.last_hand_result = None
        self.last_object_result = None

        self.result_queue = queue.Queue()
        self._shutdown_requested = False  # Initialize shutdown flag
        self._initialize_tasks()
        
        self.callback_thread = threading.Thread(target=self._process_callbacks, daemon=True)
        self.callback_thread.start()

    def _calculate_fps(self):
        if self.counter > 0 and self.counter % self.fps_avg_frame_count == 0:
            self.current_fps = self.fps_avg_frame_count / (time.time() - self.start_time)
            self.start_time = time.time()
        self.counter += 1

    def _process_callbacks(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Add shutdown flag for safer exit
        self._shutdown_requested = False
        
        while not self._shutdown_requested:
            try:
                # Add timeout to prevent infinite blocking
                result_type, result, timestamp = self.result_queue.get(timeout=1.0)
                if result_type == 'shutdown':
                    self._shutdown_requested = True
                    break
                
                if self.analysis_engine:
                    if result_type == 'face' and hasattr(self.analysis_engine, 'on_face_result'):
                        loop.run_until_complete(self.analysis_engine.on_face_result(result, timestamp=timestamp))
                    elif result_type == 'pose' and hasattr(self.analysis_engine, 'on_pose_result'):
                        loop.run_until_complete(self.analysis_engine.on_pose_result(result, timestamp=timestamp))
                    elif result_type == 'hand' and hasattr(self.analysis_engine, 'on_hand_result'):
                        loop.run_until_complete(self.analysis_engine.on_hand_result(result, timestamp=timestamp))
                    elif result_type == 'object' and hasattr(self.analysis_engine, 'on_object_result'):
                        loop.run_until_complete(self.analysis_engine.on_object_result(result, timestamp=timestamp))
            except queue.Empty:
                # Timeout occurred, check if shutdown was requested
                continue
            except Exception as e:
                logger.error(f"Callback 처리 중 오류: {e}")
                # On critical errors, consider shutting down gracefully
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    self._shutdown_requested = True
                    break
        
        logger.info("Callback processing loop exiting gracefully")
        loop.close()

    def _on_face_result(self, result, output_image, timestamp_ms):
        self.last_face_result = result
        self.result_queue.put(('face', result, timestamp_ms))

    def _on_pose_result(self, result, output_image, timestamp_ms):
        self.last_pose_result = result
        self.result_queue.put(('pose', result, timestamp_ms))

    def _on_hand_result(self, result, output_image, timestamp_ms):
        self.last_hand_result = result
        self.result_queue.put(('hand', result, timestamp_ms))

    def _on_object_result(self, result, output_image, timestamp_ms):
        self.last_object_result = result
        self.result_queue.put(('object', result, timestamp_ms))

    def _initialize_tasks(self):
        # FaceLandmarker
        try:
            face_options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path='models/face_landmarker.task'),
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                result_callback=self._on_face_result)
            self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
            self.task_health["face"] = True
            logger.info("✅ FaceLandmarker 초기화 완료")
        except Exception as e:
            logger.error(f"❌ FaceLandmarker 초기화 실패: {e}")

        # PoseLandmarker
        try:
            pose_options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path='models/pose_landmarker_full.task'),
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_poses=1,
                result_callback=self._on_pose_result)
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
            self.task_health["pose"] = True
            logger.info("✅ PoseLandmarker 초기화 완료")
        except Exception as e:
            logger.error(f"❌ PoseLandmarker 초기화 실패: {e}")

        # HandLandmarker
        try:
            hand_options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path='models/hand_landmarker.task'),
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_hands=2,
                result_callback=self._on_hand_result)
            self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
            self.task_health["hand"] = True
            logger.info("✅ HandLandmarker 초기화 완료")
        except Exception as e:
            logger.error(f"❌ HandLandmarker 초기화 실패: {e}")

        # ObjectDetector
        try:
            object_options = vision.ObjectDetectorOptions(
                base_options=python.BaseOptions(model_asset_path='models/efficientdet_lite0.tflite'),
                running_mode=vision.RunningMode.LIVE_STREAM,
                max_results=5,
                score_threshold=0.3,
                result_callback=self._on_object_result)
            self.object_detector = vision.ObjectDetector.create_from_options(object_options)
            self.task_health["object"] = True
            logger.info("✅ ObjectDetector 초기화 완료")
        except Exception as e:
            logger.error(f"❌ ObjectDetector 초기화 실패: {e}")

    def run_tasks(self, frame):
        self._calculate_fps()
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self.last_timestamp:
            timestamp_ms = self.last_timestamp + 1
        self.last_timestamp = timestamp_ms

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            if "face" in self.active_tasks and self.face_landmarker:
                self.face_landmarker.detect_async(mp_image, timestamp_ms)
            if "pose" in self.active_tasks and self.pose_landmarker:
                self.pose_landmarker.detect_async(mp_image, timestamp_ms)
            if "hand" in self.active_tasks and self.hand_landmarker:
                self.hand_landmarker.detect_async(mp_image, timestamp_ms)
            if "object" in self.active_tasks and self.object_detector:
                self.object_detector.detect_async(mp_image, timestamp_ms)
        except Exception as e:
            logger.error(f"MediaPipe 태스크 실행 중 오류: {e}")

    def get_latest_results(self):
        return {
            "face": self.last_face_result,
            "pose": self.last_pose_result,
            "hand": self.last_hand_result,
            "object": self.last_object_result,
        }

    def close(self):
        logger.info("MediaPipe Task 리소스 정리 시작...")
        
        # Set shutdown flag first
        self._shutdown_requested = True
        
        # Send shutdown signal to queue
        try:
            self.result_queue.put(('shutdown', None, None))
        except Exception as e:
            logger.warning(f"Shutdown signal 전송 실패: {e}")
        
        # Wait for callback thread to finish with timeout
        if self.callback_thread and self.callback_thread.is_alive():
            self.callback_thread.join(timeout=2.0)
            if self.callback_thread.is_alive():
                logger.warning("Callback thread가 정상적으로 종료되지 않았습니다.")
        
        # Close MediaPipe tasks
        for task in [self.face_landmarker, self.pose_landmarker, self.hand_landmarker, self.object_detector]:
            if task:
                try:
                    task.close()
                except Exception as e:
                    logger.warning(f"MediaPipe task 종료 중 오류: {e}")
        
        logger.info("MediaPipe Task 리소스 정리 완료.")