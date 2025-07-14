"""
S-Class DMS v19+ 차세대 MediaPipe Tasks Manager
최신 MediaPipe Tasks API (0.10.9+) 적용

NOTE: Python API에서는 GPU delegate를 명시적으로 지정할 수 없습니다.
시스템에 CUDA/TF Lite delegate가 설치되어 있으면 자동 활용됩니다.
최대 성능을 원할 경우, CUDA/TF Lite delegate가 설치된 환경에서 실행하세요.
"""

import cv2
import mediapipe as mp
import time
import asyncio
import logging
import threading
import queue
import numpy as np
from typing import List, Dict, Optional, Callable, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from collections import deque
import time
import cv2
import mediapipe as mp
import platform
# MediaPipe Tasks API imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision, audio, text

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """MediaPipe Tasks 타입"""
    FACE_LANDMARKER = "face_landmarker"
    POSE_LANDMARKER = "pose_landmarker" 
    HAND_LANDMARKER = "hand_landmarker"
    GESTURE_RECOGNIZER = "gesture_recognizer"
    OBJECT_DETECTOR = "object_detector"
    IMAGE_CLASSIFIER = "image_classifier"
    FACE_DETECTOR = "face_detector"
    HOLISTIC_LANDMARKER = "holistic_landmarker"

@dataclass
class TaskConfig:
    """MediaPipe Task 설정"""
    task_type: TaskType
    model_path: str
    max_results: int = 1
    score_threshold: float = 0.5
    running_mode: vision.RunningMode = vision.RunningMode.LIVE_STREAM
    enable_face_blendshapes: bool = False
    enable_facial_transformation_matrix: bool = False
    enable_segmentation_masks: bool = False
    num_poses: int = 1
    num_hands: int = 2
    num_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

class AdvancedMediaPipeManager:
    """
    차세대 MediaPipe Tasks Manager - 성능 최적화
    - 최신 Tasks API 완전 활용
    - 동적 모델 로딩/언로딩
    - 성능 최적화 및 메모리 관리
    - 포괄적 오류 처리
    """
    
    def __init__(self, analysis_engine=None, config_file: Optional[str] = None):
        self.analysis_engine = analysis_engine
        self.active_tasks: Dict[TaskType, Any] = {}
        self.task_configs: Dict[TaskType, TaskConfig] = {}
        self.result_callbacks: Dict[TaskType, Callable] = {}
        self.task_health: Dict[TaskType, bool] = {}
        
        # 성능 모니터링 - 최적화된 deque 사용
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.frame_processing_times = deque(maxlen=100)  # O(1) operations
        
        # 결과 관리
        self.latest_results: Dict[TaskType, Any] = {}
        self.result_queue = queue.Queue(maxsize=100)
        self.processing_lock = asyncio.Lock()
        self.last_timestamp = 0
        
        # 모델 경로 설정
        self.model_base_path = Path("models")
        self.ensure_model_directory()
        
        # 비동기 처리 스레드
        self.callback_thread = None
        self.running = False
        
        # 기본 설정 로드
        self._load_default_configs()
        
        # 외부 설정 파일 로드 (있는 경우)
        if config_file:
            self._load_config_file(config_file)
        
        logger.info("AdvancedMediaPipeManager v2.0 (최적화) 초기화 완료")

    def ensure_model_directory(self):
        """모델 디렉토리 존재 확인"""
        self.model_base_path.mkdir(exist_ok=True)
        logger.info(f"모델 디렉토리: {self.model_base_path.absolute()}")

    def _load_default_configs(self):
        """기본 Task 설정 로드"""
        
        # Face Landmarker 설정
        self.task_configs[TaskType.FACE_LANDMARKER] = TaskConfig(
            task_type=TaskType.FACE_LANDMARKER,
            model_path=str(self.model_base_path / "face_landmarker_v2_with_blendshapes.task"),
            num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_face_blendshapes=True,
            enable_facial_transformation_matrix=True
        )
        
        # Pose Landmarker 설정
        self.task_configs[TaskType.POSE_LANDMARKER] = TaskConfig(
            task_type=TaskType.POSE_LANDMARKER,
            model_path=str(self.model_base_path / "pose_landmarker_heavy.task"),
            num_poses=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation_masks=True
        )
        
        # Hand Landmarker 설정
        self.task_configs[TaskType.HAND_LANDMARKER] = TaskConfig(
            task_type=TaskType.HAND_LANDMARKER,
            model_path=str(self.model_base_path / "hand_landmarker.task"),
            num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Gesture Recognizer 설정 (새로운 기능)
        self.task_configs[TaskType.GESTURE_RECOGNIZER] = TaskConfig(
            task_type=TaskType.GESTURE_RECOGNIZER,
            model_path=str(self.model_base_path / "gesture_recognizer.task"),
            num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Object Detector 설정
        self.task_configs[TaskType.OBJECT_DETECTOR] = TaskConfig(
            task_type=TaskType.OBJECT_DETECTOR,
            model_path=str(self.model_base_path / "efficientdet_lite0.tflite"),
            max_results=5,
            score_threshold=0.3
        )
        
        # Holistic Landmarker 설정 (최신 통합 모델)
        self.task_configs[TaskType.HOLISTIC_LANDMARKER] = TaskConfig(
            task_type=TaskType.HOLISTIC_LANDMARKER,
            model_path=str(self.model_base_path / "holistic_landmarker.task"),
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _load_config_file(self, config_file: str):
        """외부 설정 파일 로드"""
        try:
            # JSON 또는 YAML 설정 파일 파싱 로직
            logger.info(f"설정 파일 로드: {config_file}")
            # 구현 필요 시 추가
        except Exception as e:
            logger.warning(f"설정 파일 로드 실패: {e}")

    def _detect_dsp(self):
        """RB2 등 DSP(Hexagon) 감지: 실제 환경에 맞게 확장 필요"""
        # 예시: /proc/cpuinfo, lscpu, 환경변수 등으로 DSP 감지
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                if 'hexagon' in cpuinfo or 'dsp' in cpuinfo:
                    return 'HEXAGON'
        except Exception:
            pass
        # 환경변수 등 추가 감지 로직 필요시 확장
        return None

    async def initialize_task(self, task_type: TaskType) -> bool:
        """개별 Task 초기화 (DSP 감지 및 delegate 적용)"""
        if task_type in self.active_tasks:
            logger.warning(f"{task_type.value} 이미 초기화됨")
            return True
        config = self.task_configs.get(task_type)
        if not config:
            logger.error(f"{task_type.value} 설정 없음")
            return False
        try:
            # 모델 파일 존재 확인
            if not Path(config.model_path).exists():
                logger.warning(f"모델 파일 없음: {config.model_path}")
                return False

            # delegate 자동 분기 (공식 지원: CPU/GPU만)
            system = platform.system()
            # GPU delegate는 Linux, macOS에서만 공식 지원
            if system in ["Linux", "Darwin"]:
                base_options_kwargs = dict(model_asset_path=config.model_path)
                base_options_kwargs['delegate'] = BaseOptions.Delegate.GPU
                logger.info(f"GPU delegate 적용: {system}")
            else:
                base_options_kwargs = dict(model_asset_path=config.model_path)
                base_options_kwargs['delegate'] = BaseOptions.Delegate.CPU
                logger.info(f"CPU delegate 적용: {system}")
            base_options = python.BaseOptions(**base_options_kwargs)

            # Task별 초기화 (이하 기존 코드)
            if task_type == TaskType.FACE_LANDMARKER:
                task = await self._initialize_face_landmarker(base_options, config)
            elif task_type == TaskType.POSE_LANDMARKER:
                task = await self._initialize_pose_landmarker(base_options, config)
            elif task_type == TaskType.HAND_LANDMARKER:
                task = await self._initialize_hand_landmarker(base_options, config)
            elif task_type == TaskType.GESTURE_RECOGNIZER:
                task = await self._initialize_gesture_recognizer(base_options, config)
            elif task_type == TaskType.OBJECT_DETECTOR:
                task = await self._initialize_object_detector(base_options, config)
            elif task_type == TaskType.HOLISTIC_LANDMARKER:
                task = await self._initialize_holistic_landmarker(base_options, config)
            else:
                logger.error(f"지원되지 않는 Task 타입: {task_type}")
                return False
            if task:
                self.active_tasks[task_type] = task
                self.task_health[task_type] = True
                logger.info(f"✅ {task_type.value} 초기화 완료")
                return True
        except Exception as e:
            logger.error(f"❌ {task_type.value} 초기화 실패: {e}")
            self.task_health[task_type] = False
        return False

    async def _initialize_face_landmarker(self, base_options, config):
        """Face Landmarker 초기화"""
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=config.running_mode,
            num_faces=config.num_faces,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            output_face_blendshapes=config.enable_face_blendshapes,
            output_facial_transformation_matrixes=config.enable_facial_transformation_matrix,
            result_callback=self._create_result_callback(TaskType.FACE_LANDMARKER)
        )
        return vision.FaceLandmarker.create_from_options(options)

    async def _initialize_pose_landmarker(self, base_options, config):
        """Pose Landmarker 초기화"""
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=config.running_mode,
            num_poses=config.num_poses,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            output_segmentation_masks=config.enable_segmentation_masks,
            result_callback=self._create_result_callback(TaskType.POSE_LANDMARKER)
        )
        return vision.PoseLandmarker.create_from_options(options)

    async def _initialize_hand_landmarker(self, base_options, config):
        """Hand Landmarker 초기화"""
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=config.running_mode,
            num_hands=config.num_hands,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            result_callback=self._create_result_callback(TaskType.HAND_LANDMARKER)
        )
        return vision.HandLandmarker.create_from_options(options)

    async def _initialize_gesture_recognizer(self, base_options, config):
        """Gesture Recognizer 초기화"""
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=config.running_mode,
            num_hands=config.num_hands,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            result_callback=self._create_result_callback(TaskType.GESTURE_RECOGNIZER)
        )
        return vision.GestureRecognizer.create_from_options(options)

    async def _initialize_object_detector(self, base_options, config):
        """Object Detector 초기화"""
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=config.running_mode,
            max_results=config.max_results,
            score_threshold=config.score_threshold,
            result_callback=self._create_result_callback(TaskType.OBJECT_DETECTOR)
        )
        return vision.ObjectDetector.create_from_options(options)

    async def _initialize_holistic_landmarker(self, base_options, config):
        """Holistic Landmarker 초기화 (최신 통합 모델)"""
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=config.running_mode,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            result_callback=self._create_result_callback(TaskType.HOLISTIC_LANDMARKER)
        )
        return vision.HolisticLandmarker.create_from_options(options)

    def _create_result_callback(self, task_type: TaskType):
        """결과 콜백 함수 생성"""
        def callback(result, output_image, timestamp_ms):
            try:
                self.latest_results[task_type] = result
                self.result_queue.put((task_type, result, timestamp_ms), timeout=0.1)
            except queue.Full:
                logger.warning(f"{task_type.value} 결과 큐 오버플로우")
            except Exception as e:
                logger.error(f"{task_type.value} 콜백 오류: {e}")
        return callback

    async def initialize_all_tasks(self) -> Dict[TaskType, bool]:
        """모든 설정된 Task 초기화"""
        results = {}
        
        # 기본 Tasks 초기화
        core_tasks = [
            TaskType.FACE_LANDMARKER,
            TaskType.POSE_LANDMARKER,
            TaskType.HAND_LANDMARKER
        ]
        
        # 선택적 Tasks
        optional_tasks = [
            TaskType.GESTURE_RECOGNIZER,
            TaskType.OBJECT_DETECTOR,
            TaskType.HOLISTIC_LANDMARKER
        ]
        
        # 병렬 초기화
        for task_type in core_tasks:
            results[task_type] = await self.initialize_task(task_type)
        
        # 선택적 Tasks (실패해도 계속)
        for task_type in optional_tasks:
            try:
                results[task_type] = await self.initialize_task(task_type)
            except Exception as e:
                logger.warning(f"선택적 Task {task_type.value} 초기화 실패: {e}")
                results[task_type] = False
        
        # 콜백 처리 스레드 시작
        if not self.callback_thread or not self.callback_thread.is_alive():
            self.running = True
            self.callback_thread = threading.Thread(target=self._process_callbacks, daemon=True)
            self.callback_thread.start()
        
        initialized_count = sum(results.values())
        total_count = len(results)
        logger.info(f"Task 초기화 완료: {initialized_count}/{total_count}")
        
        return results

    def _process_callbacks(self):
        """비동기 콜백 처리"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        logger.info("콜백 처리 스레드 시작")
        
        while self.running:
            try:
                task_type, result, timestamp = self.result_queue.get(timeout=1.0)
                
                if task_type == "shutdown":
                    break
                
                # Analysis engine으로 결과 전달
                if self.analysis_engine:
                    loop.run_until_complete(self._forward_result_to_analysis_engine(task_type, result, timestamp))
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"콜백 처리 오류: {e}")
        
        loop.close()
        logger.info("콜백 처리 스레드 종료")

    async def _forward_result_to_analysis_engine(self, task_type: TaskType, result, timestamp):
        """Analysis Engine으로 결과 전달"""
        try:
            if hasattr(self.analysis_engine, 'on_face_result') and task_type == TaskType.FACE_LANDMARKER:
                await self.analysis_engine.on_face_result(result, timestamp=timestamp)
            elif hasattr(self.analysis_engine, 'on_pose_result') and task_type == TaskType.POSE_LANDMARKER:
                await self.analysis_engine.on_pose_result(result, timestamp=timestamp)
            elif hasattr(self.analysis_engine, 'on_hand_result') and task_type == TaskType.HAND_LANDMARKER:
                await self.analysis_engine.on_hand_result(result, timestamp=timestamp)
            elif hasattr(self.analysis_engine, 'on_gesture_result') and task_type == TaskType.GESTURE_RECOGNIZER:
                await self.analysis_engine.on_gesture_result(result, timestamp=timestamp)
            elif hasattr(self.analysis_engine, 'on_object_result') and task_type == TaskType.OBJECT_DETECTOR:
                await self.analysis_engine.on_object_result(result, timestamp=timestamp)
            elif hasattr(self.analysis_engine, 'on_holistic_result') and task_type == TaskType.HOLISTIC_LANDMARKER:
                await self.analysis_engine.on_holistic_result(result, timestamp=timestamp)
        except Exception as e:
            logger.error(f"Analysis engine 전달 오류 ({task_type.value}): {e}")

    async def process_frame(self, frame: np.ndarray) -> Dict[TaskType, Any]:
        """
        프레임 처리 - 입력은 반드시 numpy(ndarray)여야 함 (UMat 금지)
        """
        if isinstance(frame, cv2.UMat):
            try:
                frame = frame.get()
            except Exception:
                pass
        # 이하 기존 numpy 처리 로직 유지
        start_time = time.time()
        self._calculate_fps()
        if self.fps_counter % 30 == 0:
            self.adjust_dynamic_resources()
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self.last_timestamp:
            timestamp_ms = self.last_timestamp + 1
        self.last_timestamp = timestamp_ms
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            processing_tasks = []
            for task_type, task in self.active_tasks.items():
                if self.task_health.get(task_type, False):
                    try:
                        if hasattr(task, 'detect_async'):
                            task.detect_async(mp_image, timestamp_ms)
                        elif hasattr(task, 'recognize_async'):
                            task.recognize_async(mp_image, timestamp_ms)
                    except Exception as e:
                        logger.warning(f"{task_type.value} 처리 오류: {e}")
                        self.task_health[task_type] = False
            processing_time = time.time() - start_time
            self.frame_processing_times.append(processing_time)
        except Exception as e:
            logger.error(f"프레임 처리 오류: {e}")
        return self.latest_results.copy()

    def _calculate_fps(self):
        """FPS 계산"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # 30프레임마다 계산
            current_time = time.time()
            self.current_fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        avg_processing_time = np.mean(self.frame_processing_times) if self.frame_processing_times else 0
        
        return {
            "fps": self.current_fps,
            "avg_processing_time_ms": avg_processing_time * 1000,
            "active_tasks": len(self.active_tasks),
            "healthy_tasks": sum(self.task_health.values()),
            "task_health": self.task_health.copy(),
            "queue_size": self.result_queue.qsize()
        }

    def get_latest_results(self) -> Dict[str, Any]:
        """최신 결과 반환 (호환성을 위해 문자열 키 사용)"""
        return {
            "face": self.latest_results.get(TaskType.FACE_LANDMARKER),
            "pose": self.latest_results.get(TaskType.POSE_LANDMARKER),
            "hand": self.latest_results.get(TaskType.HAND_LANDMARKER),
            "gesture": self.latest_results.get(TaskType.GESTURE_RECOGNIZER),
            "object": self.latest_results.get(TaskType.OBJECT_DETECTOR),
            "holistic": self.latest_results.get(TaskType.HOLISTIC_LANDMARKER)
        }

    async def close(self):
        """리소스 정리"""
        logger.info("MediaPipe Manager 정리 시작...")
        
        self.running = False
        
        # 콜백 스레드 종료
        if self.callback_thread and self.callback_thread.is_alive():
            self.result_queue.put(("shutdown", None, None))
            self.callback_thread.join(timeout=5.0)
        
        # 모든 Task 정리
        for task_type, task in self.active_tasks.items():
            try:
                if hasattr(task, 'close'):
                    task.close()
                logger.info(f"✅ {task_type.value} 정리 완료")
            except Exception as e:
                logger.warning(f"❌ {task_type.value} 정리 오류: {e}")
        
        self.active_tasks.clear()
        self.task_health.clear()
        self.latest_results.clear()
        
        logger.info("MediaPipe Manager 정리 완료")

    def adjust_dynamic_resources(self):
        """
        동적 리소스 관리: FPS, 처리시간, 큐 크기 등 실시간 성능 통계 기반으로
        frame_queue/result_queue/deque의 maxlen을 동적으로 조정
        """
        stats = self.get_performance_stats()
        fps = stats.get("fps", 0)
        avg_processing_time = stats.get("avg_processing_time_ms", 0)
        queue_size = stats.get("queue_size", 0)
        # 예시: FPS가 10 이하로 떨어지면 큐/버퍼 크기 축소, 30 이상이면 확대
        if fps < 10 or avg_processing_time > 100:
            self.result_queue.maxsize = max(3, self.result_queue.maxsize // 2)
        elif fps > 30 and avg_processing_time < 40:
            self.result_queue.maxsize = min(100, self.result_queue.maxsize * 2)
        # 필요시 각종 deque의 maxlen도 조정 가능
        # (실제 적용은 각 버퍼/큐의 구조에 맞게 추가 구현)

    # 레거시 호환성 메소드들
    def run_tasks(self, frame):
        """레거시 호환성을 위한 동기 인터페이스"""
        return asyncio.run(self.process_frame(frame))

    @property 
    def current_fps(self):
        return getattr(self, '_current_fps', 0.0)
    
    @current_fps.setter
    def current_fps(self, value):
        self._current_fps = value