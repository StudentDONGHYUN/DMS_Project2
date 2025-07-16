"""
S-Class DMS v19+ 차세대 MediaPipe Tasks Manager
최신 MediaPipe Tasks API (0.10.9+) 적용
(수정본: origin 파일의 세부 파라미터 적용)

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
from mediapipe.tasks.python.core.base_options import BaseOptions

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
    """MediaPipe Task 설정 (MediaPipe 0.10.21 API)"""

    task_type: TaskType
    model_path: str
    running_mode: str = "LIVE_STREAM"  # "IMAGE", "VIDEO", "LIVE_STREAM"

    # Face Landmarker 파라미터
    num_faces: int = 1
    output_face_blendshapes: bool = False
    output_facial_transformation_matrixes: bool = False

    # Pose Landmarker 파라미터
    num_poses: int = 1
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5  # Hand, Pose 공통 사용 가능
    output_segmentation_masks: bool = False

    # Hand Landmarker 파라미터
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5

    # Object Detector 파라미터
    max_results: int = 5
    score_threshold: float = 0.5
    category_allowlist: Optional[List[str]] = None

    # Holistic Landmarker는 별도 파라미터가 적어 생략 (필요시 추가 가능)


class AdvancedMediaPipeManager:
    """
    차세대 MediaPipe Tasks Manager - 성능 최적화
    - 최신 Tasks API 완전 활용
    - 동적 모델 로딩/언로딩
    - 성능 최적화 및 메모리 관리
    - 포괄적 오류 처리
    """

    def __init__(self, analysis_engine=None, config_file: Optional[str] = None):
        logger.info("AdvancedMediaPipeManager 초기화 시작...")

        try:
            # MediaPipe 모듈 가져오기 테스트
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions

            logger.info(f"MediaPipe 모듈 로드 성공 - 버전: {mp.__version__}")
        except ImportError as e:
            logger.error(f"MediaPipe 모듈 가져오기 실패: {e}")
            raise

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
        self.main_loop = None  # ◀◀◀ [수정] 메인 이벤트 루프를 저장할 변수 추가
        self.callback_thread = None
        self.running = False

        # 기본 설정 로드
        self._load_default_configs()

        # 외부 설정 파일 로드 (있는 경우)
        if config_file:
            self._load_config_file(config_file)

        self.is_embedded = self._detect_dsp() == "HEXAGON"

        logger.info("AdvancedMediaPipeManager v2.0 (최적화) 초기화 완료")

    def ensure_model_directory(self):
        """모델 디렉토리 존재 확인"""
        self.model_base_path.mkdir(exist_ok=True)
        logger.info(f"모델 디렉토리: {self.model_base_path.absolute()}")

    def _load_default_configs(self):
        """기본 Task 설정 로드 (origin 파일의 세부 설정 반영)"""

        # Face Landmarker 설정
        self.task_configs[TaskType.FACE_LANDMARKER] = TaskConfig(
            task_type=TaskType.FACE_LANDMARKER,
            model_path=str(self.model_base_path / "face_landmarker.task"),
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode="LIVE_STREAM",
        )

        # Pose Landmarker 설정 (origin의 full 모델 및 신뢰도 값 반영)
        self.task_configs[TaskType.POSE_LANDMARKER] = TaskConfig(
            task_type=TaskType.POSE_LANDMARKER,
            model_path=str(
                self.model_base_path / "pose_landmarker_full.task"
            ),  # 모델 경로 수정
            num_poses=1,
            min_pose_detection_confidence=0.7,  # 신뢰도 값 추가
            min_pose_presence_confidence=0.9,  # 신뢰도 값 추가
            min_tracking_confidence=0.8,  # 신뢰도 값 추가
            output_segmentation_masks=True,
            running_mode="LIVE_STREAM",
        )

        # Hand Landmarker 설정 (origin의 신뢰도 값 반영)
        self.task_configs[TaskType.HAND_LANDMARKER] = TaskConfig(
            task_type=TaskType.HAND_LANDMARKER,
            model_path=str(self.model_base_path / "hand_landmarker.task"),
            num_hands=2,
            min_hand_detection_confidence=0.4,  # 신뢰도 값 추가
            min_hand_presence_confidence=0.5,  # 신뢰도 값 추가
            min_tracking_confidence=0.7,  # 신뢰도 값 추가
            running_mode="LIVE_STREAM",
        )

        # Gesture Recognizer 설정
        self.task_configs[TaskType.GESTURE_RECOGNIZER] = TaskConfig(
            task_type=TaskType.GESTURE_RECOGNIZER,
            model_path=str(self.model_base_path / "gesture_recognizer.task"),
            num_hands=2,
            running_mode="LIVE_STREAM",
        )

        # Object Detector 설정 (origin의 허용 목록 반영)
        self.task_configs[TaskType.OBJECT_DETECTOR] = TaskConfig(
            task_type=TaskType.OBJECT_DETECTOR,
            model_path=str(self.model_base_path / "efficientdet_lite0.tflite"),
            max_results=5,
            score_threshold=0.3,
            category_allowlist=[
                "cell phone",
                "cup",
                "bottle",
                "sandwich",
                "book",
                "laptop",
            ],  # 허용 목록 추가
            running_mode="LIVE_STREAM",
        )

        # Holistic Landmarker 설정
        holistic_model_path = self.model_base_path / "holistic_landmarker.task"
        if holistic_model_path.exists():
            self.task_configs[TaskType.HOLISTIC_LANDMARKER] = TaskConfig(
                task_type=TaskType.HOLISTIC_LANDMARKER,
                model_path=str(holistic_model_path),
                # min_detection_confidence=0.5,  # TaskConfig에 없는 파라미터는 여기서 직접 사용
                # min_tracking_confidence=0.5,
            )
        else:
            logger.warning(
                f"Holistic Landmarker 모델 파일을 찾을 수 없습니다: {holistic_model_path}"
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
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read().lower()
                if "hexagon" in cpuinfo or "dsp" in cpuinfo:
                    return "HEXAGON"
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
                logger.error(f"모델 파일 없음: {config.model_path}")
                logger.error(f"현재 작업 디렉토리: {Path.cwd()}")
                logger.error(
                    f"models 디렉토리 내용: {list(self.model_base_path.glob('*'))}"
                )
                return False

            # 모델 파일 크기 확인
            model_size = Path(config.model_path).stat().st_size
            logger.info(
                f"모델 파일 로드 중: {config.model_path} ({model_size / (1024 * 1024):.2f}MB)"
            )

            # delegate 자동 분기 (공식 지원: CPU/GPU만)
            system = platform.system()
            # GPU delegate는 Linux, macOS에서만 공식 지원
            if system in ["Linux", "Darwin"]:
                base_options_kwargs = dict(model_asset_path=config.model_path)
                base_options_kwargs["delegate"] = BaseOptions.Delegate.GPU
                logger.info(f"GPU delegate 적용: {system}")
            else:
                base_options_kwargs = dict(model_asset_path=config.model_path)
                base_options_kwargs["delegate"] = BaseOptions.Delegate.CPU
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
        """Face Landmarker 초기화 (MediaPipe 0.10.21 API)"""
        from mediapipe.tasks.python import vision

        # running_mode 변환
        if hasattr(config, "running_mode") and config.running_mode == "LIVE_STREAM":
            running_mode = vision.RunningMode.LIVE_STREAM
        else:
            running_mode = vision.RunningMode.LIVE_STREAM

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=getattr(config, "num_faces", 1),
            output_face_blendshapes=getattr(config, "output_face_blendshapes", True),
            output_facial_transformation_matrixes=getattr(
                config, "output_facial_transformation_matrixes", True
            ),
            result_callback=self._create_result_callback(TaskType.FACE_LANDMARKER),
        )
        return vision.FaceLandmarker.create_from_options(options)

    async def _initialize_pose_landmarker(self, base_options, config):
        """Pose Landmarker 초기화 (세부 파라미터 적용)"""
        from mediapipe.tasks.python import vision

        running_mode = vision.RunningMode.LIVE_STREAM

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_poses=getattr(config, "num_poses", 1),
            min_pose_detection_confidence=getattr(
                config, "min_pose_detection_confidence", 0.5
            ),
            min_pose_presence_confidence=getattr(
                config, "min_pose_presence_confidence", 0.5
            ),
            min_tracking_confidence=getattr(config, "min_tracking_confidence", 0.5),
            output_segmentation_masks=getattr(
                config, "output_segmentation_masks", True
            ),
            result_callback=self._create_result_callback(TaskType.POSE_LANDMARKER),
        )
        return vision.PoseLandmarker.create_from_options(options)

    async def _initialize_hand_landmarker(self, base_options, config):
        """Hand Landmarker 초기화 (세부 파라미터 적용)"""
        from mediapipe.tasks.python import vision

        running_mode = vision.RunningMode.LIVE_STREAM

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=getattr(config, "num_hands", 2),
            min_hand_detection_confidence=getattr(
                config, "min_hand_detection_confidence", 0.5
            ),
            min_hand_presence_confidence=getattr(
                config, "min_hand_presence_confidence", 0.5
            ),
            min_tracking_confidence=getattr(config, "min_tracking_confidence", 0.5),
            result_callback=self._create_result_callback(TaskType.HAND_LANDMARKER),
        )
        return vision.HandLandmarker.create_from_options(options)

    async def _initialize_gesture_recognizer(self, base_options, config):
        """Gesture Recognizer 초기화 (MediaPipe 0.10.21 API)"""
        from mediapipe.tasks.python import vision

        # running_mode 변환
        if hasattr(config, "running_mode") and config.running_mode == "LIVE_STREAM":
            running_mode = vision.RunningMode.LIVE_STREAM
        else:
            running_mode = vision.RunningMode.LIVE_STREAM

        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=getattr(config, "num_hands", 2),
            result_callback=self._create_result_callback(TaskType.GESTURE_RECOGNIZER),
        )
        return vision.GestureRecognizer.create_from_options(options)

    async def _initialize_object_detector(self, base_options, config):
        """Object Detector 초기화 (세부 파라미터 적용)"""
        from mediapipe.tasks.python import vision

        running_mode = vision.RunningMode.LIVE_STREAM

        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=running_mode,
            max_results=getattr(config, "max_results", 5),
            score_threshold=getattr(config, "score_threshold", 0.3),
            category_allowlist=getattr(config, "category_allowlist", None),
            result_callback=self._create_result_callback(TaskType.OBJECT_DETECTOR),
        )
        return vision.ObjectDetector.create_from_options(options)

    async def _initialize_holistic_landmarker(self, base_options, config):
        """Holistic Landmarker 초기화 (MediaPipe 0.10.21 API)"""
        from mediapipe.tasks.python import vision

        # running_mode 변환
        if hasattr(config, "running_mode") and config.running_mode == "LIVE_STREAM":
            running_mode = vision.RunningMode.LIVE_STREAM
        else:
            running_mode = vision.RunningMode.LIVE_STREAM

        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            result_callback=self._create_result_callback(TaskType.HOLISTIC_LANDMARKER),
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
            TaskType.HAND_LANDMARKER,
        ]

        # 선택적 Tasks
        optional_tasks = [
            TaskType.GESTURE_RECOGNIZER,
            TaskType.OBJECT_DETECTOR,
            TaskType.HOLISTIC_LANDMARKER,
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
            logger.info("콜백 처리 스레드 시작 준비...")
            self.running = True
            self.main_loop = asyncio.get_running_loop()  # ◀◀◀ [수정] 메인 루프 가져오기
            self.callback_thread = threading.Thread(
                target=self._process_callbacks, daemon=True
            )
            self.callback_thread.start()
            logger.info("콜백 처리 스레드 시작됨")

        initialized_count = sum(results.values())
        total_count = len(results)
        logger.info(f"Task 초기화 완료: {initialized_count}/{total_count}")

        return results

    def _process_callbacks(self):
        """비동기 콜백 처리 (수정: run_coroutine_threadsafe 사용)"""
        logger.info("콜백 처리 스레드 시작")

        while self.running:
            try:
                # 큐에서 결과 가져오기
                task_type, result, timestamp = self.result_queue.get(timeout=1.0)

                if task_type == "shutdown":
                    break

                # Analysis engine으로 결과 전달 (메인 이벤트 루프에서 실행하도록 스케줄링)
                if self.analysis_engine and self.main_loop:
                    asyncio.run_coroutine_threadsafe(
                        self._forward_result_to_analysis_engine(
                            task_type, result, timestamp
                        ),
                        self.main_loop,
                    )

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"콜백 처리 스레드에서 예외 발생: {e}")

        logger.info("콜백 처리 스레드 종료")

    async def _forward_result_to_analysis_engine(
        self, task_type: TaskType, result, timestamp
    ):
        """Analysis Engine으로 결과 전달"""
        # analysis_engine이 None인 경우 스킵
        if self.analysis_engine is None:
            return

        try:
            if (
                hasattr(self.analysis_engine, "on_face_result")
                and task_type == TaskType.FACE_LANDMARKER
            ):
                await self.analysis_engine.on_face_result(result, timestamp=timestamp)
            elif (
                hasattr(self.analysis_engine, "on_pose_result")
                and task_type == TaskType.POSE_LANDMARKER
            ):
                await self.analysis_engine.on_pose_result(result, timestamp=timestamp)
            elif (
                hasattr(self.analysis_engine, "on_hand_result")
                and task_type == TaskType.HAND_LANDMARKER
            ):
                await self.analysis_engine.on_hand_result(result, timestamp=timestamp)
            elif (
                hasattr(self.analysis_engine, "on_gesture_result")
                and task_type == TaskType.GESTURE_RECOGNIZER
            ):
                await self.analysis_engine.on_gesture_result(
                    result, timestamp=timestamp
                )
            elif (
                hasattr(self.analysis_engine, "on_object_result")
                and task_type == TaskType.OBJECT_DETECTOR
            ):
                await self.analysis_engine.on_object_result(result, timestamp=timestamp)
            elif (
                hasattr(self.analysis_engine, "on_holistic_result")
                and task_type == TaskType.HOLISTIC_LANDMARKER
            ):
                await self.analysis_engine.on_holistic_result(
                    result, timestamp=timestamp
                )
        except Exception as e:
            logger.error(f"Analysis engine 전달 오류 ({task_type.value}): {e}")

    async def process_frame(self, frame: np.ndarray) -> Dict[TaskType, Any]:
        if self.is_embedded:
            return await self.process_frame_embedded(frame)
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
                        if hasattr(task, "detect_async"):
                            task.detect_async(mp_image, timestamp_ms)
                        elif hasattr(task, "recognize_async"):
                            task.recognize_async(mp_image, timestamp_ms)
                    except Exception as e:
                        logger.warning(f"{task_type.value} 처리 오류: {e}")
                        self.task_health[task_type] = False
            processing_time = time.time() - start_time
            self.frame_processing_times.append(processing_time)
        except Exception as e:
            logger.error(f"프레임 처리 오류: {e}")
        return self.latest_results.copy()

    async def process_frame_embedded(self, frame: np.ndarray) -> Dict[TaskType, Any]:
        # 1. PoseLandmarker 먼저 실행 (동기/await)
        pose_task = self.active_tasks.get(TaskType.POSE_LANDMARKER)
        if not pose_task:
            logger.error("PoseLandmarker가 초기화되지 않음")
            return {}
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        pose_result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: pose_task.detect(mp_image)
        )
        logger.info(f"pose_result: {pose_result}")
        logger.info(f"pose_landmarks: {getattr(pose_result, 'pose_landmarks', None)}")
        if hasattr(pose_result, "pose_landmarks"):
            logger.info(f"pose_landmarks length: {len(pose_result.pose_landmarks)}")

        # 2. ROI 추출
        face_roi = self.extract_face_roi(pose_result, frame)
        hand_roi = self.extract_hand_roi(pose_result, frame)
        logger.info(f"face_roi: {face_roi}, hand_roi: {hand_roi}")

        # 3. ROI crop & mp.Image 변환
        face_mp_image = self.crop_and_convert_to_mp_image(frame, face_roi)
        hand_mp_image = self.crop_and_convert_to_mp_image(frame, hand_roi)
        logger.info(
            f"face_mp_image is None: {face_mp_image is None}, hand_mp_image is None: {hand_mp_image is None}"
        )

        # 4. Face/Hand Landmarker 비동기 실행
        face_task = self.active_tasks.get(TaskType.FACE_LANDMARKER)
        hand_task = self.active_tasks.get(TaskType.HAND_LANDMARKER)
        results = {}

        async def run_landmarker(task, mp_img, task_type):
            if not task or mp_img is None:
                return None
            try:
                return await asyncio.get_event_loop().run_in_executor(
                    None, lambda: task.detect(mp_img)
                )
            except Exception as e:
                logger.error(f"{task_type.value} ROI 추론 오류: {e}")
                return None

        face_result, hand_result = await asyncio.gather(
            run_landmarker(face_task, face_mp_image, TaskType.FACE_LANDMARKER),
            run_landmarker(hand_task, hand_mp_image, TaskType.HAND_LANDMARKER),
        )
        results[TaskType.POSE_LANDMARKER] = pose_result
        results[TaskType.FACE_LANDMARKER] = face_result
        results[TaskType.HAND_LANDMARKER] = hand_result
        return results

    # mediapipe_manager.py 파일의 extract_face_roi 메서드를 아래 코드로 교체하세요.

    def extract_face_roi(self, pose_result: Any, frame: np.ndarray) -> Optional[tuple]:
        """포즈 랜드마크로부터 얼굴의 관심 영역(ROI)을 추출합니다.

        MediaPipe Pose v2 모델의 얼굴(0-10) 및 어깨(11-12) 랜드마크를
        사용하여 얼굴 영역의 경계 상자를 정밀하게 계산합니다. 이 메서드는
        얼굴의 방향(정면/측면)을 자동으로 감지하여 적합한 ROI 추출 전략을 사용합니다.

        Args:
            pose_result (Any): MediaPipe PoseLandmarker의 결과 객체.
            frame (np.ndarray): ROI를 추출할 원본 비디오 프레임.

        Returns:
            Optional[tuple]: (x_min, y_min, x_max, y_max) 형태의
            얼굴 ROI 픽셀 좌표 튜플. 감지 실패 시 None을 반환합니다.

        개선사항: 어깨 랜드마크를 참조하여 ROI의 하단 경계를 제한, 정확도 향상.
        """
        VISIBILITY_THRESHOLD = 0.65  # 랜드마크 가시성 임계값

        try:
            # 포즈 랜드마크 결과가 없으면 아무것도 하지 않음
            if not getattr(pose_result, "pose_landmarks", None):
                return None

            landmarks = pose_result.pose_landmarks[0]
            h, w = frame.shape[:2]

            # 1. 얼굴 방향 진단: 양쪽 귀의 가시성을 확인
            left_ear_visible = landmarks[7].visibility > VISIBILITY_THRESHOLD
            right_ear_visible = landmarks[8].visibility > VISIBILITY_THRESHOLD

            x_min, y_min, x_max, y_max = 0.0, 0.0, 0.0, 0.0

            # 2. 계산 전략 분기
            if left_ear_visible and right_ear_visible:
                # [전략 A: 정면] 모든 얼굴 랜드마크를 사용해 정밀 계산
                logger.debug("정면 얼굴 감지, ROI 정밀 계산")
                face_indices = list(range(11))
                face_landmarks = [landmarks[i] for i in face_indices]

                x_coords = [lm.x for lm in face_landmarks]
                y_coords = [lm.y for lm in face_landmarks]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                shoulder_y = (landmarks[11].y + landmarks[12].y) / 2

                padding_w = (x_max - x_min) * 0.15
                padding_h = (y_max - y_min) * 0.15
                x_min -= padding_w
                x_max += padding_w
                y_min -= padding_h
                y_max += padding_h
                y_max = min(y_max, shoulder_y)

            elif left_ear_visible != right_ear_visible:
                # [전략 B: 측면] 어깨너비를 기준으로 얼굴 크기 추정
                logger.debug("측면 얼굴 감지, ROI 추정 계산")
                shoulder_vec = np.array(
                    [
                        landmarks[11].x - landmarks[12].x,
                        landmarks[11].y - landmarks[12].y,
                    ]
                )
                shoulder_dist = np.linalg.norm(shoulder_vec)

                # 얼굴 크기를 어깨너비의 비율로 추정
                face_width_ratio = 0.7
                face_height_ratio = 0.8
                roi_width = shoulder_dist * face_width_ratio
                roi_height = shoulder_dist * face_height_ratio

                # 코를 중심으로 ROI 설정
                nose = landmarks[0]
                x_min = nose.x - roi_width / 2
                x_max = nose.x + roi_width / 2
                y_min = nose.y - roi_height / 2
                y_max = nose.y + roi_height / 2

            else:
                # 양쪽 귀가 모두 보이지 않으면(예: 뒷모습) ROI 추출 실패 처리
                logger.warning("얼굴 방향 특정 불가 (양쪽 귀 Failsafe)")
                return None

            # 3. 최종 좌표 변환 및 안전 확인
            x_min_px = max(0, int(x_min * w))
            x_max_px = min(w, int(x_max * w))
            y_min_px = max(0, int(y_min * h))
            y_max_px = min(h, int(y_max * h))

            if x_max_px <= x_min_px or y_max_px <= y_min_px:
                return None

            return (x_min_px, y_min_px, x_max_px, y_max_px)

        except Exception as e:
            logger.error(f"얼굴 ROI 추출 중 오류 발생: {e}")
            return None

    # mediapipe_manager.py 파일의 extract_hand_roi 메서드를 아래 코드로 교체하세요.

    def extract_hand_roi(
        self, pose_result: Any, frame: np.ndarray
    ) -> Dict[str, Optional[tuple]]:
        """
        (개선된 최종 버전) 팔과 손의 모든 가용 랜드마크를 활용하여 ROI를 정밀하게 추출합니다.

        - 손가락 랜드마크(15, 17, 19, 21 / 16, 18, 20, 22)를 이용해 기본 Bounding Box 생성
        - 팔꿈치-손목, 어깨-팔꿈치 길이를 동적으로 계산하여 ROI 여백(Padding) 조절
        - 각 랜드마크의 가시성을 개별적으로 확인하여 신뢰도 극대화

        Args:
            pose_result (Any): MediaPipe PoseLandmarker의 결과 객체.
            frame (np.ndarray): ROI를 추출할 원본 비디오 프레임.

        Returns:
            Dict[str, Optional[tuple]]: {'left': roi_tuple, 'right': roi_tuple}
        """
        VISIBILITY_THRESHOLD = 0.6
        PADDING_SCALE_FACTOR = 0.55  # 팔 길이 대비 ROI 여백 비율 (손 제스처 공간 확보)

        rois = {"left": None, "right": None}

        try:
            if not getattr(pose_result, "pose_landmarks", None):
                return rois

            landmarks = pose_result.pose_landmarks[0]
            h, w = frame.shape[:2]

            # 손/팔 랜드마크 인덱스 정의
            hand_landmark_indices = {
                "left": {
                    "shoulder": 11,
                    "elbow": 13,
                    "wrist": 15,
                    "pinky": 17,
                    "index": 19,
                    "thumb": 21,
                },
                "right": {
                    "shoulder": 12,
                    "elbow": 14,
                    "wrist": 16,
                    "pinky": 18,
                    "index": 20,
                    "thumb": 22,
                },
            }

            for hand, indices in hand_landmark_indices.items():
                # 1. 손가락/손목 랜드마크 중 보이는 것들만 수집
                hand_points = [
                    landmarks[i]
                    for i in [
                        indices["wrist"],
                        indices["pinky"],
                        indices["index"],
                        indices["thumb"],
                    ]
                    if landmarks[i].visibility > VISIBILITY_THRESHOLD
                ]

                # 손을 특정할 랜드마크가 없으면 해당 손은 건너뛰기
                if not hand_points:
                    continue

                # 2. 보이는 손 랜드마크를 모두 포함하는 최소 Bounding Box 계산
                x_coords = [p.x * w for p in hand_points]
                y_coords = [p.y * h for p in hand_points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # 3. 팔 길이를 기반으로 동적 패딩(여백) 계산
                padding = w * 0.1  # 기본 패딩값 (대체용)

                shoulder = landmarks[indices["shoulder"]]
                elbow = landmarks[indices["elbow"]]
                wrist = landmarks[indices["wrist"]]

                # 팔꿈치-손목 거리를 우선적으로 사용 (가장 정확한 스케일링)
                if (
                    elbow.visibility > VISIBILITY_THRESHOLD
                    and wrist.visibility > VISIBILITY_THRESHOLD
                ):
                    arm_segment_length = np.linalg.norm(
                        [(elbow.x - wrist.x) * w, (elbow.y - wrist.y) * h]
                    )
                    padding = arm_segment_length * PADDING_SCALE_FACTOR
                    logger.debug(f"{hand} hand: 팔꿈치-손목 거리 기반 패딩 적용.")
                # 차선책으로 어깨-팔꿈치 거리 사용
                elif (
                    shoulder.visibility > VISIBILITY_THRESHOLD
                    and elbow.visibility > VISIBILITY_THRESHOLD
                ):
                    arm_segment_length = np.linalg.norm(
                        [(shoulder.x - elbow.x) * w, (shoulder.y - elbow.y) * h]
                    )
                    padding = arm_segment_length * PADDING_SCALE_FACTOR
                    logger.debug(f"{hand} hand: 어깨-팔꿈치 거리 기반 패딩 적용.")

                # 4. Bounding Box에 패딩 적용 및 최종 ROI 좌표 계산
                x_min_pad = max(0, int(x_min - padding))
                y_min_pad = max(0, int(y_min - padding))
                x_max_pad = min(w, int(x_max + padding))
                y_max_pad = min(h, int(y_max + padding))

                if x_max_pad > x_min_pad and y_max_pad > y_min_pad:
                    rois[hand] = (x_min_pad, y_min_pad, x_max_pad, y_max_pad)

            return rois

        except Exception as e:
            logger.error(f"개선된 양손 ROI 추출 중 오류 발생: {e}")
            return {"left": None, "right": None}

    def crop_and_convert_to_mp_image(self, frame, roi):
        # ROI 영역 crop 후 mp.Image로 변환
        if roi is None:
            return None
        x_min, y_min, x_max, y_max = roi
        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return None
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)

    def _calculate_fps(self):
        """FPS 계산"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # 30프레임마다 계산
            current_time = time.time()
            self.current_fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        avg_processing_time = (
            np.mean(self.frame_processing_times) if self.frame_processing_times else 0
        )

        return {
            "fps": self.current_fps,
            "avg_processing_time_ms": avg_processing_time * 1000,
            "active_tasks": len(self.active_tasks),
            "healthy_tasks": sum(self.task_health.values()),
            "task_health": self.task_health.copy(),
            "queue_size": self.result_queue.qsize(),
        }

    def get_latest_results(self) -> Dict[str, Any]:
        """최신 결과 반환 (호환성을 위해 문자열 키 사용)"""
        return {
            "face": self.latest_results.get(TaskType.FACE_LANDMARKER),
            "pose": self.latest_results.get(TaskType.POSE_LANDMARKER),
            "hand": self.latest_results.get(TaskType.HAND_LANDMARKER),
            "gesture": self.latest_results.get(TaskType.GESTURE_RECOGNIZER),
            "object": self.latest_results.get(TaskType.OBJECT_DETECTOR),
            "holistic": self.latest_results.get(TaskType.HOLISTIC_LANDMARKER),
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
                if hasattr(task, "close"):
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
        return getattr(self, "_current_fps", 0.0)

    @current_fps.setter
    def current_fps(self, value):
        self._current_fps = value
