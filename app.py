# app.py - S-Class DMS System (비동기 초기화 문제 해결)

import cv2
import time
import asyncio
from pathlib import Path
from datetime import datetime
import logging
import threading
import queue
import numpy as np

# core 모듈
from core.definitions import CameraPosition
from core.state_manager import EnhancedStateManager

# integration 모듈 - S-Class 시스템
from integration.integrated_system import IntegratedDMSSystem, AnalysisSystemType

# systems 모듈
from systems.mediapipe_manager import AdvancedMediaPipeManager
from systems.performance import PerformanceOptimizer
from systems.personalization import PersonalizationEngine
from systems.dynamic import DynamicAnalysisEngine
from systems.backup import SensorBackupManager

# io_handler 모듈
from io_handler.video_input import VideoInputManager, MultiVideoCalibrationManager
from io_handler.ui import SClassAdvancedUIManager

# utils 모듈 - 랜드마크 그리기 함수들
from utils.drawing import draw_face_landmarks_on_image, draw_pose_landmarks_on_image, draw_hand_landmarks_on_image
from utils.memory_monitor import MemoryMonitor, log_memory_usage

logger = logging.getLogger(__name__)

# 이하 app_backup_20250714_075833.py의 전체 코드 복원 (DummyAnalysisEngine, IntegratedCallbackAdapter, DMSApp 등)

class DummyAnalysisEngine:
    def on_face_result(self, *args, **kwargs): pass
    def on_pose_result(self, *args, **kwargs): pass
    def on_hand_result(self, *args, **kwargs): pass
    def on_object_result(self, *args, **kwargs): pass
    frame_buffer = {}

class IntegratedCallbackAdapter:
    """
    통합 콜백 어댑터 - MediaPipe 결과를 IntegratedDMSSystem으로 전달 (수정된 버전)
    """
    def __init__(self, integrated_system, result_target=None):
        self.integrated_system = integrated_system
        self.result_target = result_target
        self.result_buffer = {}
        self.processing_lock = asyncio.Lock()
        self.last_processed_timestamp = 0
        self.last_integrated_results = self._get_fallback_results()
        self.RESULT_TIMEOUT = 0.5 # 500ms
        self.MAX_BUFFER_SIZE = 100  # 최대 버퍼 크기
        self.buffer_cleanup_counter = 0
        logger.info("IntegratedCallbackAdapter (수정) 초기화 완료")

    async def on_face_result(self, result, timestamp=None, *args, **kwargs):
        await self._on_result('face', result, timestamp)

    async def on_pose_result(self, result, timestamp=None, *args, **kwargs):
        await self._on_result('pose', result, timestamp)

    async def on_hand_result(self, result, timestamp=None, *args, **kwargs):
        await self._on_result('hand', result, timestamp)

    async def on_object_result(self, result, timestamp=None, *args, **kwargs):
        await self._on_result('object', result, timestamp)

    async def _on_result(self, result_type, result, timestamp):
        ts = timestamp or int(time.time() * 1000)
        try:
            lock_acquisition_task = asyncio.create_task(self.processing_lock.acquire())
            try:
                await asyncio.wait_for(lock_acquisition_task, timeout=2.0)
                try:
                    if len(self.result_buffer) >= self.MAX_BUFFER_SIZE:
                        await self._emergency_buffer_cleanup()
                    if ts not in self.result_buffer:
                        self.result_buffer[ts] = {'timestamp': time.time()}
                    self.result_buffer[ts][result_type] = result
                    logger.debug(f"Received {result_type} for ts {ts}. Buffer has keys: {list(self.result_buffer[ts].keys())}")
                    self.buffer_cleanup_counter += 1
                    if self.buffer_cleanup_counter % 10 == 0:
                        await self._prune_buffer()
                    if 'face' in self.result_buffer[ts] and 'pose' in self.result_buffer[ts]:
                        await self._process_results(ts)
                finally:
                    self.processing_lock.release()
            except asyncio.TimeoutError:
                if not lock_acquisition_task.done():
                    lock_acquisition_task.cancel()
                    try:
                        await lock_acquisition_task
                    except asyncio.CancelledError:
                        pass
                raise
        except asyncio.TimeoutError:
            logger.warning(f"Lock 획득 타임아웃 - {result_type} 결과 무시됨 (ts: {ts})")
        except Exception as e:
            logger.error(f"_on_result 처리 중 오류: {e}")

    async def _process_results(self, timestamp):
        if timestamp <= self.last_processed_timestamp:
            if timestamp in self.result_buffer:
                del self.result_buffer[timestamp]
            return
        results_to_process = self.result_buffer.pop(timestamp, None)
        if not results_to_process:
            return
        logger.info(f"Processing results for timestamp {timestamp}")
        try:
            integrated_results = await self.integrated_system.process_frame(results_to_process, timestamp)
            self.last_integrated_results = integrated_results
            self.last_processed_timestamp = timestamp
        except Exception as e:
            logger.error(f"통합 분석 중 오류: {e}")
            self.last_integrated_results = self._get_fallback_results()
        await self._prune_buffer()

    async def _prune_buffer(self):
        current_time = time.time()
        keys_to_delete = [ts for ts, data in self.result_buffer.items() if current_time - data['timestamp'] > self.RESULT_TIMEOUT]
        for ts in keys_to_delete:
            logger.warning(f"Timeout for timestamp {ts}, removing from buffer.")
            del self.result_buffer[ts]

    async def _emergency_buffer_cleanup(self):
        if len(self.result_buffer) == 0:
            return
        logger.warning(f"긴급 버퍼 정리 실행 - 현재 크기: {len(self.result_buffer)}")
        sorted_timestamps = sorted(self.result_buffer.keys())
        target_size = max(self.MAX_BUFFER_SIZE // 2, 1)
        current_size = len(self.result_buffer)
        if current_size <= target_size:
            logger.info(f"버퍼 크기가 이미 목표 크기 이하입니다: {current_size} <= {target_size}")
            return
        items_to_remove = current_size - target_size
        items_to_remove = min(items_to_remove, len(sorted_timestamps))
        removed_count = 0
        for i in range(items_to_remove):
            if i < len(sorted_timestamps):
                ts = sorted_timestamps[i]
                if ts in self.result_buffer:
                    del self.result_buffer[ts]
                    removed_count += 1
        logger.info(f"긴급 정리 완료 - 제거된 항목: {removed_count}, 새 크기: {len(self.result_buffer)}")

    def get_latest_integrated_results(self):
        return self.last_integrated_results

    def _get_fallback_results(self):
        return {
            'fatigue_risk_score': 0.0,
            'distraction_risk_score': 0.0,
            'confidence_score': 0.0,
            'face_analysis': {},
            'pose_analysis': {},
            'hand_analysis': {},
            'object_analysis': {},
            'fusion_analysis': {},
            'system_health': 'unknown'
        }

class DMSApp:
    """
    S-Class DMS 애플리케이션 - 통합 시스템 연동 수정 버전
    """
    def __init__(
        self,
        input_source=0,
        user_id: str = "default",
        camera_position: CameraPosition = CameraPosition.REARVIEW_MIRROR,
        enable_calibration: bool = True,
        is_same_driver: bool = True,
        system_type: AnalysisSystemType = AnalysisSystemType.STANDARD,
        use_legacy_engine: bool = False,
        sclass_features: dict = None,
    ):
        logger.info("[수정] app_fixed.py: DMSApp.__init__ 진입")
        self.input_source = input_source
        self.user_id = user_id
        self.camera_position = camera_position
        self.enable_calibration = enable_calibration
        self.is_same_driver = is_same_driver
        self.system_type = system_type
        self.use_legacy_engine = use_legacy_engine
        self.sclass_features = sclass_features or {}
        self.running = False
        self.paused = False
        self.current_processed_frame = None
        self.initialization_completed = False
        self.safe_mode = False
        self.error_count = 0
        self.performance_monitor = PerformanceOptimizer()
        self.personalization_engine = PersonalizationEngine(user_id)
        self.dynamic_analysis = DynamicAnalysisEngine()
        self.backup_manager = SensorBackupManager()
        self.calibration_manager = MultiVideoCalibrationManager(user_id)
        self.memory_monitor = MemoryMonitor(
            warning_threshold_mb=600,
            critical_threshold_mb=1000,
            cleanup_callback=self._perform_memory_cleanup
        )
        self.ui_manager = SClassAdvancedUIManager()
        if isinstance(input_source, (list, tuple)) and len(input_source) > 1:
            self.calibration_manager.set_driver_continuity(self.is_same_driver)
        logger.info("[수정] S-Class 시스템 초기화 완료")

    async def initialize(self) -> bool:
        logger.info("[수정] S-Class DMS 시스템 초기화 시작...")
        try:
            # 1. 상태 관리자 초기화
            self.state_manager = EnhancedStateManager(self.user_id)
            # 2. 비디오 입력 초기화
            self.video_input_manager = VideoInputManager(self.input_source)
            if not self.video_input_manager.initialize():
                logger.error("비디오 입력 초기화 실패")
                return False
            # 3. 통합 분석 시스템 초기화 (파라미터 수정)
            custom_config = {
                "user_id": self.user_id,
                "camera_position": self.camera_position,
                "enable_calibration": self.enable_calibration,
                "sclass_features": self.sclass_features
            }
            self.integrated_system = IntegratedDMSSystem(
                system_type=self.system_type,
                custom_config=custom_config,
                use_legacy_engine=self.use_legacy_engine
            )
            # 4. MediaPipe 매니저 초기화
            self.mediapipe_manager = AdvancedMediaPipeManager(DummyAnalysisEngine())
            # 5. 콜백 어댑터 연결
            self.callback_adapter = IntegratedCallbackAdapter(self.integrated_system)
            logger.info("[수정] S-Class DMS 시스템 초기화 완료")
            self.initialization_completed = True
            return True
        except Exception as e:
            logger.error(f"S-Class DMS 시스템 초기화 실패: {e}", exc_info=True)
            return False

    def run(self):
        self.running = True
        logger.info("[수정] app.py: run 진입")
        import asyncio
        frame_queue = queue.Queue(maxsize=5)
        stop_event = threading.Event()
        def opencv_display_loop():
            logger.info("[수정] app.py: run - opencv_display_loop 진입")
            last_frame = None
            while not stop_event.is_set():
                try:
                    frame = frame_queue.get(timeout=0.1)
                    if frame is None:
                        break
                    last_frame = frame
                except queue.Empty:
                    pass
                if last_frame is not None:
                    try:
                        frame_to_show = cv2.UMat(last_frame) if not isinstance(last_frame, cv2.UMat) else last_frame
                    except Exception:
                        frame_to_show = last_frame
                    cv2.imshow("S-Class DMS v18+ - Research Integrated", frame_to_show)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    stop_event.set()
                    break
                elif key == ord('s'):
                    if last_frame is not None:
                        filename = f"screenshot_{int(time.time())}.png"
                        try:
                            frame_to_save = cv2.UMat(last_frame) if not isinstance(last_frame, cv2.UMat) else last_frame
                        except Exception:
                            frame_to_save = last_frame
                        cv2.imwrite(filename, frame_to_save)
            cv2.destroyAllWindows()
        async def async_frame_producer():
            logger.info("[수정] app.py: run - async_frame_producer 진입")
            await self.initialize()
            logger.info("[수정] app.py: run - S-Class DMS 시스템 초기화 완료")
            await asyncio.sleep(0.1)
            frame_count = 0
            last_perf_log_time = time.time()
            while not stop_event.is_set():
                frame = self.video_input_manager.get_frame()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                frame_count += 1
                # GEMINI.md 성능 최적화: MediaPipe 처리 전 writeable=False 적용
                if hasattr(frame, 'flags'):
                    frame.flags.writeable = False
                # (여기서 MediaPipe 처리/분석이 실제로 일어나는 경우에만 적용)
                # --- [성능 통계 수집] ---
                perf_stats = self.mediapipe_manager.get_performance_stats()
                annotated_frame = self._create_basic_info_overlay(frame, frame_count, perf_stats)
                if annotated_frame is not None:
                    try:
                        frame_queue.put_nowait(annotated_frame)
                    except queue.Full:
                        # 프레임 드롭/스킵: 가장 오래된 프레임을 버리고 최신 프레임만 유지
                        try:
                            frame_queue.get_nowait()
                            frame_queue.put_nowait(annotated_frame)
                        except queue.Empty:
                            pass
                # --- [성능 최적화 자동 호출] ---
                if frame_count % 30 == 0:
                    processing_time = 0.0  # 실제 처리 시간 측정 필요시 측정값 사용
                    fps = perf_stats.get("fps", 0.0)
                    self.performance_monitor.log_performance(processing_time, fps)
                    self.mediapipe_manager.adjust_dynamic_resources()
                    self._perform_memory_cleanup()
                await asyncio.sleep(0.010)
            try:
                frame_queue.put(None, timeout=0.1)
            except queue.Full:
                pass
        display_thread = threading.Thread(target=opencv_display_loop)
        display_thread.start()
        asyncio.run(async_frame_producer())
        stop_event.set()
        display_thread.join()

    def _create_basic_info_overlay(self, frame, frame_count, perf_stats=None):
        # Ensure overlay is drawn on UMat
        try:
            annotated_frame = frame if isinstance(frame, cv2.UMat) else cv2.UMat(frame)
        except Exception:
            annotated_frame = frame
        height, width = annotated_frame.get().shape[:2] if isinstance(annotated_frame, cv2.UMat) else annotated_frame.shape[:2]
        # 예시: 프레임 번호 및 FPS 표시
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if perf_stats is not None:
            fps = perf_stats.get("fps", 0.0)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        return annotated_frame

    def _perform_memory_cleanup(self):
        logger.info("메모리 정리 실행")