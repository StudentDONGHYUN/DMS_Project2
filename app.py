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
from utils.drawing import (
    draw_face_landmarks_on_image,
    draw_pose_landmarks_on_image,
    draw_hand_landmarks_on_image,
)
from utils.memory_monitor import MemoryMonitor, log_memory_usage

logger = logging.getLogger(__name__)

# 이하 app_backup_20250714_075833.py의 전체 코드 복원 (DummyAnalysisEngine, IntegratedCallbackAdapter, DMSApp 등)


class DummyAnalysisEngine:
    def on_face_result(self, *args, **kwargs):
        pass

    def on_pose_result(self, *args, **kwargs):
        pass

    def on_hand_result(self, *args, **kwargs):
        pass

    def on_object_result(self, *args, **kwargs):
        pass

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
        self.RESULT_TIMEOUT = 0.5  # 500ms
        self.MAX_BUFFER_SIZE = 100  # 최대 버퍼 크기
        self.buffer_cleanup_counter = 0
        logger.info("IntegratedCallbackAdapter (수정) 초기화 완료")

    async def on_face_result(self, result, timestamp=None, *args, **kwargs):
        await self._on_result("face", result, timestamp)

    async def on_pose_result(self, result, timestamp=None, *args, **kwargs):
        await self._on_result("pose", result, timestamp)

    async def on_hand_result(self, result, timestamp=None, *args, **kwargs):
        await self._on_result("hand", result, timestamp)

    async def on_object_result(self, result, timestamp=None, *args, **kwargs):
        await self._on_result("object", result, timestamp)

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
                        self.result_buffer[ts] = {"timestamp": time.time()}
                    self.result_buffer[ts][result_type] = result
                    logger.debug(
                        f"Received {result_type} for ts {ts}. Buffer has keys: {list(self.result_buffer[ts].keys())}"
                    )
                    self.buffer_cleanup_counter += 1
                    if self.buffer_cleanup_counter % 10 == 0:
                        await self._prune_buffer()
                    if (
                        "face" in self.result_buffer[ts]
                        and "pose" in self.result_buffer[ts]
                    ):
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
            integrated_results = await self.integrated_system.process_frame(
                results_to_process, timestamp
            )
            self.last_integrated_results = integrated_results
            self.last_processed_timestamp = timestamp
        except Exception as e:
            logger.error(f"통합 분석 중 오류: {e}")
            self.last_integrated_results = self._get_fallback_results()
        await self._prune_buffer()

    async def _prune_buffer(self):
        current_time = time.time()
        keys_to_delete = [
            ts
            for ts, data in self.result_buffer.items()
            if current_time - data["timestamp"] > self.RESULT_TIMEOUT
        ]
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
            logger.info(
                f"버퍼 크기가 이미 목표 크기 이하입니다: {current_size} <= {target_size}"
            )
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
        logger.info(
            f"긴급 정리 완료 - 제거된 항목: {removed_count}, 새 크기: {len(self.result_buffer)}"
        )

    def get_latest_integrated_results(self):
        return self.last_integrated_results

    def _get_fallback_results(self):
        return {
            "fatigue_risk_score": 0.0,
            "distraction_risk_score": 0.0,
            "confidence_score": 0.0,
            "face_analysis": {},
            "pose_analysis": {},
            "hand_analysis": {},
            "object_analysis": {},
            "fusion_analysis": {},
            "system_health": "unknown",
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
        enable_performance_optimization: bool = True,
        edition: str = "RESEARCH",  # 🆕 추가된 파라미터
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
        self.edition = edition  # 🆕 edition 저장
        # 🆕 edition에 따른 기능 활성화/비활성화 처리
        self._configure_edition_features()

        self.running = False
        self.paused = False
        self.current_processed_frame = None
        self.initialization_completed = False
        self.safe_mode = False
        self.error_count = 0
        # GUI/설정에서 enable_performance_optimization 값을 받아서 전달
        self.performance_monitor = PerformanceOptimizer(
            enable_optimization=enable_performance_optimization
        )
        self.personalization_engine = PersonalizationEngine(user_id)
        self.dynamic_analysis = DynamicAnalysisEngine()
        self.backup_manager = SensorBackupManager()
        self.calibration_manager = MultiVideoCalibrationManager(user_id)
        self.memory_monitor = MemoryMonitor(
            warning_threshold_mb=600,
            critical_threshold_mb=1000,
            cleanup_callback=self._perform_memory_cleanup,
        )
        self.ui_manager = SClassAdvancedUIManager()
        if isinstance(input_source, (list, tuple)) and len(input_source) > 1:
            self.calibration_manager.set_driver_continuity(self.is_same_driver)
        logger.info("[수정] S-Class 시스템 초기화 완료")

    def _configure_edition_features(self):
        """에디션에 따른 기능 설정"""
        logger.info(f"에디션 '{self.edition}'에 따른 기능 설정 중...")

        if self.edition == "COMMUNITY":
            # 커뮤니티 에디션: 기본 기능만
            self.sclass_features.update(
                {
                    "enable_rppg": False,
                    "enable_saccade": False,
                    "enable_spinal_analysis": False,
                    "enable_tremor_fft": False,
                    "enable_bayesian_prediction": False,
                    "enable_emotion_ai": False,
                    "enable_predictive_safety": False,
                    "enable_biometric_fusion": False,
                    "enable_adaptive_thresholds": False,
                }
            )

        elif self.edition == "PRO":
            # 프로 에디션: S-Class 고급 기능 포함
            self.sclass_features.update(
                {
                    "enable_rppg": True,
                    "enable_saccade": True,
                    "enable_spinal_analysis": True,
                    "enable_tremor_fft": True,
                    "enable_bayesian_prediction": True,
                    "enable_emotion_ai": False,
                    "enable_predictive_safety": False,
                    "enable_biometric_fusion": False,
                    "enable_adaptive_thresholds": False,
                }
            )

        elif self.edition == "ENTERPRISE":
            # 엔터프라이즈 에디션: Neural AI 포함
            self.sclass_features.update(
                {
                    "enable_rppg": True,
                    "enable_saccade": True,
                    "enable_spinal_analysis": True,
                    "enable_tremor_fft": True,
                    "enable_bayesian_prediction": True,
                    "enable_emotion_ai": True,
                    "enable_predictive_safety": True,
                    "enable_biometric_fusion": True,
                    "enable_adaptive_thresholds": True,
                }
            )

        elif self.edition == "RESEARCH":
            # 연구용 에디션: 모든 기능 활성화
            self.sclass_features.update(
                {
                    "enable_rppg": True,
                    "enable_saccade": True,
                    "enable_spinal_analysis": True,
                    "enable_tremor_fft": True,
                    "enable_bayesian_prediction": True,
                    "enable_emotion_ai": True,
                    "enable_predictive_safety": True,
                    "enable_biometric_fusion": True,
                    "enable_adaptive_thresholds": True,
                }
            )

        logger.info(f"에디션 '{self.edition}' 기능 설정 완료: {self.sclass_features}")

    async def initialize(self) -> bool:
        logger.info("[수정] S-Class DMS 시스템 초기화 시작...")
        try:
            # 1. 상태 관리자 초기화
            self.state_manager = EnhancedStateManager()

            # 2. 비디오 입력 초기화
            self.video_input_manager = VideoInputManager(self.input_source)
            if not self.video_input_manager.initialize():
                logger.error("비디오 입력 초기화 실패")
                return False

            # 🆕 3. 이벤트 시스템 초기화 (통합 시스템 초기화 전에 반드시 실행)
            from events.event_bus import initialize_event_system

            await initialize_event_system()
            logger.info("✅ 이벤트 시스템 초기화 완료")

            # 4. 통합 분석 시스템 초기화
            custom_config = {
                "user_id": self.user_id,
                "camera_position": self.camera_position,
                "enable_calibration": self.enable_calibration,
                "sclass_features": self.sclass_features,
            }
            self.integrated_system = IntegratedDMSSystem(
                system_type=self.system_type,
                custom_config=custom_config,
                use_legacy_engine=self.use_legacy_engine,
            )
            # ... 나머지 초기화 코드
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
                        frame_to_show = (
                            cv2.UMat(last_frame)
                            if not isinstance(last_frame, cv2.UMat)
                            else last_frame
                        )
                    except Exception:
                        frame_to_show = last_frame
                    cv2.imshow("S-Class DMS v18+ - Research Integrated", frame_to_show)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_event.set()
                    break
                elif key == ord("s"):
                    if last_frame is not None:
                        filename = f"screenshot_{int(time.time())}.png"
                        try:
                            frame_to_save = (
                                cv2.UMat(last_frame)
                                if not isinstance(last_frame, cv2.UMat)
                                else last_frame
                            )
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
            try:
                while not stop_event.is_set():
                    frame = self.video_input_manager.get_frame()  # 항상 numpy
                    if frame is None:
                        await asyncio.sleep(0.01)
                        continue
                    frame_count += 1
                    # GEMINI.md 성능 최적화: MediaPipe 처리 전 writeable=False 적용
                    if hasattr(frame, "flags"):
                        frame.flags.writeable = False
                        # MediaPipe 처리 및 통합 분석 시스템 실행
                        try:
                            # 1. MediaPipe 결과 획득
                            mediapipe_results = (
                                await self.mediapipe_manager.process_frame(frame)
                            )
                            # 2. 통합 분석 시스템으로 처리 및 시각화
                            annotated_frame = (
                                await self.integrated_system.process_and_annotate_frame(
                                    mediapipe_results, time.time()
                                )
                            )
                            # 3. 기본 정보 오버레이 추가
                            if annotated_frame is not None:
                                # annotated_frame이 numpy array일 때만 UMat 변환
                                import numpy as np

                                if isinstance(annotated_frame, np.ndarray):
                                    try:
                                        annotated_frame = cv2.UMat(annotated_frame)
                                    except Exception as e:
                                        logger.warning(
                                            f"UMat 변환 실패, numpy array 사용: {e}"
                                        )
                                        # 변환 실패 시 annotated_frame은 그대로 numpy array
                                elif isinstance(annotated_frame, cv2.UMat):
                                    pass  # 이미 UMat이면 변환하지 않음
                                else:
                                    logger.warning(
                                        "annotated_frame이 numpy array도 UMat도 아님!"
                                    )
                                # 프레임 정보 오버레이
                                if isinstance(annotated_frame, (np.ndarray, cv2.UMat)):
                                    try:
                                        cv2.putText(
                                            annotated_frame,
                                            f"Frame: {frame_count}",
                                            (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1,
                                            (0, 255, 0),
                                            2,
                                        )
                                    except Exception as e:
                                        logger.warning(f"텍스트 오버레이 실패: {e}")
                                else:
                                    logger.warning(
                                        "putText 대상이 numpy array도 UMat도 아님!"
                                    )
                            else:
                                # 폴백: 기본 오버레이만 표시
                                annotated_frame = self._create_basic_info_overlay(
                                    frame, frame_count, perf_stats=None
                                )
                        except Exception as e:
                            logger.error(f"MediaPipe 분석 오류: {e}")
                            # 오류 발생시 기본 오버레이 표시
                            annotated_frame = self._create_basic_info_overlay(
                                frame, frame_count, perf_stats=None
                            )
                    if annotated_frame is not None:
                        try:
                            frame_queue.put_nowait(annotated_frame)
                        except queue.Full:
                            try:
                                frame_queue.get_nowait()
                                frame_queue.put_nowait(annotated_frame)
                            except queue.Empty:
                                pass
                    # --- [성능 최적화 자동 호출] ---
                    if frame_count % 30 == 0:
                        processing_time = 0.0  # 실제 처리 시간 측정 필요시 측정값 사용
                        fps = 0.0
                        self.performance_monitor.log_performance(processing_time, fps)
                        self.mediapipe_manager.adjust_dynamic_resources()
                        self._perform_memory_cleanup()
                    await asyncio.sleep(0.010)
                try:
                    frame_queue.put(None, timeout=0.1)
                except queue.Full:
                    pass
            finally:
                # 🆕 정리 작업 추가 (동일 루프에서 await로 처리)
                try:
                    if hasattr(self, "mediapipe_manager"):
                        await self.mediapipe_manager.close()
                    if hasattr(self, "integrated_system"):
                        await self.integrated_system.shutdown()
                except Exception as e:
                    logger.warning(f"정리 작업 중 오류: {e}")

        try:
            # 기존 실행 코드
            display_thread = threading.Thread(target=opencv_display_loop)
            display_thread.start()
            asyncio.run(async_frame_producer())
            stop_event.set()
            display_thread.join()
        finally:
            # finally 블록에서는 더 이상 asyncio.run()을 호출하지 않음
            pass

    def _create_basic_info_overlay(self, frame, frame_count, perf_stats=None):
        try:
            # 안전한 UMat 변환
            if isinstance(frame, cv2.UMat):
                annotated_frame = frame
            else:
                try:
                    annotated_frame = cv2.UMat(frame)
                except Exception:
                    # UMat 변환 실패시 numpy array 그대로 사용
                    annotated_frame = frame

            # 프레임 번호 표시
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # 성능 정보 표시
            if perf_stats is not None:
                fps = perf_stats.get("fps", 0.0)
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )

            return annotated_frame

        except Exception as e:
            logger.error(f"기본 오버레이 생성 실패: {e}")
            # 최종 폴백: 원본 프레임 반환
            return frame

    def _perform_memory_cleanup(self):
        """DMS 시스템 메모리 정리 작업"""
        logger.info("메모리 정리 실행")

        try:
            # 1. 가비지 컬렉션 실행
            import gc

            before_cleanup = self._get_memory_usage()

            # 모든 세대의 가비지 컬렉션 실행
            collected_objects = 0
            for generation in range(3):
                collected = gc.collect(generation)
                collected_objects += collected
                if collected > 0:
                    logger.debug(f"GC 세대 {generation}: {collected}개 객체 정리")

            logger.info(f"가비지 컬렉션 완료: 총 {collected_objects}개 객체 정리")

            # 2. MediaPipe 관련 메모리 정리
            if hasattr(self, "mediapipe_manager") and self.mediapipe_manager:
                try:
                    # MediaPipe 결과 버퍼 정리
                    if hasattr(self.mediapipe_manager, "clear_result_buffers"):
                        self.mediapipe_manager.clear_result_buffers()
                        logger.debug("MediaPipe 결과 버퍼 정리 완료")

                    # MediaPipe 내부 캐시 정리
                    if hasattr(self.mediapipe_manager, "cleanup_cache"):
                        self.mediapipe_manager.cleanup_cache()
                        logger.debug("MediaPipe 캐시 정리 완료")
                except Exception as e:
                    logger.warning(f"MediaPipe 메모리 정리 중 오류: {e}")

            # 3. 통합 시스템 콜백 어댑터 정리
            if hasattr(self, "callback_adapter") and self.callback_adapter:
                try:
                    # 결과 버퍼 정리
                    if hasattr(self.callback_adapter, "result_buffer"):
                        buffer_size = len(self.callback_adapter.result_buffer)
                        self.callback_adapter.result_buffer.clear()
                        logger.debug(
                            f"콜백 어댑터 버퍼 정리: {buffer_size}개 항목 제거"
                        )

                    # 긴급 버퍼 정리 실행
                    if hasattr(self.callback_adapter, "_emergency_buffer_cleanup"):
                        import asyncio

                        if asyncio.iscoroutinefunction(
                            self.callback_adapter._emergency_buffer_cleanup
                        ):
                            # 비동기 함수인 경우 동기적으로 실행
                            loop = None
                            try:
                                loop = asyncio.get_running_loop()
                            except RuntimeError:
                                pass

                            if loop and loop.is_running():
                                # 이미 실행 중인 루프가 있는 경우 태스크로 생성
                                asyncio.create_task(
                                    self.callback_adapter._emergency_buffer_cleanup()
                                )
                            else:
                                # 새 루프에서 실행
                                asyncio.run(
                                    self.callback_adapter._emergency_buffer_cleanup()
                                )
                        else:
                            self.callback_adapter._emergency_buffer_cleanup()
                        logger.debug("콜백 어댑터 긴급 정리 완료")
                except Exception as e:
                    logger.warning(f"콜백 어댑터 메모리 정리 중 오류: {e}")

            # 4. 통합 시스템 메모리 정리
            if hasattr(self, "integrated_system") and self.integrated_system:
                try:
                    # 분석 엔진 메모리 정리
                    if hasattr(self.integrated_system, "analysis_engine"):
                        engine = self.integrated_system.analysis_engine

                        # 각 분석기의 메모리 정리
                        cleanup_methods = [
                            "clear_buffers",
                            "cleanup_memory",
                            "reset_buffers",
                            "clear_cache",
                            "cleanup_resources",
                        ]

                        for method_name in cleanup_methods:
                            if hasattr(engine, method_name):
                                try:
                                    method = getattr(engine, method_name)
                                    method()
                                    logger.debug(f"분석 엔진 {method_name} 완료")
                                except Exception as e:
                                    logger.debug(f"분석 엔진 {method_name} 실패: {e}")

                    # 통합 시스템 전체 정리
                    if hasattr(self.integrated_system, "cleanup_memory"):
                        self.integrated_system.cleanup_memory()
                        logger.debug("통합 시스템 메모리 정리 완료")
                except Exception as e:
                    logger.warning(f"통합 시스템 메모리 정리 중 오류: {e}")

            # 5. 프레임 버퍼 및 큐 정리
            try:
                # 현재 처리된 프레임 정리
                if hasattr(self, "current_processed_frame"):
                    self.current_processed_frame = None
                    logger.debug("현재 프레임 버퍼 정리 완료")

                # 시스템 구성요소들의 메모리 정리
                components = [
                    "performance_monitor",
                    "personalization_engine",
                    "dynamic_analysis",
                    "backup_manager",
                    "calibration_manager",
                ]

                for component_name in components:
                    if hasattr(self, component_name):
                        component = getattr(self, component_name)

                        # 각 컴포넌트의 정리 메서드 호출
                        cleanup_methods = ["cleanup_memory", "clear_cache", "reset"]
                        for method_name in cleanup_methods:
                            if hasattr(component, method_name):
                                try:
                                    method = getattr(component, method_name)
                                    if callable(method):
                                        method()
                                        logger.debug(
                                            f"{component_name}.{method_name} 완료"
                                        )
                                except Exception as e:
                                    logger.debug(
                                        f"{component_name}.{method_name} 실패: {e}"
                                    )
            except Exception as e:
                logger.warning(f"컴포넌트 메모리 정리 중 오류: {e}")

            # 6. OpenCV 메모리 정리
            try:
                import cv2

                # OpenCV 내부 캐시 정리 (가능한 경우)
                if hasattr(cv2, "setUseOptimized"):
                    cv2.setUseOptimized(True)  # 최적화 재활성화
                logger.debug("OpenCV 메모리 정리 완료")
            except Exception as e:
                logger.debug(f"OpenCV 메모리 정리 중 오류: {e}")

            # 7. NumPy 메모리 정리
            try:
                import numpy as np

                # NumPy 메모리 풀 정리 (가능한 경우)
                if hasattr(np, "clear_cache"):
                    np.clear_cache()
                    logger.debug("NumPy 캐시 정리 완료")
            except Exception as e:
                logger.debug(f"NumPy 메모리 정리 중 오류: {e}")

            # 8. 최종 가비지 컬렉션
            final_collected = gc.collect()
            if final_collected > 0:
                logger.debug(f"최종 가비지 컬렉션: {final_collected}개 객체 정리")

            # 9. 메모리 사용량 확인 및 로깅
            after_cleanup = self._get_memory_usage()
            memory_freed = before_cleanup - after_cleanup

            logger.info(
                f"메모리 정리 완료 - "
                f"정리 전: {before_cleanup:.1f}MB, "
                f"정리 후: {after_cleanup:.1f}MB, "
                f"확보된 메모리: {memory_freed:.1f}MB"
            )

            # 10. 메모리 정리 효과가 미미한 경우 경고
            if memory_freed < 10.0:  # 10MB 미만
                logger.warning(f"메모리 정리 효과가 제한적입니다: {memory_freed:.1f}MB")

                # 추가 정리 작업 제안
                if after_cleanup > 1000:  # 1GB 이상 사용 중
                    logger.warning(
                        "높은 메모리 사용량 지속 - 시스템 재시작을 고려하세요"
                    )

            return {
                "before_mb": before_cleanup,
                "after_mb": after_cleanup,
                "freed_mb": memory_freed,
                "objects_collected": collected_objects + final_collected,
            }

        except Exception as e:
            logger.error(f"메모리 정리 중 예외 발생: {e}", exc_info=True)
            return {
                "error": str(e),
                "before_mb": 0,
                "after_mb": 0,
                "freed_mb": 0,
                "objects_collected": 0,
            }

    def _get_memory_usage(self) -> float:
        """현재 메모리 사용량 반환 (MB)"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception as e:
            logger.error(f"메모리 사용량 조회 실패: {e}")
            return 0.0
