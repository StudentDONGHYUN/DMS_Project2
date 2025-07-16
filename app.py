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

# ✅ FIXED: OpenCV 안전 처리 추가
from utils.opencv_safe import (
    OpenCVSafeHandler,
    safe_create_basic_info_overlay,
    safe_frame_preprocessing_for_mediapipe,
)

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
            logger.error(f"_on_result 처리 중 오류: {e}", exc_info=True)

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
            logger.error(f"통합 분석 중 오류: {e}", exc_info=True)
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
            self.state_manager = EnhancedStateManager()
            self.video_input_manager = VideoInputManager(self.input_source)
            if not self.video_input_manager.initialize():
                logger.error("비디오 입력 초기화 실패")
                return False
            from events.event_bus import initialize_event_system

            try:
                initialize_event_system()  # 기존 동기 호출
                logger.info("✅ 이벤트 시스템 초기화 완료")
            except Exception as e:
                logger.error(f"❌ 이벤트 시스템 초기화 실패: {e}")
                logger.warning("이벤트 시스템 없이 안전 모드로 계속 진행")
            # 1. 통합 시스템 인스턴스 생성
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
            # 2. 비동기 초기화
            await self.integrated_system.initialize()
            # 3. 나머지 컴포넌트 초기화
            self.mediapipe_manager = AdvancedMediaPipeManager(DummyAnalysisEngine())
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

        # ✅ FIXED: 손실된 참조 변수 추가
        self.frame_queue = frame_queue
        self.stop_event = stop_event

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
                        # ✅ FIXED: UMat 변환 제거 - 이미 numpy array로 처리됨
                        frame_to_show = last_frame
                        
                        # 프레임 검증
                        if frame_to_show is None:
                            logger.error("frame_to_show is None!")
                            continue
                            
                        if not isinstance(frame_to_show, np.ndarray):
                            logger.error(f"frame_to_show 타입 오류: {type(frame_to_show)}")
                            continue
                            
                        if frame_to_show.ndim != 3 or frame_to_show.shape[2] != 3:
                            logger.error(f"frame_to_show shape 오류: {frame_to_show.shape}")
                            continue
                            
                        # 안전한 화면 표시
                        cv2.imshow("S-Class DMS v18+ - Research Integrated", frame_to_show)
                        
                    except Exception as display_e:
                        logger.error(f"화면 표시 오류: {display_e}")
                        # 에러가 발생해도 계속 진행
                        
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_event.set()
                    break
                elif key == ord("s"):
                    if last_frame is not None:
                        filename = f"screenshot_{int(time.time())}.png"
                        try:
                            cv2.imwrite(filename, last_frame)
                            logger.info(f"스크린샷 저장: {filename}")
                        except Exception as save_e:
                            logger.error(f"스크린샷 저장 실패: {save_e}")
                            
            cv2.destroyAllWindows()

        async def async_frame_producer():
            """
            ✅ FIXED: 통합된 비동기 프레임 처리 파이프라인
            """
            logger.info("[수정] 비동기 프레임 프로듀서 시작")

            # 초기화 및 준비 대기
            await self.initialize()
            logger.info("[수정] S-Class DMS 시스템 초기화 완료")

            # 시스템 안정화를 위한 짧은 대기
            await asyncio.sleep(0.1)

            # 프레임 처리 루프 변수
            frame_count = 0
            consecutive_errors = 0
            max_consecutive_errors = 5

            # 성능 모니터링 변수
            from collections import deque

            frame_processing_times = deque(maxlen=100)

            logger.info("비동기 프레임 처리 루프 시작")

            try:
                while not stop_event.is_set():
                    loop_start_time = time.time()

                    try:
                        # === 1. 프레임 획득 (비동기화) ===
                        frame = await self._async_get_frame()
                        if frame is None:
                            await asyncio.sleep(0.01)  # 짧은 대기 후 재시도
                            continue

                        frame_count += 1

                        # === 2. 프레임 처리 (완전 비동기) ===
                        annotated_frame = (
                            await self._safe_process_frame_with_error_recovery(
                                frame, frame_count
                            )
                        )

                        # === 3. 프레임 큐에 추가 (비블로킹) ===
                        await self._async_enqueue_frame(annotated_frame)

                        # === 4. 성능 모니터링 및 최적화 ===
                        frame_time = time.time() - loop_start_time
                        frame_processing_times.append(frame_time)

                        # 30프레임마다 성능 로깅 및 최적화
                        if frame_count % 30 == 0:
                            await self._async_performance_optimization(
                                frame_processing_times
                            )

                        # 연속 오류 카운터 리셋
                        consecutive_errors = 0

                        # 적응형 대기 시간 (목표 FPS 기준)
                        target_frame_time = 1.0 / 60.0  # 60 FPS 목표
                        remaining_time = target_frame_time - frame_time
                        if remaining_time > 0:
                            await asyncio.sleep(remaining_time)

                    except (asyncio.CancelledError, KeyboardInterrupt):
                        logger.info("Frame processing loop cancelled.")
                        break
                    except Exception as e:
                        logger.info("비동기 프레임 처리 루프 종료")
                        break

                logger.info("비동기 프레임 처리 루프 종료")

                # 정리 신호 전송
                try:
                    frame_queue.put(None, timeout=0.1)
                except queue.Full:
                    pass

            finally:
                # 정리 작업
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
        """
        ✅ FIXED: 안전한 기본 오버레이 생성 (기존 UMat 오류 해결)
        """
        try:
            # 안전한 오버레이 생성 사용
            return safe_create_basic_info_overlay(frame, frame_count, perf_stats)
        except (cv2.error, TypeError, ValueError) as e:
            logger.error(f"안전한 오버레이 생성 실패: {e}")
            # 최종 폴백: 원본 프레임 반환 또는 폴백 프레임 생성
            if frame is not None:
                return frame
            else:
                return OpenCVSafeHandler.create_fallback_frame()

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
                except (AttributeError, TypeError) as e:
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
                except (AttributeError, TypeError, RuntimeError) as e:
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
                except (AttributeError, TypeError) as e:
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
            except (AttributeError, TypeError) as e:
                logger.warning(f"컴포넌트 메모리 정리 중 오류: {e}")

            # 6. OpenCV 메모리 정리
            try:
                import cv2

                # OpenCV 내부 캐시 정리 (가능한 경우)
                if hasattr(cv2, "setUseOptimized"):
                    cv2.setUseOptimized(True)  # 최적화 재활성화
                logger.debug("OpenCV 메모리 정리 완료")
            except (cv2.error, AttributeError) as e:
                logger.debug(f"OpenCV 메모리 정리 중 오류: {e}")

            # 7. NumPy 메모리 정리
            try:
                import numpy as np

                # NumPy 메모리 풀 정리 (가능한 경우)
                if hasattr(np, "clear_cache"):
                    np.clear_cache()
                    logger.debug("NumPy 캐시 정리 완료")
            except AttributeError as e:
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

        except (ImportError, gc.error) as e:
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
        except (ImportError, psutil.Error) as e:
            logger.error(f"메모리 사용량 조회 실패: {e}")
            return 0.0

    # ✅ FIXED: 비동기 처리를 위한 헬퍼 함수들 추가

    async def _async_get_frame(self):
        """
        ✅ 비동기 프레임 획득
        """
        try:
            # 동기 get_frame()을 executor에서 실행하여 비동기화
            loop = asyncio.get_running_loop()
            frame = await loop.run_in_executor(None, self.video_input_manager.get_frame)
            return frame
        except (AttributeError, TypeError) as e:
            logger.debug(f"프레임 획득 실패: {e}")
            return None

    async def _async_enqueue_frame(self, frame):
        """
        ✅ 비동기 프레임 큐 추가
        """
        try:
            # 큐가 가득 찬 경우 오래된 프레임 제거
            if hasattr(self, "frame_queue"):
                try:
                    # 비블로킹으로 큐에 추가 시도
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # 큐가 가득 찬 경우 가장 오래된 프레임 제거 후 추가
                    try:
                        self.frame_queue.get_nowait()  # 오래된 프레임 제거
                        self.frame_queue.put_nowait(frame)  # 새 프레임 추가
                    except queue.Empty:
                        # 큐가 비어있는 경우 그냥 추가
                        self.frame_queue.put_nowait(frame)
        except (AttributeError, queue.Full) as e:
            logger.warning(f"프레임 큐 추가 실패: {e}")

    async def _async_performance_optimization(self, frame_processing_times):
        """
        ✅ 비동기 성능 최적화
        """
        try:
            # 평균 프레임 처리 시간 계산
            import numpy as np

            avg_processing_time = (
                np.mean(frame_processing_times) if frame_processing_times else 0.0
            )
            current_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0.0

            # 성능 통계 로깅
            logger.info(
                f"성능 통계 - FPS: {current_fps:.1f}, 평균 처리시간: {avg_processing_time * 1000:.1f}ms"
            )

            # MediaPipe 리소스 동적 조정
            if hasattr(self, "mediapipe_manager"):
                self.mediapipe_manager.adjust_dynamic_resources()

            # 메모리 정리 (executor에서 실행)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._perform_memory_cleanup)

            # 성능 모니터 업데이트
            if hasattr(self, "performance_monitor") and self.performance_monitor:
                self.performance_monitor.log_performance(
                    avg_processing_time, current_fps
                )

        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"성능 최적화 실패: {e}")

    async def _async_error_recovery(self):
        """
        ✅ 비동기 오류 복구
        """
        logger.info("시스템 오류 복구 시작...")

        try:
            # 1. MediaPipe 시스템 상태 점검
            if hasattr(self, "mediapipe_manager"):
                try:
                    health_check = self.mediapipe_manager.get_performance_stats()
                    logger.info(f"MediaPipe 상태: {health_check}")
                except Exception as e:
                    logger.warning(f"MediaPipe 상태 점검 실패: {e}")

            # 2. 통합 시스템 상태 점검
            if hasattr(self, "integrated_system"):
                try:
                    # 이벤트 시스템 재초기화 시도
                    from events.event_bus import initialize_event_system

                    initialize_event_system()
                    logger.info("이벤트 시스템 재초기화 완료")
                except Exception as e:
                    logger.warning(f"이벤트 시스템 재초기화 실패: {e}")

            # 3. 메모리 강제 정리
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._perform_aggressive_memory_cleanup)

            # 4. 복구 대기 시간
            await asyncio.sleep(1.0)

            logger.info("시스템 오류 복구 완료")

        except Exception as e:
            logger.error(f"오류 복구 실패: {e}", exc_info=True)

    async def _safe_process_frame_with_error_recovery(self, frame, frame_count):
        """
        ✅ FIXED: 오류 복구 기능이 있는 안전한 프레임 처리 (완전 수정)
        """
        try:
            # 1. 입력 프레임 검증
            if frame is None:
                logger.warning("None 프레임 입력")
                return OpenCVSafeHandler.create_fallback_frame()

            # 2. MediaPipe 전처리 (안전한 처리)
            try:
                preprocessed_frame = safe_frame_preprocessing_for_mediapipe(frame)
            except Exception as e:
                logger.warning(f"MediaPipe 전처리 실패: {e}")
                preprocessed_frame = frame

            # 3. MediaPipe 결과 획득
            try:
                mediapipe_results = await self.mediapipe_manager.process_frame(
                    preprocessed_frame
                )
            except Exception as e:
                logger.warning(f"MediaPipe 처리 실패: {e}")
                # 빈 결과로 계속 진행
                mediapipe_results = {
                    'face': None,
                    'pose': None, 
                    'hand': None,
                    'object': None
                }

            # 4. 프레임 데이터 구조 준비 (이중 파이프라인)
            frame_data = {
                "image": preprocessed_frame,  # MediaPipe용 numpy 배열
                "frame": preprocessed_frame,  # 호환성을 위한 별칭
                "visualization_frame": None,  # UMat은 통합 시스템에서 생성
                "timestamp": time.time(),
            }

            # MediaPipe 결과와 프레임 데이터 통합
            frame_data.update(mediapipe_results)

            # 5. 통합 분석 시스템으로 처리
            try:
                analysis_results = await self.integrated_system.process_and_annotate_frame(
                    frame_data, time.time()
                )
            except Exception as e:
                logger.warning(f"통합 시스템 처리 실패: {e}")
                # 기본 결과 구조 생성
                analysis_results = {
                    'fatigue_risk_score': 0.0,
                    'distraction_risk_score': 0.0,
                    'confidence_score': 0.0,
                    'system_health': 'error',
                    'visualization': None
                }

            # 6. 시각화 프레임 처리 (안전한 추출)
            annotated_frame = None
            
            # 시각화 프레임이 분석 결과에 있는지 확인
            if 'visualization' in analysis_results and analysis_results['visualization'] is not None:
                annotated_frame = analysis_results['visualization']
            else:
                # 시각화 프레임이 없으면 기본 오버레이 생성
                try:
                    perf_stats = None
                    if hasattr(self, "mediapipe_manager"):
                        perf_stats = self.mediapipe_manager.get_performance_stats()
                    
                    annotated_frame = safe_create_basic_info_overlay(
                        frame, frame_count, perf_stats
                    )
                    
                    # 시스템 상태 표시
                    if analysis_results.get('system_health') == 'error':
                        annotated_frame = OpenCVSafeHandler.safe_frame_annotation(
                            annotated_frame,
                            "SYSTEM ERROR - SAFE MODE",
                            position=(10, 120),
                            color=(0, 0, 255),  # 빨간색
                            font_scale=0.7,
                        )
                        
                except Exception as viz_e:
                    logger.error(f"기본 오버레이 생성 실패: {viz_e}")
                    annotated_frame = frame

            # 7. 프레임 타입 안전 처리 (UMat 오류 해결)
            if annotated_frame is not None:
                try:
                    # UMat을 numpy array로 변환 (display에 안전)
                    if isinstance(annotated_frame, cv2.UMat):
                        # UMat.get()으로 numpy array 추출
                        annotated_frame = annotated_frame.get()
                        
                    # numpy array 검증
                    if not isinstance(annotated_frame, np.ndarray):
                        logger.warning(f"예상치 못한 프레임 타입: {type(annotated_frame)}")
                        annotated_frame = frame
                        
                    # 프레임 차원 검증
                    if annotated_frame.ndim != 3 or annotated_frame.shape[2] != 3:
                        logger.warning(f"잘못된 프레임 형태: {annotated_frame.shape}")
                        annotated_frame = frame
                        
                except Exception as conv_e:
                    logger.error(f"프레임 타입 변환 실패: {conv_e}")
                    annotated_frame = frame

            return annotated_frame

        except Exception as e:
            logger.error(f"프레임 처리 중 치명적 오류: {e}", exc_info=True)
            
            # 최종 폴백: 기본 정보 오버레이 생성
            try:
                fallback_frame = safe_create_basic_info_overlay(frame, frame_count, None)
                
                # 오류 상태 표시
                error_frame = OpenCVSafeHandler.safe_frame_annotation(
                    fallback_frame,
                    f"CRITICAL ERROR: {str(e)[:30]}...",
                    position=(10, 90),
                    color=(0, 0, 255),  # 빨간색
                    font_scale=0.6,
                )
                
                # UMat을 numpy로 변환
                if isinstance(error_frame, cv2.UMat):
                    error_frame = error_frame.get()
                    
                return error_frame
                
            except Exception as final_e:
                logger.error(f"최종 폴백도 실패: {final_e}")
                # 최종 폴백: 원본 프레임 또는 검은 화면
                if frame is not None:
                    return frame
                else:
                    return OpenCVSafeHandler.create_fallback_frame()

    def _perform_aggressive_memory_cleanup(self):
        """공격적 메모리 정리 (동기 함수)"""
        try:
            import gc

            # 가비지 컶렉션 강제 실행
            collected = gc.collect()
            logger.info(f"가비지 컶렉션으로 {collected}개 객체 정리")

            # MediaPipe 결과 버퍼 정리
            if hasattr(self, "callback_adapter") and self.callback_adapter:
                if hasattr(self.callback_adapter, "result_buffer"):
                    self.callback_adapter.result_buffer.clear()
                    logger.info("MediaPipe 결과 버퍼 정리 완료")

            # 프레임 큐 정리
            if hasattr(self, "frame_queue"):
                queue_size = self.frame_queue.qsize()
                if queue_size > 0:
                    # 큐의 절반 정리
                    for _ in range(queue_size // 2):
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    logger.info(f"프레임 큐 정리: {queue_size // 2}개 프레임 제거")

        except (ImportError, gc.error, AttributeError, queue.Empty) as e:
            logger.warning(f"공격적 메모리 정리 실패: {e}")
