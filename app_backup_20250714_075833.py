
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
from systems.mediapipe_manager import EnhancedMediaPipeManager
from systems.performance import PerformanceOptimizer
from systems.personalization import PersonalizationEngine
from systems.dynamic import DynamicAnalysisEngine
from systems.backup import SensorBackupManager

# io_handler 모듈
from io_handler.video_input import VideoInputManager, MultiVideoCalibrationManager
from io_handler.ui import EnhancedUIManager

# utils 모듈 - 랜드마크 그리기 함수들
from utils.drawing import draw_face_landmarks_on_image, draw_pose_landmarks_on_image, draw_hand_landmarks_on_image
from utils.memory_monitor import MemoryMonitor, log_memory_usage

logger = logging.getLogger(__name__)


class DummyAnalysisEngine:
    def on_face_result(self, *args, **kwargs): pass
    def on_pose_result(self, *args, **kwargs): pass
    def on_hand_result(self, *args, **kwargs): pass
    def on_object_result(self, *args, **kwargs): pass
    frame_buffer = {}


class IntegratedCallbackAdapter:
    """통합 콜백 어댑터 - MediaPipe 결과를 IntegratedDMSSystem으로 전달 (수정된 버전)"""
    
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
            # Use async context manager for safer lock handling with timeout
            lock_acquisition_task = asyncio.create_task(self.processing_lock.acquire())
            try:
                await asyncio.wait_for(lock_acquisition_task, timeout=2.0)
                
                try:
                    # 버퍼 크기 관리
                    if len(self.result_buffer) >= self.MAX_BUFFER_SIZE:
                        await self._emergency_buffer_cleanup()
                    
                    if ts not in self.result_buffer:
                        self.result_buffer[ts] = {'timestamp': time.time()}
                    self.result_buffer[ts][result_type] = result
                    logger.debug(f"Received {result_type} for ts {ts}. Buffer has keys: {list(self.result_buffer[ts].keys())}")
                    
                    # 주기적 정리
                    self.buffer_cleanup_counter += 1
                    if self.buffer_cleanup_counter % 10 == 0:
                        await self._prune_buffer()
                    
                    if 'face' in self.result_buffer[ts] and 'pose' in self.result_buffer[ts]:
                        await self._process_results(ts)
                finally:
                    # Ensure lock is always released
                    self.processing_lock.release()
                    
            except asyncio.TimeoutError:
                # Cancel the acquisition task if it's still pending
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
        """긴급 버퍼 정리 - 가장 오래된 항목들을 강제로 제거"""
        if len(self.result_buffer) == 0:
            return
        
        logger.warning(f"긴급 버퍼 정리 실행 - 현재 크기: {len(self.result_buffer)}")
        
        # 타임스탬프 순으로 정렬하여 오래된 것부터 제거
        sorted_timestamps = sorted(self.result_buffer.keys())
        
        # Calculate target size (keep half of max buffer size)
        target_size = max(self.MAX_BUFFER_SIZE // 2, 1)  # Ensure at least 1 item remains
        current_size = len(self.result_buffer)
        
        if current_size <= target_size:
            # Buffer is already within target size, no cleanup needed
            logger.info(f"버퍼 크기가 이미 목표 크기 이하입니다: {current_size} <= {target_size}")
            return
        
        items_to_remove = current_size - target_size
        
        # Safety check to prevent removing more items than available
        items_to_remove = min(items_to_remove, len(sorted_timestamps))
        
        removed_count = 0
        for i in range(items_to_remove):
            if i < len(sorted_timestamps):
                ts = sorted_timestamps[i]
                if ts in self.result_buffer:  # Double-check key exists
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
    """S-Class DMS 애플리케이션 - 통합 시스템 연동 수정 버전"""

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
        
        # S-Class 시스템 구성요소들
        self.performance_monitor = PerformanceOptimizer()
        self.personalization_engine = PersonalizationEngine(user_id)
        self.dynamic_analysis = DynamicAnalysisEngine()
        self.backup_manager = SensorBackupManager()
        self.calibration_manager = MultiVideoCalibrationManager(user_id)
        
        # 메모리 모니터링 설정
        self.memory_monitor = MemoryMonitor(
            warning_threshold_mb=600,
            critical_threshold_mb=1000,
            cleanup_callback=self._perform_memory_cleanup
        )
        
        if isinstance(input_source, (list, tuple)) and len(input_source) > 1:
            self.calibration_manager.set_driver_continuity(self.is_same_driver)
        
        logger.info("[수정] S-Class 시스템 초기화 완료")

    async def initialize(self) -> bool:
        logger.info("[수정] S-Class DMS 시스템 초기화 시작...")
        try:
            # 1. 상태 관리자 초기화
            self.state_manager = EnhancedStateManager()
            
            # 2. 통합 시스템 초기화
            self.integrated_system = IntegratedDMSSystem(
                system_type=self.system_type,
                use_legacy_engine=self.use_legacy_engine
            )
            
            # 3. 비동기 컴포넌트 초기화
            await self._initialize_async_components()
            
            # 4. MediaPipe 관리자 초기화 (수정된 콜백 어댑터 사용)
            self.mediapipe_manager = EnhancedMediaPipeManager(DummyAnalysisEngine())
            
            # 5. 통합 콜백 어댑터 설정 (핵심 수정 사항)
            self.callback_adapter = IntegratedCallbackAdapter(
                self.integrated_system, 
                self.mediapipe_manager
            )
            self.mediapipe_manager.analysis_engine = self.callback_adapter
            
            if hasattr(self.mediapipe_manager, "rebind_callbacks"):
                self.mediapipe_manager.rebind_callbacks()
            
            # 6. 비디오 입력 관리자 초기화
            self.video_input_manager = VideoInputManager(self.input_source)
            if not self.video_input_manager.initialize():
                raise RuntimeError("입력 소스 초기화 실패")
            
            # 7. UI 관리자 초기화 (랜드마크 시각화를 위해 추가)
            self.ui_manager = EnhancedUIManager()
            
            logger.info("[수정] S-Class DMS 시스템 v18+ (통합 수정) 초기화 완료")
            self.initialization_completed = True
            return True
            
        except Exception as e:
            logger.error(f"[수정] 초기화 중 오류: {e}", exc_info=True)
            return False

    async def _initialize_async_components(self):
        """비동기 컴포넌트 초기화"""
        try:
            init_tasks = [
                self.integrated_system.initialize(),
                self.personalization_engine.initialize(),
                self.dynamic_analysis.initialize()
            ]
            await asyncio.gather(*init_tasks)
            logger.info("[수정] 모든 S-Class 비동기 컴포넌트 초기화 완료")
        except Exception as e:
            logger.error(f"[수정] 비동기 컴포넌트 초기화 중 오류: {e}", exc_info=True)
            raise

    def _annotate_frame_with_integrated_results(self, frame, mediapipe_results, integrated_results):
        """통합 분석 결과를 포함한 고급 프레임 어노테이션"""
        try:
            if self.ui_manager:
                # ui_manager의 draw_enhanced_results 메서드에 필요한 모든 인자를 전달
                # 현재 mediapipe_results는 analysis_engine에서 처리되므로, ui_manager에는 integrated_results만 전달
                # 필요하다면 mediapipe_results에서 직접 랜드마크를 추출하여 전달할 수도 있음
                annotated_frame = self.ui_manager.draw_enhanced_results(
                    frame,
                    integrated_results, # metrics
                    self.state_manager.get_current_state(), # state
                    mediapipe_results, # results (for landmarks)
                    None, # gaze_classifier (simplified)
                    None, # dynamic_analyzer (simplified)
                    None, # sensor_backup (simplified)
                    {"fps": getattr(self.mediapipe_manager, 'current_fps', 0.0), "system_health": 1.0}, # perf_stats
                    {"mode": "webcam"}, # playback_info (간소화)
                    None, # driver_identifier (simplified)
                    None, # predictive_safety (simplified)
                    None, # emotion_recognizer (simplified)
                )
            else:
                annotated_frame = self._create_basic_info_overlay(frame, 0) # Fallback to basic overlay
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"[수정] 통합 어노테이션 중 오류: {e}")
            return self._create_basic_info_overlay(frame, 0)

    def _draw_basic_landmarks(self, frame, mediapipe_results):
        """기본 MediaPipe 랜드마크 그리기"""
        if mediapipe_results.get('face'):
            frame = draw_face_landmarks_on_image(frame, mediapipe_results['face'])
        if mediapipe_results.get('pose'):
            frame = draw_pose_landmarks_on_image(frame, mediapipe_results['pose'])
        if mediapipe_results.get('hand'):
            frame = draw_hand_landmarks_on_image(frame, mediapipe_results['hand'])
        return frame

    def _draw_integrated_analysis_overlay(self, frame, integrated_results):
        """S-Class 통합 분석 결과 오버레이"""
        height, width = frame.shape[:2]
        
        # 위험도 점수 표시
        fatigue_score = integrated_results.get('fatigue_risk_score', 0.0)
        distraction_score = integrated_results.get('distraction_risk_score', 0.0)
        confidence = integrated_results.get('confidence_score', 0.0)
        
        # 위험도에 따른 색상 결정
        fatigue_color = self._get_risk_color(fatigue_score)
        distraction_color = self._get_risk_color(distraction_score)
        
        # 메인 위험도 표시
        cv2.putText(frame, f"Fatigue Risk: {fatigue_score:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fatigue_color, 2)
        cv2.putText(frame, f"Distraction Risk: {distraction_score:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, distraction_color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # S-Class 세부 분석 결과 표시
        y_offset = 120
        
        # 얼굴 분석 결과
        face_analysis = integrated_results.get('face_analysis', {})
        if 'rppg' in face_analysis:
            rppg_data = face_analysis['rppg']
            hr_bpm = rppg_data.get('estimated_hr_bpm', 0)
            signal_quality = rppg_data.get('signal_quality', 0)
            cv2.putText(frame, f"Heart Rate: {hr_bpm:.0f} BPM (Q:{signal_quality:.1f})", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25
        
        if 'saccade' in face_analysis:
            saccade_data = face_analysis['saccade']
            saccade_velocity = saccade_data.get('saccade_velocity_norm', 0)
            cv2.putText(frame, f"Saccade Velocity: {saccade_velocity:.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += 25
        
        if 'drowsiness' in face_analysis:
            drowsiness_data = face_analysis['drowsiness']
            status = drowsiness_data.get('status', 'unknown')
            drowsiness_confidence = drowsiness_data.get('confidence', 0)
            status_color = (0, 255, 0) if status == 'alert' else (0, 0, 255)
            cv2.putText(frame, f"Drowsiness: {status} ({drowsiness_confidence:.2f})", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            y_offset += 25
        
        # 시선 분석 결과
        if 'gaze' in face_analysis:
            gaze_data = face_analysis['gaze']
            current_zone = gaze_data.get('current_zone', 'UNKNOWN')
            attention_focus = gaze_data.get('attention_focus', 0)
            cv2.putText(frame, f"Gaze: {current_zone} (Focus:{attention_focus:.2f})", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # 시스템 상태 표시
        system_health = integrated_results.get('system_health', 'unknown')
        execution_quality = integrated_results.get('execution_quality', {})
        pipeline_mode = execution_quality.get('pipeline_mode', 'unknown')
        
        health_color = (0, 255, 0) if system_health == 'healthy' else (0, 255, 255)
        cv2.putText(frame, f"System: {system_health} | Mode: {pipeline_mode}", 
                   (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, health_color, 1)
        
        # 위험도 시각화 바
        self._draw_risk_bars(frame, fatigue_score, distraction_score)

    def _get_risk_color(self, risk_score):
        """위험도에 따른 색상 반환"""
        if risk_score >= 0.8:
            return (0, 0, 255)  # 빨간색 - 위험
        elif risk_score >= 0.6:
            return (0, 127, 255)  # 주황색 - 경고
        elif risk_score >= 0.4:
            return (0, 255, 255)  # 노란색 - 주의
        else:
            return (0, 255, 0)  # 녹색 - 안전

    def _draw_risk_bars(self, frame, fatigue_score, distraction_score):
        """위험도 시각화 바 그리기"""
        height, width = frame.shape[:2]
        
        # 피로도 바
        bar_width = 200
        bar_height = 20
        bar_x = width - bar_width - 20
        fatigue_y = 30
        
        # 배경
        cv2.rectangle(frame, (bar_x, fatigue_y), (bar_x + bar_width, fatigue_y + bar_height), 
                     (50, 50, 50), -1)
        
        # 피로도 바
        fatigue_fill = int(bar_width * min(1.0, fatigue_score))
        if fatigue_fill > 0:
            cv2.rectangle(frame, (bar_x, fatigue_y), (bar_x + fatigue_fill, fatigue_y + bar_height), 
                         self._get_risk_color(fatigue_score), -1)
        
        # 주의산만도 바
        distraction_y = fatigue_y + bar_height + 10
        cv2.rectangle(frame, (bar_x, distraction_y), (bar_x + bar_width, distraction_y + bar_height), 
                     (50, 50, 50), -1)
        
        distraction_fill = int(bar_width * min(1.0, distraction_score))
        if distraction_fill > 0:
            cv2.rectangle(frame, (bar_x, distraction_y), (bar_x + distraction_fill, distraction_y + bar_height), 
                         self._get_risk_color(distraction_score), -1)
        
        # 라벨
        cv2.putText(frame, "Fatigue", (bar_x - 80, fatigue_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Distraction", (bar_x - 80, distraction_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _create_basic_info_overlay(self, frame, frame_count):
        """기본 정보 오버레이"""
        try:
            annotated_frame = frame.copy()
            height, width = frame.shape[:2]
            
            # 기본 상태 표시
            cv2.putText(annotated_frame, "S-Class DMS System Running", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # MediaPipe 상태
            mp_status = "MediaPipe: "
            if hasattr(self, 'mediapipe_manager') and self.mediapipe_manager:
                results = self.mediapipe_manager.get_latest_results()
                if results:
                    detections = []
                    if results.get('face') and results['face'] and hasattr(results['face'], 'face_landmarks') and results['face'].face_landmarks:
                        detections.append("Face")
                    if results.get('pose') and results['pose'] and hasattr(results['pose'], 'pose_landmarks') and results['pose'].pose_landmarks:
                        detections.append("Pose")
                    if results.get('hand') and results['hand'] and hasattr(results['hand'], 'hand_landmarks') and results['hand'].hand_landmarks:
                        detections.append("Hand")
                    
                    if detections:
                        mp_status += ", ".join(detections)
                        color = (0, 255, 0)
                    else:
                        mp_status += "No Detection"
                        color = (0, 255, 255)
                else:
                    mp_status += "No Results"
                    color = (0, 0, 255)
            else:
                mp_status += "Not Available"
                color = (128, 128, 128)
            
            cv2.putText(annotated_frame, mp_status, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 프레임 카운터
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # FPS 정보 추가
            if hasattr(self, 'mediapipe_manager') and hasattr(self.mediapipe_manager, 'current_fps'):
                cv2.putText(annotated_frame, f"FPS: {self.mediapipe_manager.current_fps:.1f}", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 사용자 정보
            cv2.putText(annotated_frame, f"User: {self.user_id}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"[수정] 기본 오버레이 생성 중 오류: {e}")
            return frame

    def run(self):
        """메인 실행 메서드 (수정된 버전)"""
        self.running = True
        logger.info("[수정] S-Class DMS 시스템 시작")
        
        frame_queue = queue.Queue(maxsize=5)
        stop_event = threading.Event()
        
        def opencv_display_loop():
            """OpenCV 디스플레이 루프"""
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
                    cv2.imshow("S-Class DMS v18+ - Integrated System", last_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    stop_event.set()
                    break
                elif key == ord('s') and last_frame is not None:
                    filename = f"screenshot_{int(time.time())}.png"
                    cv2.imwrite(filename, last_frame)
                    logger.info(f"스크린샷 저장: {filename}")
            
            cv2.destroyAllWindows()
        
        async def async_frame_producer():
            """비동기 프레임 생산자 (통합 시스템 연동)"""
            await self.initialize()
            logger.info("[수정] S-Class 통합 시스템 시작")
            
            # 메모리 모니터링 시작
            self.memory_monitor.start_monitoring(interval=15.0)  # 15초마다 체크
            log_memory_usage("시스템 시작 후 ")
            
            await asyncio.sleep(0.1)  # 초기화 대기
            frame_count = 0
            
            while not stop_event.is_set():
                frame = self.video_input_manager.get_frame()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                
                cv2.flip(frame, 1, frame)
                frame_count += 1
                annotated_frame = None
                
                logger.debug(f"[Frame {frame_count}] Processing frame. Timestamp: {int(time.time() * 1000)}")

                if hasattr(self, 'mediapipe_manager') and self.mediapipe_manager:
                    try:
                        # MediaPipe 작업 실행
                        self.mediapipe_manager.run_tasks(frame.copy())
                        
                        # 콜백 처리 시간 확보
                        await asyncio.sleep(0.01) # 짧은 대기 추가
                        
                        # MediaPipe 결과 가져오기
                        mediapipe_results = self.mediapipe_manager.get_latest_results()
                        logger.debug(f"[Frame {frame_count}] MediaPipe results obtained. Face: {bool(mediapipe_results.get('face'))}, Pose: {bool(mediapipe_results.get('pose'))}")

                        # 통합 분석 결과 가져오기 (핵심 수정 사항)
                        integrated_results = None
                        if hasattr(self, 'callback_adapter'):
                            integrated_results = self.callback_adapter.get_latest_integrated_results()
                            logger.debug(f"[Frame {frame_count}] Integrated results obtained. Fatigue: {integrated_results.get('fatigue_risk_score', 0):.2f}")

                        # 결과 로깅 (10프레임마다)
                        if frame_count % 10 == 0:
                            self._log_frame_status(frame_count, mediapipe_results, integrated_results)
                        
                        # 통합 결과를 포함한 어노테이션
                        if mediapipe_results or integrated_results:
                            annotated_frame = self._annotate_frame_with_integrated_results(
                                frame, mediapipe_results, integrated_results)
                        else:
                            annotated_frame = self._create_basic_info_overlay(frame, frame_count)
                        
                    except Exception as e:
                        logger.error(f"프레임 처리 중 오류: {e}", exc_info=True)
                        annotated_frame = self._create_basic_info_overlay(frame, frame_count)
                else:
                    annotated_frame = self._create_basic_info_overlay(frame, frame_count)
                
                # 프레임 큐에 추가
                if annotated_frame is not None:
                    try:
                        frame_queue.put_nowait(annotated_frame)
                    except queue.Full:
                        try:
                            frame_queue.get_nowait() # 오래된 프레임 제거
                            frame_queue.put_nowait(annotated_frame)
                        except queue.Empty:
                            pass
                
                await asyncio.sleep(0.015)  # 15ms 대기
            
            # 종료 시그널
            try:
                frame_queue.put(None, timeout=0.1)
            except queue.Full:
                pass
        
        # 실행
        display_thread = threading.Thread(target=opencv_display_loop)
        display_thread.start()
        
        asyncio.run(async_frame_producer())
        
        stop_event.set()
        display_thread.join()

    def _log_frame_status(self, frame_count, mediapipe_results, integrated_results):
        """프레임 상태 로깅"""
        try:
            # MediaPipe 상태
            face_status = "YES" if mediapipe_results and mediapipe_results.get('face') and mediapipe_results['face'].face_landmarks else "NO"
            pose_status = "YES" if mediapipe_results and mediapipe_results.get('pose') and mediapipe_results['pose'].pose_landmarks else "NO"
            hand_status = "YES" if mediapipe_results and mediapipe_results.get('hand') and mediapipe_results['hand'].hand_landmarks else "NO"
            
            # 통합 분석 상태
            if integrated_results:
                fatigue = integrated_results.get('fatigue_risk_score', 0)
                distraction = integrated_results.get('distraction_risk_score', 0)
                confidence = integrated_results.get('confidence_score', 0)
                
                logger.info(f"Frame {frame_count}: Face={face_status}, Pose={pose_status}, Hand={hand_status} | "
                           f"Fatigue={fatigue:.2f}, Distraction={distraction:.2f}, Confidence={confidence:.2f}")
            else:
                logger.info(f"Frame {frame_count}: Face={face_status}, Pose={pose_status}, Hand={hand_status} | "
                           f"Integrated=NO_RESULTS")
        except Exception as e:
            logger.error(f"프레임 상태 로깅 중 오류: {e}")

    def _handle_keyboard_input(self, key: int) -> bool:
        """키보드 입력 처리"""
        if key == ord("q") or key == 27:
            return False
        elif key == ord(" "):
            self.paused = not self.paused
            logger.info(f"일시정지 토글: {self.paused}")
        elif key == ord("s") and self.current_processed_frame is not None:
            filename = f"captures/sclass_dms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            Path("captures").mkdir(exist_ok=True)
            cv2.imwrite(filename, self.current_processed_frame)
            logger.info(f"프레임 저장: {filename}")
        elif key == ord("i"):
            try:
                if hasattr(self.integrated_system, 'get_system_status'):
                    status = self.integrated_system.get_system_status()
                    logger.info(f"시스템 상태: {status}")
                else:
                    logger.info("시스템 상태 정보 없음")
            except Exception as e:
                logger.error(f"시스템 정보 출력 실패: {e}")
        
        return True

    def _perform_memory_cleanup(self):
        """메모리 정리 콜백 함수"""
        try:
            logger.info("사용자 정의 메모리 정리 시작...")
            
            # 통합 콜백 어댑터 버퍼 정리
            if hasattr(self, 'callback_adapter') and self.callback_adapter:
                old_size = len(self.callback_adapter.result_buffer)
                self.callback_adapter.result_buffer.clear()
                logger.info(f"콜백 어댑터 버퍼 정리: {old_size}개 항목 제거")
            
            # MediaPipe 관련 정리 (실제 존재하는 속성만 사용)
            if hasattr(self, 'mediapipe_manager') and self.mediapipe_manager:
                logger.info("MediaPipe 매니저 상태 확인 완료")
            
            # 상태 관리자 관련 정리 (실제 존재하는 속성만 사용)
            if hasattr(self, 'state_manager') and self.state_manager:
                logger.info("상태 관리자 확인 완료")
            
            log_memory_usage("정리 후 ")
            
        except Exception as e:
            logger.error(f"메모리 정리 중 오류: {e}")

    def emergency_cleanup(self):
        """긴급 메모리 정리 (메모리 모니터에서 호출)"""
        try:
            logger.critical("긴급 메모리 정리 시작...")
            
            # 기본 정리 수행
            self._perform_memory_cleanup()
            
            # 추가 긴급 조치
            if hasattr(self, 'callback_adapter'):
                self.callback_adapter.buffer_cleanup_counter = 0
                if hasattr(self.callback_adapter, 'result_buffer'):
                    self.callback_adapter.result_buffer.clear()
            
            logger.critical("긴급 메모리 정리 완료")
            
        except Exception as e:
            logger.error(f"긴급 메모리 정리 중 오류: {e}")

    async def _cleanup_async(self):
        """시스템 정리"""
        logger.info("[수정] 시스템 정리 시작")
        try:
            # 메모리 모니터링 중지
            if hasattr(self, 'memory_monitor'):
                self.memory_monitor.stop_monitoring()
            
            if hasattr(self, 'integrated_system') and hasattr(self.integrated_system, 'shutdown'):
                await self.integrated_system.shutdown()
            
            if hasattr(self, 'mediapipe_manager') and hasattr(self.mediapipe_manager, 'close'):
                self.mediapipe_manager.close()
            
            if hasattr(self, 'video_input_manager') and hasattr(self.video_input_manager, 'close'):
                self.video_input_manager.close()
                    
        except Exception as e:
            logger.error(f"정리 중 오류: {e}")
