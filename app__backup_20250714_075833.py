# app.py - S-Class DMS System (비동기 초기화 문제 해결)

import cv2
import time
import asyncio
from pathlib import Path
from datetime import datetime
import logging
import threading
import queue

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

# utils 모듈
logger = logging.getLogger(__name__)


class DummyAnalysisEngine:
    def on_face_result(self, *args, **kwargs): pass
    def on_pose_result(self, *args, **kwargs): pass
    def on_hand_result(self, *args, **kwargs): pass
    def on_object_result(self, *args, **kwargs): pass
    frame_buffer = {}

# app.py - 개선된 SafeOrchestratorCallbackAdapter

class SafeOrchestratorCallbackAdapter:
    """개선된 안전한 오케스트레이터 콜백 어댑터 - DRY 원칙 적용 및 성능 최적화"""
    
    def __init__(self, orchestrator, result_target=None):
        self.orchestrator = orchestrator
        self.result_target = result_target
        self.frame_buffer = getattr(orchestrator, 'frame_buffer', {})
        self.callback_errors = 0
        self.max_consecutive_errors = 5
        self.is_degraded = False
        
        # 콜백 매핑 테이블로 중복 코드 제거
        self.callback_mapping = {
            'face': {
                'processor_key': 'face',
                'result_setter': self._set_face_result,
                'requires_image': True
            },
            'pose': {
                'processor_key': 'pose',
                'result_setter': self._set_pose_result,
                'requires_image': False
            },
            'hand': {
                'processor_key': 'hand',
                'result_setter': self._set_hand_result,
                'requires_image': False
            },
            'object': {
                'processor_key': 'object',
                'result_setter': self._set_object_result,
                'requires_image': False
            }
        }

    async def _safe_callback_execution(self, callback_name: str, callback_func, *args, **kwargs):
        """개선된 안전한 콜백 실행 - 회로차단기 패턴 적용"""
        
        # 회로차단기: 연속 오류가 많으면 일시적으로 차단
        if self.is_degraded:
            logger.debug(f"{callback_name} 콜백 차단됨 (degraded mode)")
            return False
            
        try:
            if asyncio.iscoroutinefunction(callback_func):
                await callback_func(*args, **kwargs)
            else:
                callback_func(*args, **kwargs)
            
            # 성공 시 오류 카운터 리셋
            self.callback_errors = max(0, self.callback_errors - 1)
            logger.debug(f"{callback_name} 콜백 실행 성공")
            return True
            
        except Exception as e:
            self.callback_errors += 1
            logger.warning(f"{callback_name} 콜백 실행 실패: {e} (오류 {self.callback_errors}/{self.max_consecutive_errors})")
            
            # 임계값 초과 시 degraded mode 활성화
            if self.callback_errors >= self.max_consecutive_errors:
                self.is_degraded = True
                logger.error(f"{callback_name} 콜백 차단 활성화 - 시스템 안정성 우선")
                
            return False

    async def _generic_callback(self, callback_type: str, result, timestamp=None, mediapipe_results=None, *args, **kwargs):
        """통합된 콜백 처리 메서드 - 코드 중복 제거"""
        
        if callback_type not in self.callback_mapping:
            logger.warning(f"알 수 없는 콜백 타입: {callback_type}")
            return
            
        config = self.callback_mapping[callback_type]
        
        # 결과 저장 (result_target이 있는 경우)
        if self.result_target:
            try:
                config['result_setter'](result)
            except Exception as e:
                logger.warning(f"{callback_type} result 저장 실패: {e}")
        
        # 프로세서 실행
        if (hasattr(self.orchestrator, 'processors') and 
            config['processor_key'] in self.orchestrator.processors):
            
            processor = self.orchestrator.processors[config['processor_key']]
            
            # 인자 준비
            args_list = [result]
            if config['requires_image'] and mediapipe_results and 'image' in mediapipe_results:
                args_list.append(mediapipe_results['image'])
            if timestamp is not None:
                args_list.append(timestamp)
            
            await self._safe_callback_execution(
                callback_type, processor.process_data, *args_list
            )

    # 결과 저장 메서드들 (간소화)
    def _set_face_result(self, result):
        if hasattr(self.result_target, 'set_last_face_result'):
            self.result_target.set_last_face_result(result)
            
    def _set_pose_result(self, result):
        if hasattr(self.result_target, 'set_last_pose_result'):
            self.result_target.set_last_pose_result(result)
            
    def _set_hand_result(self, result):
        if hasattr(self.result_target, 'set_last_hand_result'):
            self.result_target.set_last_hand_result(result)
            
    def _set_object_result(self, result):
        if hasattr(self.result_target, 'set_last_object_result'):
            self.result_target.set_last_object_result(result)

    # 개별 콜백 메서드들 (통합된 로직 사용)
    async def on_face_result(self, result, timestamp=None, mediapipe_results=None, *args, **kwargs):
        await self._generic_callback('face', result, timestamp, mediapipe_results, *args, **kwargs)

    async def on_pose_result(self, result, timestamp=None, *args, **kwargs):
        await self._generic_callback('pose', result, timestamp, None, *args, **kwargs)

    async def on_hand_result(self, result, timestamp=None, *args, **kwargs):
        await self._generic_callback('hand', result, timestamp, None, *args, **kwargs)

    async def on_object_result(self, result, timestamp=None, *args, **kwargs):
        await self._generic_callback('object', result, timestamp, None, *args, **kwargs)

    def get_status(self):
        """어댑터 상태 정보 반환"""
        return {
            'callback_errors': self.callback_errors,
            'is_degraded': self.is_degraded,
            'max_consecutive_errors': self.max_consecutive_errors,
            'available_callbacks': list(self.callback_mapping.keys())
        }

    def reset_degraded_mode(self):
        """Degraded mode 수동 리셋"""
        self.is_degraded = False
        self.callback_errors = 0
        logger.info("SafeOrchestratorCallbackAdapter degraded mode 리셋됨")


class DMSApp:
    """S-Class DMS 애플리케이션 - 연구 결과 통합 버전 (비동기 초기화 수정)"""

    def __init__(
        self,
        input_source=0,
        user_id: str = "default",
        camera_position: CameraPosition = CameraPosition.REARVIEW_MIRROR,
        enable_calibration: bool = True,
        is_same_driver: bool = True,
        system_type: AnalysisSystemType = AnalysisSystemType.STANDARD,
        use_legacy_engine: bool = False,  # S-Class 시스템을 기본으로 사용
        sclass_features: dict = None,
    ):
        logger.info("[진단] app.py: DMSApp.__init__ 진입")
        logger.info(f"[진단] app.py: DMSApp.__init__ - input_source={input_source}")
        self.input_source = input_source
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.input_source={self.input_source}")
        logger.info(f"[진단] app.py: DMSApp.__init__ - user_id={user_id}")
        self.user_id = user_id
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.user_id={self.user_id}")
        logger.info(f"[진단] app.py: DMSApp.__init__ - camera_position={camera_position}")
        self.camera_position = camera_position
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.camera_position={self.camera_position}")
        logger.info(f"[진단] app.py: DMSApp.__init__ - enable_calibration={enable_calibration}")
        self.enable_calibration = enable_calibration
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.enable_calibration={self.enable_calibration}")
        logger.info(f"[진단] app.py: DMSApp.__init__ - is_same_driver={is_same_driver}")
        self.is_same_driver = is_same_driver
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.is_same_driver={self.is_same_driver}")
        logger.info(f"[진단] app.py: DMSApp.__init__ - system_type={system_type}")
        self.system_type = system_type
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.system_type={self.system_type}")
        logger.info(f"[진단] app.py: DMSApp.__init__ - use_legacy_engine={use_legacy_engine}")
        self.use_legacy_engine = use_legacy_engine
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.use_legacy_engine={self.use_legacy_engine}")
        logger.info(f"[진단] app.py: DMSApp.__init__ - sclass_features={sclass_features}")
        self.sclass_features = sclass_features or {}
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.sclass_features={self.sclass_features}")
        self.running = False
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.running={self.running}")
        self.paused = False
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.paused={self.paused}")
        self.current_processed_frame = None
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.current_processed_frame={self.current_processed_frame}")
        self.initialization_completed = False
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.initialization_completed={self.initialization_completed}")
        self.safe_mode = False
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.safe_mode={self.safe_mode}")
        self.error_count = 0
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.error_count={self.error_count}")
        logger.info("[진단] app.py: DMSApp.__init__ - S-Class 시스템 초기화 시작")
        self.performance_monitor = PerformanceOptimizer()
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.performance_monitor={self.performance_monitor}")
        self.personalization_engine = PersonalizationEngine(user_id)
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.personalization_engine={self.personalization_engine}")
        self.dynamic_analysis = DynamicAnalysisEngine()
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.dynamic_analysis={self.dynamic_analysis}")
        self.backup_manager = SensorBackupManager()
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.backup_manager={self.backup_manager}")
        self.calibration_manager = MultiVideoCalibrationManager(user_id)
        logger.info(f"[진단] app.py: DMSApp.__init__ - self.calibration_manager={self.calibration_manager}")
        if isinstance(input_source, (list, tuple)) and len(input_source) > 1:
            logger.info("[진단] app.py: DMSApp.__init__ - input_source가 다중 입력, set_driver_continuity 호출 전")
            self.calibration_manager.set_driver_continuity(self.is_same_driver)
            logger.info("[진단] app.py: DMSApp.__init__ - set_driver_continuity 호출 완료")
        logger.info("[진단] app.py: DMSApp.__init__ - 캘리브레이션 매니저 초기화 완료")

    async def initialize(self) -> bool:
        logger.info("[진단] app.py: initialize 진입")
        try:
            logger.info("[진단] app.py: initialize - S-Class DMS 시스템 초기화 시작...")
            self.state_manager = EnhancedStateManager()
            logger.info(f"[진단] app.py: initialize - self.state_manager={self.state_manager}")
            self.integrated_system = IntegratedDMSSystem(
                system_type=self.system_type,
                use_legacy_engine=self.use_legacy_engine
            )
            logger.info(f"[진단] app.py: initialize - self.integrated_system={self.integrated_system}")
            logger.info("[진단] app.py: initialize - 비동기 컴포넌트 초기화 await 전")
            await self._initialize_async_components()
            logger.info("[진단] app.py: initialize - 비동기 컴포넌트 초기화 완료")
            self.mediapipe_manager = EnhancedMediaPipeManager(DummyAnalysisEngine())
            logger.info(f"[진단] app.py: initialize - self.mediapipe_manager(더미)={self.mediapipe_manager}")
            callback_adapter = SafeOrchestratorCallbackAdapter(self.integrated_system.analysis_orchestrator, self.mediapipe_manager)
            logger.info(f"[진단] app.py: initialize - callback_adapter={callback_adapter}")
            self.mediapipe_manager.analysis_engine = callback_adapter
            logger.info(f"[진단] app.py: initialize - self.mediapipe_manager.analysis_engine={self.mediapipe_manager.analysis_engine}")
            if hasattr(self.mediapipe_manager, "rebind_callbacks"):
                logger.info("[진단] app.py: initialize - 콜백 재설정 전")
                self.mediapipe_manager.rebind_callbacks()
                logger.info("[진단] app.py: initialize - 콜백 재설정 완료")
            self.video_input_manager = VideoInputManager(self.input_source)
            logger.info(f"[진단] app.py: initialize - self.video_input_manager={self.video_input_manager}")
            if not self.video_input_manager.initialize():
                logger.error("[진단] app.py: initialize - 입력 소스 초기화 실패")
                raise RuntimeError("입력 소스 초기화 실패")
            logger.info("[진단] app.py: initialize - S-Class DMS 시스템 v18+ (연구 결과 통합) 초기화 완료")
            logger.info("[진단] app.py: initialize - S-Class 기능/어텐션/인지 부하/적응형 파이프라인 활성화")
            self.initialization_completed = True
            logger.info(f"[진단] app.py: initialize - self.initialization_completed={self.initialization_completed}")
            return True
        except Exception as e:
            logger.error(f"[진단] app.py: initialize 예외: {e}", exc_info=True)
            return False

    async def _initialize_async_components(self):
        logger.info("[진단] app.py: _initialize_async_components 진입")
        try:
            logger.info("[진단] app.py: _initialize_async_components - init_tasks 리스트 생성 전")
            init_tasks = [
                self.integrated_system.initialize(),
                self.personalization_engine.initialize(),
                self.dynamic_analysis.initialize()
            ]
            logger.info(f"[진단] app.py: _initialize_async_components - init_tasks={init_tasks}")
            logger.info("[진단] app.py: _initialize_async_components - asyncio.gather await 전")
            await asyncio.gather(*init_tasks)
            logger.info("[진단] app.py: _initialize_async_components - 모든 S-Class 비동기 컴포넌트 초기화 완료")
        except Exception as e:
            logger.error(f"[진단] app.py: _initialize_async_components 예외: {e}", exc_info=True)
            raise

    async def main_async_loop(self):
        logger.info("[진단] app.py: main_async_loop 진입")
        logger.info("[진단] app.py: main_async_loop - last_displayed_frame = None 할당 전")
        last_displayed_frame = None
        logger.info(f"[진단] app.py: main_async_loop - last_displayed_frame = {last_displayed_frame} 할당 완료")
        logger.info("[진단] app.py: main_async_loop - frame_count = 0 할당 전")
        frame_count = 0
        logger.info(f"[진단] app.py: main_async_loop - frame_count = {frame_count} 할당 완료")
        logger.info("[진단] app.py: main_async_loop - try(OpenCV 창 초기화) 진입 전")
        try:
            logger.info("[진단] app.py: main_async_loop - OpenCV 창 초기화 시도")
            cv2.namedWindow("S-Class DMS v18+ - Research Integrated", cv2.WINDOW_AUTOSIZE)
            logger.info("[진단] app.py: main_async_loop - OpenCV 창 초기화 완료")
        except Exception as e:
            logger.error(f"[진단] app.py: main_async_loop - OpenCV 창 초기화 실패: {e}")
        logger.info("[진단] app.py: main_async_loop - try(메인 루프) 진입 전")
        try:
            logger.info(f"[진단] app.py: main_async_loop - self.running = {self.running} 체크 전")
            while self.running:
                logger.info(f"[진단] app.py: main_async_loop - while 루프 진입, frame_count={frame_count}")
                logger.info("[진단] app.py: main_async_loop - queue_processed_this_loop = False 할당 전")
                queue_processed_this_loop = False
                logger.info(f"[진단] app.py: main_async_loop - queue_processed_this_loop = {queue_processed_this_loop} 할당 완료")
                logger.info(f"[진단] app.py: main_async_loop - if not self.video_input_manager.is_running() 체크 전")
                if not self.video_input_manager.is_running():
                    logger.info("[진단] app.py: main_async_loop - 비디오 입력 종료됨, 루프 break")
                    break
                logger.info(f"[진단] app.py: main_async_loop - self.paused = {self.paused} 체크 전")
                if self.paused:
                    logger.info(f"[진단] app.py: main_async_loop - 일시정지 상태, last_displayed_frame={last_displayed_frame}")
                    if last_displayed_frame is not None:
                        logger.info("[진단] app.py: main_async_loop - 일시정지 프레임 표시 전")
                        cv2.imshow("S-Class DMS v18+ - Research Integrated", last_displayed_frame)
                        logger.info("[진단] app.py: main_async_loop - 일시정지 프레임 표시 완료")
                    logger.info("[진단] app.py: main_async_loop - 일시정지 키 입력 대기 전")
                    key = cv2.waitKey(30)
                    logger.info(f"[진단] app.py: main_async_loop - 일시정지 키 입력: {key}")
                    if not self._handle_keyboard_input(key):
                        logger.info("[진단] app.py: main_async_loop - 일시정지 중 사용자 종료 요청, 루프 break")
                        break
                    logger.info("[진단] app.py: main_async_loop - 일시정지 continue")
                    continue
                logger.info(f"[진단] app.py: main_async_loop - if self.video_input_manager.is_running() 체크 전")
                if self.video_input_manager.is_running():
                    try:
                        logger.info("[진단] app.py: main_async_loop - 프레임 획득 시도")
                        original_frame = self.video_input_manager.get_frame()
                        logger.info(f"[진단] app.py: main_async_loop - 프레임 획득 성공, type={type(original_frame)}, shape={getattr(original_frame, 'shape', None)}")
                    except Exception as e:
                        logger.error(f"[진단] app.py: main_async_loop - 프레임 획득 중 오류: {e}")
                        self.error_count += 1
                        logger.info(f"[진단] app.py: main_async_loop - error_count 증가: {self.error_count}")
                        await asyncio.sleep(0.01)
                        logger.info("[진단] app.py: main_async_loop - 프레임 획득 실패 continue")
                        continue
                    logger.info(f"[진단] app.py: main_async_loop - original_frame is not None 체크 전")
                    if original_frame is not None:
                        frame_count += 1
                        logger.info(f"[진단] app.py: main_async_loop - frame_count 증가: {frame_count}")
                        timestamp = time.time()
                        logger.info(f"[진단] app.py: main_async_loop - timestamp: {timestamp}")
                        logger.info(f"[진단] app.py: main_async_loop - self.mediapipe_manager={self.mediapipe_manager}")
                        logger.info(f"[진단] app.py: main_async_loop - self.paused={getattr(self, 'paused', None)}")
                        logger.info(f"[진단] app.py: main_async_loop - original_frame type={type(original_frame)}, shape={getattr(original_frame, 'shape', None)}")
                        logger.info(f"[진단] app.py: main_async_loop - if not self.mediapipe_manager 체크 전")
                        if not self.mediapipe_manager:
                            logger.info("[진단] app.py: main_async_loop - self.mediapipe_manager가 None, continue")
                            continue
                        logger.info(f"[진단] app.py: main_async_loop - if getattr(self, 'paused', False) 체크 전")
                        if getattr(self, 'paused', False):
                            logger.info("[진단] app.py: main_async_loop - self.paused=True, continue")
                            continue
                        logger.info(f"[진단] app.py: main_async_loop - if original_frame is None 체크 전")
                        if original_frame is None:
                            logger.info("[진단] app.py: main_async_loop - original_frame is None, continue")
                            continue
                        logger.info("[진단] app.py: main_async_loop - run_tasks 호출 준비")
                        try:
                            logger.info("[진단] app.py: main_async_loop - hasattr(self.mediapipe_manager, 'run_tasks') 체크 전")
                            if hasattr(self.mediapipe_manager, 'run_tasks'):
                                logger.info("[진단] app.py: main_async_loop - run_tasks 호출 전")
                                self.mediapipe_manager.run_tasks(original_frame.copy())
                                logger.info("[진단] app.py: main_async_loop - run_tasks 호출 완료")
                            else:
                                logger.warning("[진단] app.py: main_async_loop - MediaPipe 매니저에 run_tasks 메서드가 없습니다")
                        except Exception as e:
                            logger.error(f"[진단] app.py: main_async_loop - run_tasks 예외 발생: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                        queue_processed_this_loop = True
                        logger.info(f"[진단] app.py: main_async_loop - queue_processed_this_loop True로 변경")
                        logger.info("[진단] app.py: main_async_loop - 분석/시각화 continue")
                        continue
                logger.info(f"[진단] app.py: main_async_loop - if last_displayed_frame is not None 체크 전")
                if last_displayed_frame is not None:
                    try:
                        logger.info(f"[진단] app.py: main_async_loop - 프레임 오버레이 추가 전, frame_count={frame_count}")
                        self._add_basic_performance_overlay(last_displayed_frame, frame_count)
                        logger.info(f"[진단] app.py: main_async_loop - 프레임 오버레이 추가 완료, frame_count={frame_count}")
                        logger.info(f"[진단] app.py: main_async_loop - 프레임 화면 표시 전, frame_count={frame_count}")
                        cv2.imshow("S-Class DMS v18+ - Research Integrated", last_displayed_frame)
                        logger.info(f"[진단] app.py: main_async_loop - 프레임 화면 표시 완료, frame_count={frame_count}")
                        if frame_count % 10 == 0:
                            logger.info(f"[진단] app.py: main_async_loop - 비디오 재생 중: {frame_count}프레임 처리됨")
                    except Exception as e:
                        logger.error(f"[진단] app.py: main_async_loop - 프레임 표시 실패: {e}")
                else:
                    logger.info("[진단] app.py: main_async_loop - 표시할 프레임 없음, 대기 전")
                    await asyncio.sleep(0.01)
                    logger.info("[진단] app.py: main_async_loop - 표시할 프레임 없음, 대기 완료")
                logger.info("[진단] app.py: main_async_loop - 키 입력 대기 전")
                key = cv2.waitKey(1) & 0xFF
                logger.info(f"[진단] app.py: main_async_loop - 키 입력: {key}")
                if key != 255:
                    logger.info("[진단] app.py: main_async_loop - 키 입력 분기: key != 255")
                    if not self._handle_keyboard_input(key):
                        logger.info("[진단] app.py: main_async_loop - 사용자 종료 요청, 루프 break")
                        break
                logger.info("[진단] app.py: main_async_loop - await asyncio.sleep(0.001) 전")
                # 프레임 처리 주기 조절 (너무 빠른 처리로 인한 동기화 문제 방지)
                await asyncio.sleep(0.008)  # 8ms 대기로 안정성 향상
                logger.info("[진단] app.py: main_async_loop - 루프 끝, 다음 반복")
        except KeyboardInterrupt:
            logger.info("[진단] app.py: main_async_loop - 사용자 중단(KeyboardInterrupt)")
        except Exception as e:
            logger.error(f"[진단] app.py: main_async_loop 예외: {e}", exc_info=True)
        finally:
            logger.info("[진단] app.py: main_async_loop - finally: _cleanup_async 호출 전")
            await self._cleanup_async()
            logger.info("[진단] app.py: main_async_loop - finally: _cleanup_async 호출 완료")

    def _annotate_frame_with_results(self, frame, mediapipe_results):
        """기존 UI 대신 직접 랜드마크 그리기로 개선"""
        logger.info("[진단] app.py: _annotate_frame_with_results 진입")
        
        if not mediapipe_results or (isinstance(mediapipe_results, dict) and not any(mediapipe_results.values())):
            logger.info("[진단] app.py: _annotate_frame_with_results - mediapipe_results가 None 또는 비어있음")
            return self._create_basic_annotation(frame, {})
        
        try:
            annotated_frame = frame.copy()
            height, width = frame.shape[:2]
            
            # Face landmarks 그리기
            face_result = mediapipe_results.get('face')
            if face_result and hasattr(face_result, 'face_landmarks') and face_result.face_landmarks:
                logger.info("[진단] Face landmarks 그리기 시작")
                for landmarks in face_result.face_landmarks:
                    for landmark in landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        cv2.circle(annotated_frame, (x, y), 1, (0, 255, 0), -1)
                    
                    # 얼굴 바운딩 박스 그리기
                    x_coords = [landmark.x * width for landmark in landmarks]
                    y_coords = [landmark.y * height for landmark in landmarks]
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, "FACE", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Pose landmarks 그리기
            pose_result = mediapipe_results.get('pose')
            if pose_result and hasattr(pose_result, 'pose_landmarks') and pose_result.pose_landmarks:
                logger.info("[진단] Pose landmarks 그리기 시작")
                for landmarks in pose_result.pose_landmarks:
                    for landmark in landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        cv2.circle(annotated_frame, (x, y), 3, (255, 0, 0), -1)
                cv2.putText(annotated_frame, "POSE DETECTED", (10, height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Hand landmarks 그리기
            hand_result = mediapipe_results.get('hand')
            if hand_result and hasattr(hand_result, 'hand_landmarks') and hand_result.hand_landmarks:
                logger.info("[진단] Hand landmarks 그리기 시작")
                for hand_landmarks in hand_result.hand_landmarks:
                    for landmark in hand_landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        cv2.circle(annotated_frame, (x, y), 2, (0, 0, 255), -1)
                cv2.putText(annotated_frame, "HAND DETECTED", (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Object detection 그리기
            object_result = mediapipe_results.get('object')
            if object_result and hasattr(object_result, 'detections') and object_result.detections:
                logger.info("[진단] Object detections 그리기 시작")
                for detection in object_result.detections:
                    bbox = detection.bounding_box
                    x_min = int(bbox.origin_x * width)
                    y_min = int(bbox.origin_y * height)
                    x_max = int((bbox.origin_x + bbox.width) * width)
                    y_max = int((bbox.origin_y + bbox.height) * height)
                    
                    cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
                    
                    if detection.categories:
                        label = detection.categories[0].category_name
                        confidence = detection.categories[0].score
                        cv2.putText(annotated_frame, f"{label}: {confidence:.2f}", 
                                   (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 전체 상태 표시
            status_y = 30
            if face_result and face_result.face_landmarks:
                cv2.putText(annotated_frame, "Face: OK", (width-150, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, "Face: NO", (width-150, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            status_y += 25
            if pose_result and pose_result.pose_landmarks:
                cv2.putText(annotated_frame, "Pose: OK", (width-150, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, "Pose: NO", (width-150, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            status_y += 25
            if hand_result and hand_result.hand_landmarks:
                cv2.putText(annotated_frame, "Hand: OK", (width-150, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, "Hand: NO", (width-150, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"[진단] app.py: _annotate_frame_with_results 예외: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._create_basic_annotation(frame, mediapipe_results)

    def _create_basic_annotation(self, frame, mediapipe_results):
        """기본 상태 어노테이션 생성"""
        logger.info("[진단] app.py: _create_basic_annotation 진입")
        try:
            annotated_frame = frame.copy()
            height, width = frame.shape[:2]
            
            # 기본 상태 표시
            cv2.putText(annotated_frame, "DMS System Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # MediaPipe 상태 확인 및 표시
            detection_status = []
            
            # 더 정확한 상태 검사
            face_detected = False
            pose_detected = False
            hand_detected = False
            
            if mediapipe_results and isinstance(mediapipe_results, dict):
                face_result = mediapipe_results.get('face')
                if face_result and hasattr(face_result, 'face_landmarks') and face_result.face_landmarks:
                    face_detected = True
                
                pose_result = mediapipe_results.get('pose')
                if pose_result and hasattr(pose_result, 'pose_landmarks') and pose_result.pose_landmarks:
                    pose_detected = True
                
                hand_result = mediapipe_results.get('hand')
                if hand_result and hasattr(hand_result, 'hand_landmarks') and hand_result.hand_landmarks:
                    hand_detected = True
            
            # 상태 표시
            y_offset = 70
            statuses = [
                ("Face", face_detected),
                ("Pose", pose_detected),
                ("Hand", hand_detected)
            ]
            
            for label, detected in statuses:
                status_text = f"{label}: {'OK' if detected else 'NO'}"
                color = (0, 255, 0) if detected else (0, 0, 255)
                cv2.putText(annotated_frame, status_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 30
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"[진단] app.py: _create_basic_annotation 예외: {e}")
            return frame
    
    async def _safe_process_frame(self, mediapipe_results, timestamp):
        logger.info("[진단] app.py: _safe_process_frame 진입")
        logger.info(f"[진단] app.py: _safe_process_frame - mediapipe_results={mediapipe_results}, timestamp={timestamp}")
        try:
            logger.info("[진단] app.py: _safe_process_frame - hasattr(self.integrated_system, 'process_and_annotate_frame') 체크 전")
            if hasattr(self.integrated_system, 'process_and_annotate_frame'):
                logger.info("[진단] app.py: _safe_process_frame - process_and_annotate_frame await 전")
                result = await self.integrated_system.process_and_annotate_frame(
                    mediapipe_results, timestamp
                )
                logger.info(f"[진단] app.py: _safe_process_frame - result={result}")
                return result
            else:
                logger.info("[진단] app.py: _safe_process_frame - 통합 시스템이 사용할 수 없음, 기본 결과 반환")
                return {
                    'fatigue_risk_score': 0.0,
                    'distraction_risk_score': 0.0,
                    'confidence_score': 0.5,
                    'status': 'safe_mode'
                }
        except Exception as e:
            logger.warning(f"[진단] app.py: _safe_process_frame 예외: {e}")
            return None
    
    def _create_safe_fallback_frame(self, original_frame, frame_count):
        logger.info("[진단] app.py: _create_safe_fallback_frame 진입")
        logger.info(f"[진단] app.py: _create_safe_fallback_frame - original_frame type: {type(original_frame)}, frame_count: {frame_count}")
        import numpy as np
        import cv2
        logger.info("[진단] app.py: _create_safe_fallback_frame - numpy, cv2 import 완료")
        height, width = 480, 640
        logger.info(f"[진단] app.py: _create_safe_fallback_frame - height={height}, width={width}")
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        logger.info(f"[진단] app.py: _create_safe_fallback_frame - dummy frame 생성, shape={dummy.shape}")
        msg1 = "[프레임 획득 실패]"
        msg2 = f"프레임 번호: {frame_count}"
        msg3 = "비디오 파일/코덱/환경을 확인하세요."
        logger.info(f"[진단] app.py: _create_safe_fallback_frame - msg1={msg1}, msg2={msg2}, msg3={msg3}")
        cv2.putText(dummy, msg1, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        logger.info("[진단] app.py: _create_safe_fallback_frame - msg1 표시 완료")
        cv2.putText(dummy, msg2, (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        logger.info("[진단] app.py: _create_safe_fallback_frame - msg2 표시 완료")
        cv2.putText(dummy, msg3, (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        logger.info("[진단] app.py: _create_safe_fallback_frame - msg3 표시 완료")
        logger.info("[진단] app.py: _create_safe_fallback_frame - dummy 반환")
        return dummy
    
    def _create_basic_info_overlay(self, frame, frame_count):
        """기본 정보 오버레이가 있는 프레임 생성"""
        logger.info("[진단] app.py: _create_basic_info_overlay 진입")
        try:
            annotated_frame = frame.copy()
            height, width = frame.shape[:2]
            
            # 기본 상태 표시
            cv2.putText(annotated_frame, "DMS System Running", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # MediaPipe 상태 확인
            mp_status = "MediaPipe: "
            if hasattr(self, 'mediapipe_manager') and self.mediapipe_manager:
                results = self.mediapipe_manager.get_latest_results()
                if results:
                    detections = []
                    if results.get('face') and results['face'].face_landmarks:
                        detections.append("Face")
                    if results.get('pose') and results['pose'].pose_landmarks:
                        detections.append("Pose")
                    if results.get('hand') and results['hand'].hand_landmarks:
                        detections.append("Hand")
                    
                    if detections:
                        mp_status += ", ".join(detections)
                        color = (0, 255, 0)  # 녹색
                    else:
                        mp_status += "No Detection"
                        color = (0, 255, 255)  # 노란색
                else:
                    mp_status += "No Results"
                    color = (0, 0, 255)  # 빨간색
            else:
                mp_status += "Not Available"
                color = (128, 128, 128)  # 회색
            
            cv2.putText(annotated_frame, mp_status, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 프레임 카운터
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 사용자 정보
            cv2.putText(annotated_frame, f"User: {self.user_id}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"[진단] app.py: _create_basic_info_overlay 예외: {e}")
            return frame

    def _add_basic_performance_overlay(self, frame, frame_count):
        logger.info("[진단] app.py: _add_basic_performance_overlay 진입")
        try:
            height, width = frame.shape[:2]
            logger.info(f"[진단] app.py: _add_basic_performance_overlay - height={height}, width={width}")
            system_info = [
                f"S-Class DMS v18+ | Frame: {frame_count}",
                f"System: {self.system_type.value}",
                f"User: {self.user_id}",
                f"Resolution: {width}x{height}"
            ]
            logger.info(f"[진단] app.py: _add_basic_performance_overlay - system_info 초기값: {system_info}")
            try:
                if hasattr(self.performance_monitor, 'get_optimization_status'):
                    perf_status = self.performance_monitor.get_optimization_status()
                    logger.info(f"[진단] app.py: _add_basic_performance_overlay - perf_status={perf_status}")
                    if perf_status:
                        system_info.extend([
                            f"FPS: {perf_status.get('avg_fps', 0):.1f}",
                            f"Performance: {perf_status.get('performance_score', 0):.1%}"
                        ])
                        logger.info(f"[진단] app.py: _add_basic_performance_overlay - system_info 확장: {system_info}")
                else:
                    system_info.append("Performance: Monitoring Unavailable")
                    logger.info("[진단] app.py: _add_basic_performance_overlay - Monitoring Unavailable 추가")
            except Exception:
                system_info.append("Performance: Error")
                logger.info("[진단] app.py: _add_basic_performance_overlay - Performance: Error 추가")
            x_offset = width - 300
            y_offset = 30
            logger.info(f"[진단] app.py: _add_basic_performance_overlay - x_offset={x_offset}, y_offset={y_offset}")
            for i, info in enumerate(system_info):
                logger.info(f"[진단] app.py: _add_basic_performance_overlay - info[{i}]={info}")
                cv2.putText(frame, info, (x_offset, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                logger.info(f"[진단] app.py: _add_basic_performance_overlay - info[{i}] 텍스트 표시 완료")
        except Exception as e:
            logger.error(f"[진단] app.py: _add_basic_performance_overlay 예외: {e}")

    def _add_sclass_performance_overlay(self, frame, frame_count):
        logger.info("[진단] app.py: _add_sclass_performance_overlay 진입")
        try:
            um_frame = cv2.UMat(frame)
            logger.info(f"[진단] app.py: _add_sclass_performance_overlay - um_frame 생성, shape={um_frame.get().shape}")
            height, width = frame.shape[:2]
            logger.info(f"[진단] app.py: _add_sclass_performance_overlay - height={height}, width={width}")
            system_info = [
                f"S-Class DMS v18+ RB2 | Frame: {frame_count}",
                f"System: {self.system_type.value} (Hexagon DSP)",
                f"Legacy Mode: {'ON' if self.use_legacy_engine else 'OFF'}",
                f"User: {self.user_id}",
                "Hardware: Qualcomm RB2 Platform"
            ]
            logger.info(f"[진단] app.py: _add_sclass_performance_overlay - system_info 초기값: {system_info}")
            perf_status = self.performance_monitor.get_optimization_status()
            logger.info(f"[진단] app.py: _add_sclass_performance_overlay - perf_status={perf_status}")
            if perf_status:
                system_info.extend([
                    f"FPS: {perf_status.get('avg_fps', 0):.1f}",
                    f"Performance: {perf_status.get('performance_score', 0):.1%}",
                    "GPU Accel: UMat + Adreno 610"
                ])
                logger.info(f"[진단] app.py: _add_sclass_performance_overlay - system_info 확장: {system_info}")
            y_offset = 30
            logger.info(f"[진단] app.py: _add_sclass_performance_overlay - y_offset={y_offset}")
            for i, info in enumerate(system_info):
                logger.info(f"[진단] app.py: _add_sclass_performance_overlay - info[{i}]={info}")
                cv2.putText(um_frame, info, (10, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                logger.info(f"[진단] app.py: _add_sclass_performance_overlay - info[{i}] 텍스트 표시 완료")
            processed_frame = um_frame.get()
            logger.info(f"[진단] app.py: _add_sclass_performance_overlay - processed_frame 생성, shape={processed_frame.shape}")
            frame[:] = processed_frame
            logger.info("[진단] app.py: _add_sclass_performance_overlay - frame에 processed_frame 적용 완료")
        except Exception as e:
            logger.error(f"[진단] app.py: _add_sclass_performance_overlay 예외: {e}")
            self._add_basic_performance_overlay(frame, frame_count)
            logger.info("[진단] app.py: _add_sclass_performance_overlay - 기본 오버레이로 폴백")

    def run(self):
        self.running = True  # [AI 패치] 프레임 생산 루프 활성화
        logger.info("[진단] app.py: run 진입")
        import asyncio
        logger.info("[진단] app.py: run - asyncio import 완료")
        frame_queue = queue.Queue(maxsize=5)
        logger.info(f"[진단] app.py: run - frame_queue 생성, maxsize=5")
        stop_event = threading.Event()
        logger.info("[진단] app.py: run - stop_event 생성")
        def opencv_display_loop():
            logger.info("[진단] app.py: run - opencv_display_loop 진입")
            last_frame = None
            logger.info("[진단] app.py: run - opencv_display_loop - last_frame=None 초기화")
            while not stop_event.is_set():
                try:
                    frame = frame_queue.get(timeout=0.1)
                    logger.info(f"[진단] app.py: run - opencv_display_loop - frame_queue.get 성공, frame type: {type(frame)}")
                    if frame is None:
                        logger.info("[진단] app.py: run - opencv_display_loop - frame is None, break")
                        break
                    last_frame = frame
                    logger.info("[진단] app.py: run - opencv_display_loop - last_frame 갱신")
                except queue.Empty:
                    pass
                if last_frame is not None:
                    cv2.imshow("S-Class DMS v18+ - Research Integrated", last_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("[진단] app.py: run - opencv_display_loop - 'q' 입력, stop_event.set() 및 break")
                    stop_event.set()
                    break
                elif key == ord('s'):
                    logger.info("[진단] app.py: run - opencv_display_loop - 's' 입력, 스크린샷 저장 시도")
                    if last_frame is not None:
                        filename = f"screenshot_{int(time.time())}.png"
                        cv2.imwrite(filename, last_frame)
                        logger.info(f"[진단] app.py: run - opencv_display_loop - 스크린샷 저장 완료: {filename}")
            cv2.destroyAllWindows()
            logger.info("[진단] app.py: run - opencv_display_loop - OpenCV 창 종료")
        async def async_frame_producer():
            """비동기 프레임 생산자 - 유령 랜드마크 방지를 위한 동기화 개선"""
            logger.info("[진단] app.py: run - async_frame_producer 진입")
            await self.initialize()
            logger.info("[진단] app.py: run - S-Class DMS 시스템 초기화 완료")
            logger.info("[진단] app.py: run - S-Class 기능/어텐션/인지 부하/적응형 파이프라인 활성화")
            logger.info("[진단] app.py: run - S-Class DMS v18+ 시스템 시작 안내")
            await asyncio.sleep(0.1)
            logger.info("[진단] app.py: run - 초기 대기 완료")
            frame_count = 0
            logger.info(f"[진단] app.py: run - frame_count 초기화: {frame_count}")
            while not stop_event.is_set():
                frame = self.video_input_manager.get_frame()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                    
                frame_count += 1
                
                # MediaPipe 작업 실행 및 동기화된 결과 처리
                annotated_frame = None
                if hasattr(self, 'mediapipe_manager') and self.mediapipe_manager:
                    try:
                        # MediaPipe 작업 실행
                        self.mediapipe_manager.run_tasks(frame.copy())
                        
                        # 짧은 대기 후 결과 가져오기 (동기화 개선)
                        await asyncio.sleep(0.005)  # 5ms 대기로 콜백 처리 시간 확보
                        
                        # 최신 결과 가져오기
                        mediapipe_results = self.mediapipe_manager.get_latest_results()
                        
                        # 결과 로깅 (10프레임마다)
                        if frame_count % 10 == 0 and mediapipe_results:
                            face_status = "YES" if mediapipe_results.get('face') and mediapipe_results['face'].face_landmarks else "NO"
                            pose_status = "YES" if mediapipe_results.get('pose') and mediapipe_results['pose'].pose_landmarks else "NO"
                            hand_status = "YES" if mediapipe_results.get('hand') and mediapipe_results['hand'].hand_landmarks else "NO"
                            logger.info(f"Frame {frame_count}: Face={face_status}, Pose={pose_status}, Hand={hand_status}")
                        
                        # 현재 프레임과 동기화된 어노테이션
                        if mediapipe_results and any(mediapipe_results.values()):
                            annotated_frame = self._annotate_frame_with_results(frame, mediapipe_results)
                        else:
                            annotated_frame = self._create_basic_info_overlay(frame, frame_count)
                            
                    except Exception as e:
                        logger.error(f"MediaPipe 작업 실행 오류: {e}")
                        annotated_frame = self._create_basic_info_overlay(frame, frame_count)
                else:
                    annotated_frame = self._create_basic_info_overlay(frame, frame_count)
                
                # 프레임 큐에 추가 (유령 랜드마크 방지를 위한 동기화)
                if annotated_frame is not None:
                    try:
                        frame_queue.put_nowait(annotated_frame)
                    except queue.Full:
                        # 큐가 가득 차면 가장 오래된 프레임 제거 후 새 프레임 추가
                        try:
                            frame_queue.get_nowait()  # 오래된 프레임 제거
                            frame_queue.put_nowait(annotated_frame)  # 새 프레임 추가
                        except queue.Empty:
                            pass
                    
                # 프레임 처리 안정성을 위한 적절한 대기 시간
                await asyncio.sleep(0.010)  # 10ms 대기로 MediaPipe 콜백과 동기화
            try:
                frame_queue.put(None, timeout=0.1)
                logger.info("[진단] app.py: run - async_frame_producer - 종료 시 frame_queue에 None 삽입")
            except queue.Full:
                logger.info("[진단] app.py: run - async_frame_producer - 종료 시 frame_queue 가득참 (queue.Full)")
                pass
        display_thread = threading.Thread(target=opencv_display_loop)
        logger.info("[진단] app.py: run - display_thread 생성")
        display_thread.start()
        logger.info("[진단] app.py: run - display_thread 시작")
        asyncio.run(async_frame_producer())
        logger.info("[진단] app.py: run - asyncio.run(async_frame_producer()) 완료")
        stop_event.set()
        logger.info("[진단] app.py: run - stop_event.set() 호출")
        display_thread.join()
        logger.info("[진단] app.py: run - display_thread.join() 완료")

    def _handle_keyboard_input(self, key: int) -> bool:
        logger.info(f"[진단] app.py: _handle_keyboard_input 진입, key={key}")
        if key == ord("q") or key == 27:
            logger.info("[진단] app.py: _handle_keyboard_input - 종료 키(q/ESC) 입력, False 반환")
            return False
        elif key == ord(" "):
            self.paused = not self.paused
            logger.info(f"[진단] app.py: _handle_keyboard_input - 스페이스바 입력, self.paused={self.paused}")
        elif key == ord("s") and self.current_processed_frame is not None:
            filename = f"captures/sclass_dms_v18_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            Path("captures").mkdir(exist_ok=True)
            cv2.imwrite(filename, self.current_processed_frame)
            logger.info(f"[진단] app.py: _handle_keyboard_input - 's' 입력, 프레임 저장: {filename}")
        elif key == ord("r"):
            self.performance_monitor = PerformanceOptimizer()
            logger.info("[진단] app.py: _handle_keyboard_input - 'r' 입력, 성능 통계 리셋")
        elif key == ord("i"):
            try:
                if hasattr(self.integrated_system, 'get_system_status'):
                    status = self.integrated_system.get_system_status()
                    logger.info(f"[진단] app.py: _handle_keyboard_input - 'i' 입력, 시스템 상태: {status}")
                else:
                    logger.info("[진단] app.py: _handle_keyboard_input - 'i' 입력, 시스템 상태 정보 없음")
            except Exception as e:
                logger.error(f"[진단] app.py: _handle_keyboard_input - 시스템 정보 출력 실패: {e}")
        elif key == ord("t"):
            self.use_legacy_engine = not self.use_legacy_engine
            logger.info(f"[진단] app.py: _handle_keyboard_input - 't' 입력, 시스템 모드 전환: {'Legacy' if self.use_legacy_engine else 'S-Class'}")
        elif key == ord("d"):
            try:
                status = self.dynamic_analysis.get_analysis_status()
                logger.info(f"[진단] app.py: _handle_keyboard_input - 'd' 입력, 동적 분석 상태: {status}")
            except Exception as e:
                logger.error(f"[진단] app.py: _handle_keyboard_input - 동적 분석 정보 출력 실패: {e}")
        logger.info("[진단] app.py: _handle_keyboard_input - True 반환")
        return True

    async def _cleanup_async(self):
        logger.info("[진단] app.py: _cleanup_async 진입")
        try:
            if hasattr(self, 'integrated_system') and hasattr(self.integrated_system, 'shutdown'):
                logger.info("[진단] app.py: _cleanup_async - integrated_system.shutdown() await 전")
                coro = self.integrated_system.shutdown()
                logger.info(f"[진단] app.py: _cleanup_async - integrated_system.shutdown() 반환: {coro}")
                if coro is not None:
                    await coro
                    logger.info("[진단] app.py: _cleanup_async - integrated_system.shutdown() await 완료")
            if hasattr(self, 'mediapipe_manager') and hasattr(self.mediapipe_manager, 'shutdown'):
                logger.info("[진단] app.py: _cleanup_async - mediapipe_manager.shutdown() await 전")
                coro = self.mediapipe_manager.shutdown()
                logger.info(f"[진단] app.py: _cleanup_async - mediapipe_manager.shutdown() 반환: {coro}")
                if coro is not None:
                    await coro
                    logger.info("[진단] app.py: _cleanup_async - mediapipe_manager.shutdown() await 완료")
            if hasattr(self, 'video_input_manager') and hasattr(self.video_input_manager, 'shutdown'):
                logger.info("[진단] app.py: _cleanup_async - video_input_manager.shutdown() await 전")
                coro = self.video_input_manager.shutdown()
                logger.info(f"[진단] app.py: _cleanup_async - video_input_manager.shutdown() 반환: {coro}")
                if coro is not None:
                    await coro
                    logger.info("[진단] app.py: _cleanup_async - video_input_manager.shutdown() await 완료")
        except Exception as e:
            logger.error(f"[진단] app.py: _cleanup_async 예외: {e}")
