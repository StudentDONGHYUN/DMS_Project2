"""
Integrated DMS System
통합된 운전자 모니터링 시스템
기존 코드와 개선된 코드를 통합하여 호환성과 성능을 모두 확보
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

# Core imports
from core.definitions import CameraPosition
from core.state_manager import StateManager
from config.settings import get_config, FeatureFlagConfig
from models.data_structures import UIState

# System imports
from systems.mediapipe_manager_v2 import MediaPipeManager
from systems.performance_v2 import PerformanceOptimizer
from systems.personalization_v2 import PersonalizationEngine
from systems.dynamic_v2 import DynamicAnalysisEngine
from systems.backup_v2 import SensorBackupManager

# IO Handler imports
from io_handler.video_input_v2 import VideoInputManager
from io_handler.ui import UIHandler

# Integration imports
from integration.integrated_system import IntegratedDMSSystem, AnalysisSystemType

# Innovation Systems (v19 features)
from systems.ai_driving_coach import AIDrivingCoach
from systems.v2d_healthcare import V2DHealthcareSystem
from systems.ar_hud_system import ARHUDSystem
from systems.emotional_care_system import EmotionalCareSystem
from systems.digital_twin_platform import DigitalTwinPlatform

# Utils
from utils.drawing import draw_face_landmarks_on_image, draw_pose_landmarks_on_image, draw_hand_landmarks_on_image
from utils.memory_monitor_v2 import MemoryMonitor

logger = logging.getLogger(__name__)


@dataclass
class SystemStatus:
    """시스템 상태"""
    ai_coach_active: bool = False
    healthcare_active: bool = False
    ar_hud_active: bool = False
    emotional_care_active: bool = False
    digital_twin_active: bool = False
    
    current_sessions: Dict[str, str] = None
    last_update: float = 0.0
    
    def __post_init__(self):
        if self.current_sessions is None:
            self.current_sessions = {}


class DMSSystem:
    """통합 DMS 시스템 - 기존 코드와 개선된 코드의 통합"""

    def __init__(
        self,
        user_id: str = "default",
        camera_position: CameraPosition = CameraPosition.REARVIEW_MIRROR,
        enable_calibration: bool = True,
        system_edition: str = "RESEARCH",
        feature_flags: Optional[FeatureFlagConfig] = None,
        source_type: str = "webcam",
        webcam_id: int = 0,
        video_files: List[str] = None,
        is_same_driver: bool = True
    ):
        """
        DMS 시스템 초기화
        
        Args:
            user_id: 사용자 ID
            camera_position: 카메라 위치
            enable_calibration: 캘리브레이션 활성화 여부
            system_edition: 시스템 에디션
            feature_flags: 기능 플래그 설정
            source_type: 입력 소스 타입
            webcam_id: 웹캠 ID
            video_files: 비디오 파일 목록
            is_same_driver: 동일 운전자 세션 여부
        """
        # Configuration
        self.config = get_config()
        self.user_id = user_id
        self.camera_position = camera_position
        self.enable_calibration = enable_calibration
        self.system_edition = system_edition
        self.feature_flags = feature_flags or FeatureFlagConfig(edition=system_edition)
        self.source_type = source_type
        self.webcam_id = webcam_id
        self.video_files = video_files or []
        self.is_same_driver = is_same_driver
        
        # System state
        self.is_running = False
        self.is_initialized = False
        
        # Core components
        self.state_manager = StateManager()
        self.mediapipe_manager = None
        self.performance_optimizer = None
        self.personalization_engine = None
        self.dynamic_analyzer = None
        self.sensor_backup = None
        
        # Integration system
        self.integrated_system = None
        
        # Innovation systems
        self.innovation_systems = {}
        self.system_status = SystemStatus()
        
        # IO handlers
        self.video_input = None
        self.ui_handler = None
        
        # Performance monitoring
        self.memory_monitor = MemoryMonitor()
        self.frame_count = 0
        self.start_time = time.time()
        
        # Data storage
        self.session_data = []
        self.performance_metrics = {}
        
        logger.info(f"DMS System initialized - User: {user_id}, Edition: {system_edition}")

    async def initialize(self) -> bool:
        """시스템 초기화"""
        try:
            logger.info("Initializing DMS System...")
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Initialize integration system
            await self._initialize_integration_system()
            
            # Initialize innovation systems
            await self._initialize_innovation_systems()
            
            # Initialize IO handlers
            await self._initialize_io_handlers()
            
            self.is_initialized = True
            logger.info("DMS System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False

    async def _initialize_core_components(self):
        """핵심 컴포넌트 초기화"""
        logger.info("Initializing core components...")
        
        # MediaPipe Manager
        self.mediapipe_manager = MediaPipeManager(
            enable_face=True,
            enable_pose=True,
            enable_hand=True,
            enable_object=True
        )
        
        # Performance Optimizer
        self.performance_optimizer = PerformanceOptimizer()
        
        # Personalization Engine
        self.personalization_engine = PersonalizationEngine(self.user_id)
        
        # Dynamic Analysis Engine
        self.dynamic_analyzer = DynamicAnalysisEngine()
        
        # Sensor Backup Manager
        self.sensor_backup = SensorBackupManager()
        
        logger.info("Core components initialized")

    async def _initialize_integration_system(self):
        """통합 시스템 초기화"""
        logger.info("Initializing integration system...")
        
        # Determine system type based on edition
        system_type = AnalysisSystemType.STANDARD
        if self.system_edition in ["ENTERPRISE", "RESEARCH"]:
            system_type = AnalysisSystemType.HIGH_PERFORMANCE
        
        # Create integrated system
        self.integrated_system = IntegratedDMSSystem(
            system_type=system_type,
            use_legacy_engine=False  # Use modern system by default
        )
        
        await self.integrated_system.initialize()
        logger.info("Integration system initialized")

    async def _initialize_innovation_systems(self):
        """혁신 시스템 초기화"""
        logger.info("Initializing innovation systems...")
        
        # AI Driving Coach (PRO 이상)
        if self.feature_flags.s_class_advanced_features:
            try:
                self.innovation_systems["ai_coach"] = AIDrivingCoach(self.user_id)
                logger.info("✅ AI Driving Coach initialized")
            except Exception as e:
                logger.error(f"❌ AI Driving Coach initialization failed: {e}")
        
        # V2D Healthcare (PRO 이상)
        if self.feature_flags.s_class_advanced_features:
            try:
                self.innovation_systems["healthcare"] = V2DHealthcareSystem(self.user_id)
                logger.info("✅ V2D Healthcare initialized")
            except Exception as e:
                logger.error(f"❌ V2D Healthcare initialization failed: {e}")
        
        # AR HUD System (ENTERPRISE 이상)
        if self.feature_flags.neural_ai_features:
            try:
                self.innovation_systems["ar_hud"] = ARHUDSystem()
                logger.info("✅ AR HUD System initialized")
            except Exception as e:
                logger.error(f"❌ AR HUD System initialization failed: {e}")
        
        # Emotional Care System (ENTERPRISE 이상)
        if self.feature_flags.neural_ai_features:
            try:
                self.innovation_systems["emotional_care"] = EmotionalCareSystem(self.user_id)
                logger.info("✅ Emotional Care System initialized")
            except Exception as e:
                logger.error(f"❌ Emotional Care System initialization failed: {e}")
        
        # Digital Twin Platform (RESEARCH 에디션)
        if self.feature_flags.innovation_research_features:
            try:
                self.innovation_systems["digital_twin"] = DigitalTwinPlatform()
                logger.info("✅ Digital Twin Platform initialized")
            except Exception as e:
                logger.error(f"❌ Digital Twin Platform initialization failed: {e}")
        
        logger.info(f"Innovation systems initialized: {len(self.innovation_systems)} systems")

    async def _initialize_io_handlers(self):
        """IO 핸들러 초기화"""
        logger.info("Initializing IO handlers...")
        
        # Video Input Manager
        self.video_input = VideoInputManager(
            source_type=self.source_type,
            webcam_id=self.webcam_id,
            video_files=self.video_files,
            enable_calibration=self.enable_calibration
        )
        
        # UI Handler
        self.ui_handler = UIHandler()
        
        logger.info("IO handlers initialized")

    async def start(self) -> bool:
        """시스템 시작"""
        try:
            if not self.is_initialized:
                logger.error("System not initialized. Call initialize() first.")
                return False
            
            logger.info("Starting DMS System...")
            
            # Start video input
            await self.video_input.start()
            
            # Start UI handler
            await self.ui_handler.start()
            
            # Start innovation system sessions
            await self._start_innovation_sessions()
            
            # Start main processing loop
            self.is_running = True
            await self._run_main_loop()
            
            return True
            
        except Exception as e:
            logger.error(f"System startup failed: {e}")
            return False

    async def _start_innovation_sessions(self):
        """혁신 시스템 세션 시작"""
        logger.info("Starting innovation system sessions...")
        
        # AI Driving Coach session
        if "ai_coach" in self.innovation_systems:
            try:
                session_id = await self.innovation_systems["ai_coach"].start_driving_session()
                self.system_status.current_sessions["ai_coach"] = session_id
                self.system_status.ai_coach_active = True
                logger.info(f"🎓 AI Coach session started: {session_id}")
            except Exception as e:
                logger.error(f"AI Coach session start failed: {e}")
        
        # Healthcare session
        if "healthcare" in self.innovation_systems:
            try:
                session_id = await self.innovation_systems["healthcare"].start_health_monitoring()
                self.system_status.current_sessions["healthcare"] = session_id
                self.system_status.healthcare_active = True
                logger.info(f"🏥 Healthcare session started: {session_id}")
            except Exception as e:
                logger.error(f"Healthcare session start failed: {e}")
        
        # Emotional Care session
        if "emotional_care" in self.innovation_systems:
            try:
                session_id = await self.innovation_systems["emotional_care"].start_emotional_monitoring()
                self.system_status.current_sessions["emotional_care"] = session_id
                self.system_status.emotional_care_active = True
                logger.info(f"💙 Emotional Care session started: {session_id}")
            except Exception as e:
                logger.error(f"Emotional Care session start failed: {e}")

    async def _run_main_loop(self):
        """메인 처리 루프"""
        logger.info("Starting main processing loop...")
        
        try:
            while self.is_running:
                # Get frame from video input
                frame = await self.video_input.get_frame()
                if frame is None:
                    continue
                
                # Process frame with integrated system
                timestamp = time.time()
                ui_state = await self._process_frame(frame, timestamp)
                
                # Update UI
                await self.ui_handler.update(ui_state)
                
                # Process with innovation systems
                await self._process_with_innovation_systems(ui_state)
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Check for exit condition
                if await self._should_exit():
                    break
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Main loop error: {e}")
        finally:
            await self.stop()

    async def _process_frame(self, frame: np.ndarray, timestamp: float) -> UIState:
        """프레임 처리"""
        try:
            # Process with MediaPipe
            mediapipe_results = await self.mediapipe_manager.process_frame(frame)
            
            # Process with integrated system
            integrated_results = await self.integrated_system.process_frame(
                mediapipe_results, timestamp
            )
            
            # Create UI state
            ui_state = self._create_ui_state(integrated_results, timestamp)
            
            # Update frame count
            self.frame_count += 1
            
            return ui_state
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return self._create_fallback_ui_state(timestamp)

    def _create_ui_state(self, integrated_results: Dict[str, Any], timestamp: float) -> UIState:
        """UI 상태 생성"""
        ui_state = UIState(
            risk_score=integrated_results.get('fatigue_risk_score', 0.0),
            overall_safety_status=self._get_safety_status(integrated_results),
            timestamp=timestamp,
            user_id=self.user_id
        )
        
        # Update biometrics
        if 'biometrics' in integrated_results:
            ui_state.biometrics.heart_rate = integrated_results['biometrics'].get('heart_rate')
            ui_state.biometrics.stress_level = integrated_results['biometrics'].get('stress_level')
        
        # Update gaze data
        if 'gaze' in integrated_results:
            ui_state.gaze.attention_score = integrated_results['gaze'].get('attention_score', 0.0)
            ui_state.gaze.distraction_level = integrated_results['gaze'].get('distraction_level', 0.0)
        
        # Update face data
        if 'face' in integrated_results:
            ui_state.face.emotion_state = integrated_results['face'].get('emotion_state', 'NEUTRAL')
            ui_state.face.emotion_confidence = integrated_results['face'].get('emotion_confidence', 0.0)
        
        # Update system health
        ui_state.system_health.processing_fps = self._calculate_fps()
        ui_state.system_health.memory_usage_mb = self.memory_monitor.get_memory_usage()
        
        return ui_state

    def _create_fallback_ui_state(self, timestamp: float) -> UIState:
        """폴백 UI 상태 생성"""
        return UIState(
            risk_score=0.0,
            overall_safety_status="safe",
            timestamp=timestamp,
            user_id=self.user_id
        )

    def _get_safety_status(self, results: Dict[str, Any]) -> str:
        """안전 상태 결정"""
        risk_score = results.get('fatigue_risk_score', 0.0)
        
        if risk_score < 0.3:
            return "safe"
        elif risk_score < 0.6:
            return "warning"
        elif risk_score < 0.8:
            return "danger"
        else:
            return "critical"

    async def _process_with_innovation_systems(self, ui_state: UIState):
        """혁신 시스템과 함께 처리"""
        try:
            # AI Driving Coach
            if "ai_coach" in self.innovation_systems and self.system_status.ai_coach_active:
                result = await self.innovation_systems["ai_coach"].process_driving_data(ui_state)
                await self._handle_system_result("ai_coach", result)
            
            # Healthcare
            if "healthcare" in self.innovation_systems and self.system_status.healthcare_active:
                result = await self.innovation_systems["healthcare"].process_health_data(ui_state)
                await self._handle_system_result("healthcare", result)
            
            # Emotional Care
            if "emotional_care" in self.innovation_systems and self.system_status.emotional_care_active:
                result = await self.innovation_systems["emotional_care"].process_emotional_data(ui_state)
                await self._handle_system_result("emotional_care", result)
            
            # AR HUD
            if "ar_hud" in self.innovation_systems:
                result = await self.innovation_systems["ar_hud"].process_vehicle_context(ui_state)
                await self._handle_system_result("ar_hud", result)
            
        except Exception as e:
            logger.error(f"Innovation systems processing error: {e}")

    async def _handle_system_result(self, system_name: str, result: Any):
        """시스템 결과 처리"""
        try:
            if result and hasattr(result, 'recommendations'):
                for recommendation in result.recommendations:
                    logger.info(f"{system_name}: {recommendation}")
            
            # Store result for analysis
            self.session_data.append({
                'timestamp': time.time(),
                'system': system_name,
                'result': result
            })
            
        except Exception as e:
            logger.error(f"Error handling {system_name} result: {e}")

    async def _update_performance_metrics(self):
        """성능 메트릭 업데이트"""
        try:
            current_time = time.time()
            
            # Update performance metrics
            self.performance_metrics = {
                'fps': self._calculate_fps(),
                'memory_usage_mb': self.memory_monitor.get_memory_usage(),
                'cpu_usage_percent': self.memory_monitor.get_cpu_usage(),
                'frame_count': self.frame_count,
                'uptime_seconds': current_time - self.start_time,
                'active_systems': len(self.innovation_systems)
            }
            
            # Log performance periodically
            if self.frame_count % 300 == 0:  # Every 300 frames
                logger.info(f"Performance: {self.performance_metrics}")
            
        except Exception as e:
            logger.error(f"Performance metrics update error: {e}")

    def _calculate_fps(self) -> float:
        """FPS 계산"""
        if self.frame_count == 0:
            return 0.0
        
        elapsed_time = time.time() - self.start_time
        return self.frame_count / elapsed_time if elapsed_time > 0 else 0.0

    async def _should_exit(self) -> bool:
        """종료 조건 확인"""
        # Check if video input has ended
        if self.video_input and not self.video_input.is_active():
            return True
        
        # Check for keyboard interrupt (would need to be implemented)
        return False

    async def stop(self):
        """시스템 중지"""
        try:
            logger.info("Stopping DMS System...")
            
            self.is_running = False
            
            # Stop innovation sessions
            await self._stop_innovation_sessions()
            
            # Stop IO handlers
            if self.video_input:
                await self.video_input.stop()
            
            if self.ui_handler:
                await self.ui_handler.stop()
            
            # Save session data
            await self._save_session_data()
            
            logger.info("DMS System stopped successfully")
            
        except Exception as e:
            logger.error(f"System shutdown error: {e}")

    async def _stop_innovation_sessions(self):
        """혁신 시스템 세션 중지"""
        logger.info("Stopping innovation system sessions...")
        
        for system_name, system in self.innovation_systems.items():
            try:
                if hasattr(system, 'stop_session'):
                    await system.stop_session()
                logger.info(f"✅ {system_name} session stopped")
            except Exception as e:
                logger.error(f"❌ {system_name} session stop failed: {e}")

    async def _save_session_data(self):
        """세션 데이터 저장"""
        try:
            if not self.session_data:
                return
            
            # Create session file
            session_file = Path(f"logs/session_{self.user_id}_{int(time.time())}.json")
            session_file.parent.mkdir(exist_ok=True)
            
            # Save data (simplified for now)
            import json
            with open(session_file, 'w') as f:
                json.dump({
                    'user_id': self.user_id,
                    'session_duration': time.time() - self.start_time,
                    'frame_count': self.frame_count,
                    'performance_metrics': self.performance_metrics,
                    'data_count': len(self.session_data)
                }, f, indent=2)
            
            logger.info(f"Session data saved to {session_file}")
            
        except Exception as e:
            logger.error(f"Session data save error: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        return {
            'is_running': self.is_running,
            'is_initialized': self.is_initialized,
            'user_id': self.user_id,
            'system_edition': self.system_edition,
            'frame_count': self.frame_count,
            'uptime_seconds': time.time() - self.start_time,
            'performance_metrics': self.performance_metrics,
            'innovation_systems': {
                name: {'active': getattr(self.system_status, f'{name.replace("_", "_")}_active', False)}
                for name in self.innovation_systems.keys()
            }
        }

    async def create_digital_twin_from_session(self) -> Optional[str]:
        """세션에서 디지털 트윈 생성"""
        if "digital_twin" not in self.innovation_systems:
            logger.warning("Digital Twin Platform not available")
            return None
        
        try:
            twin_id = await self.innovation_systems["digital_twin"].create_twin_from_session(
                session_data=self.session_data,
                user_id=self.user_id
            )
            logger.info(f"Digital twin created: {twin_id}")
            return twin_id
        except Exception as e:
            logger.error(f"Digital twin creation failed: {e}")
            return None