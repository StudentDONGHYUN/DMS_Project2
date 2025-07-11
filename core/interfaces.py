"""
시스템 인터페이스 정의
의존성 역전 원칙을 적용하여 순환 임포트를 방지하고 모듈간 느슨한 결합을 구현합니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from core.definitions import AdvancedMetrics, DriverState, AnalysisEvent, EmotionState


class IMetricsUpdater(ABC):
    """메트릭 업데이트 인터페이스"""
    
    @abstractmethod
    def update_drowsiness_metrics(self, drowsiness_data: Dict[str, Any]) -> None:
        """졸음 관련 메트릭 업데이트"""
        pass
    
    @abstractmethod
    def update_emotion_metrics(self, emotion_data: Dict[str, Any]) -> None:
        """감정 관련 메트릭 업데이트"""
        pass
    
    @abstractmethod
    def update_gaze_metrics(self, gaze_data: Dict[str, Any]) -> None:
        """시선 관련 메트릭 업데이트"""
        pass
    
    @abstractmethod
    def update_distraction_metrics(self, distraction_data: Dict[str, Any]) -> None:
        """주의산만 관련 메트릭 업데이트"""
        pass
    
    @abstractmethod
    def update_prediction_metrics(self, prediction_data: Dict[str, Any]) -> None:
        """예측 관련 메트릭 업데이트"""
        pass
    
    @abstractmethod
    def get_current_metrics(self) -> AdvancedMetrics:
        """현재 메트릭 반환"""
        pass


class IStateManager(ABC):
    """상태 관리 인터페이스"""
    
    @abstractmethod
    def handle_event(self, event: AnalysisEvent) -> None:
        """이벤트 처리"""
        pass
    
    @abstractmethod
    def get_current_state(self) -> DriverState:
        """현재 상태 반환"""
        pass
    
    @abstractmethod
    def get_state_duration(self) -> float:
        """현재 상태 지속 시간 반환"""
        pass


class IEventPublisher(ABC):
    """이벤트 발행 인터페이스"""
    
    @abstractmethod
    def publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """이벤트 발행"""
        pass
    
    @abstractmethod
    def subscribe_to_event(self, event_type: str, handler: callable) -> None:
        """이벤트 구독"""
        pass


class IDataProcessor(ABC):
    """데이터 처리 기본 인터페이스"""
    
    @abstractmethod
    async def process_data(self, data: Any, timestamp: float) -> Dict[str, Any]:
        """데이터 처리 메인 메서드"""
        pass
    
    @abstractmethod
    def get_processor_name(self) -> str:
        """프로세서 이름 반환"""
        pass
    
    @abstractmethod
    def get_required_data_types(self) -> List[str]:
        """필요한 데이터 타입 목록 반환"""
        pass


class IFaceDataProcessor(IDataProcessor):
    """얼굴 데이터 처리 인터페이스"""
    
    @abstractmethod
    async def process_face_landmarks(self, landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """얼굴 랜드마크 처리"""
        pass
    
    @abstractmethod
    async def process_face_blendshapes(self, blendshapes: Any, timestamp: float) -> Dict[str, Any]:
        """얼굴 블렌드셰이프 처리"""
        pass
    
    @abstractmethod
    async def process_facial_transformation(self, transformation: Any, timestamp: float) -> Dict[str, Any]:
        """얼굴 변환 행렬 처리"""
        pass


class IPoseDataProcessor(IDataProcessor):
    """자세 데이터 처리 인터페이스"""
    
    @abstractmethod
    async def process_pose_landmarks(self, landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """자세 랜드마크 처리"""
        pass
    
    @abstractmethod
    async def process_world_landmarks(self, world_landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """3D 월드 랜드마크 처리"""
        pass


class IHandDataProcessor(IDataProcessor):
    """손 데이터 처리 인터페이스"""
    
    @abstractmethod
    async def process_hand_landmarks(self, hand_results: Any, timestamp: float) -> List[Dict[str, Any]]:
        """손 랜드마크 처리"""
        pass


class IObjectDataProcessor(IDataProcessor):
    """객체 데이터 처리 인터페이스"""
    
    @abstractmethod
    async def process_object_detections(self, detections: Any, hand_positions: List[Dict], timestamp: float) -> Dict[str, Any]:
        """객체 감지 결과 처리"""
        pass


class IDrowsinessDetector(ABC):
    """졸음 감지 인터페이스"""
    
    @abstractmethod
    def detect_drowsiness(self, face_landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """졸음 감지 메인 메서드"""
        pass
    
    @abstractmethod
    def update_personalized_threshold(self, ear_values: List[float]) -> float:
        """개인화 임계값 업데이트"""
        pass
    
    @abstractmethod
    def get_current_threshold(self) -> float:
        """현재 임계값 반환"""
        pass


class IEmotionRecognizer(ABC):
    """감정 인식 인터페이스"""
    
    @abstractmethod
    def analyze_emotion(self, face_blendshapes: Any, timestamp: float) -> Dict[str, Any]:
        """감정 분석 메인 메서드"""
        pass
    
    @abstractmethod
    def get_emotion_history(self, duration: float) -> List[Dict[str, Any]]:
        """감정 이력 반환"""
        pass


class IGazeClassifier(ABC):
    """시선 분류 인터페이스"""
    
    @abstractmethod
    def classify(self, yaw: float, pitch: float, timestamp: Optional[float] = None) -> Any:
        """시선 구역 분류"""
        pass
    
    @abstractmethod
    def get_gaze_stability(self) -> float:
        """시선 안정성 점수 반환"""
        pass
    
    @abstractmethod
    def get_attention_focus_score(self) -> float:
        """주의집중 점수 반환"""
        pass


class IDistractionDetector(ABC):
    """주의산만 감지 인터페이스"""
    
    @abstractmethod
    def analyze_detections(self, object_results: Any, hand_positions: List[Dict], timestamp: float) -> Dict[str, Any]:
        """주의산만 객체 분석"""
        pass
    
    @abstractmethod
    def get_persistent_risk_score(self) -> float:
        """지속적 위험 점수 반환"""
        pass


class IDriverIdentifier(ABC):
    """운전자 식별 인터페이스"""
    
    @abstractmethod
    def identify_driver(self, face_landmarks: Any) -> Dict[str, Any]:
        """운전자 식별"""
        pass
    
    @abstractmethod
    def register_new_driver(self, driver_id: str, features: np.ndarray) -> bool:
        """새 운전자 등록"""
        pass
    
    @abstractmethod
    def get_current_driver(self) -> Dict[str, str]:
        """현재 운전자 정보 반환"""
        pass


class IPredictiveSafety(ABC):
    """예측적 안전 시스템 인터페이스"""
    
    @abstractmethod
    def predict_risk(self, current_metrics: AdvancedMetrics, timestamp: float) -> Dict[str, Any]:
        """위험 예측"""
        pass
    
    @abstractmethod
    def get_risk_factors(self, metrics: AdvancedMetrics) -> List[str]:
        """위험 요소 식별"""
        pass


class IMultiModalAnalyzer(ABC):
    """멀티모달 분석 인터페이스"""
    
    @abstractmethod
    def fuse_drowsiness_signals(self, face_data: Dict, pose_data: Dict, emotion_data: Dict) -> float:
        """졸음 신호 융합"""
        pass
    
    @abstractmethod
    def fuse_distraction_signals(self, face_data: Dict, hand_data: Dict, object_data: Dict, emotion_data: Dict) -> float:
        """주의산만 신호 융합"""
        pass


class IPerformanceOptimizer(ABC):
    """성능 최적화 인터페이스"""
    
    @abstractmethod
    def log_performance(self, processing_time: float, fps: float) -> None:
        """성능 로깅"""
        pass
    
    @abstractmethod
    def get_optimization_status(self) -> Dict[str, Any]:
        """최적화 상태 반환"""
        pass
    
    @abstractmethod
    def is_optimization_active(self) -> bool:
        """최적화 활성화 상태 확인"""
        pass


class IBackupManager(ABC):
    """백업 시스템 인터페이스"""
    
    @abstractmethod
    def activate_backup(self, backup_type: str) -> None:
        """백업 모드 활성화"""
        pass
    
    @abstractmethod
    def deactivate_backup(self, backup_type: str) -> None:
        """백업 모드 비활성화"""
        pass
    
    @abstractmethod
    def get_backup_status(self) -> Dict[str, Any]:
        """백업 상태 반환"""
        pass


class IUIRenderer(ABC):
    """UI 렌더링 인터페이스"""
    
    @abstractmethod
    def draw_enhanced_results(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """향상된 결과 그리기"""
        pass
    
    @abstractmethod
    def draw_status_info(self, frame: np.ndarray, metrics: AdvancedMetrics, state: DriverState) -> np.ndarray:
        """상태 정보 그리기"""
        pass


class IMediaPipeManager(ABC):
    """MediaPipe 관리 인터페이스"""
    
    @abstractmethod
    def run_tasks(self, frame: np.ndarray) -> None:
        """MediaPipe 태스크 실행"""
        pass
    
    @abstractmethod
    def get_system_health(self) -> Dict[str, Any]:
        """시스템 건강 상태 반환"""
        pass
    
    @abstractmethod
    def update_active_tasks(self, required_tasks: List[str]) -> None:
        """활성 태스크 업데이트"""
        pass


class IVideoInputManager(ABC):
    """비디오 입력 관리 인터페이스"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """초기화"""
        pass
    
    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """프레임 획득"""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """실행 상태 확인"""
        pass
    
    @abstractmethod
    def release(self) -> None:
        """리소스 해제"""
        pass


class IConfigurable(ABC):
    """설정 가능한 컴포넌트 인터페이스"""
    
    @abstractmethod
    def update_config(self, config: Dict[str, Any]) -> None:
        """설정 업데이트"""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """현재 설정 반환"""
        pass
    
    @abstractmethod
    def reset_to_defaults(self) -> None:
        """기본 설정으로 리셋"""
        pass


class IHealthCheckable(ABC):
    """건강 상태 확인 가능한 컴포넌트 인터페이스"""
    
    @abstractmethod
    def check_health(self) -> Dict[str, Any]:
        """건강 상태 확인"""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """건강한 상태인지 확인"""
        pass


class ILoggable(ABC):
    """로깅 가능한 컴포넌트 인터페이스"""
    
    @abstractmethod
    def get_logger_name(self) -> str:
        """로거 이름 반환"""
        pass
    
    @abstractmethod
    def log_status(self) -> None:
        """상태 로깅"""
        pass


# 복합 인터페이스들 (여러 인터페이스를 조합)
class ICompleteAnalysisComponent(IConfigurable, IHealthCheckable, ILoggable):
    """완전한 분석 컴포넌트 인터페이스"""
    pass


class IAnalysisEngine(IMetricsUpdater, IEventPublisher):
    """분석 엔진 통합 인터페이스"""
    
    @abstractmethod
    async def process_and_annotate_frame(self, frame: np.ndarray, results: Dict, perf_stats: Dict, playback_info: Dict) -> np.ndarray:
        """프레임 처리 및 주석 추가"""
        pass
    
    @abstractmethod
    def get_latest_metrics(self) -> AdvancedMetrics:
        """최신 메트릭 반환"""
        pass


class ISystemOrchestrator(ABC):
    """시스템 오케스트레이터 인터페이스 - 전체 시스템 조율"""
    
    @abstractmethod
    def initialize_system(self) -> bool:
        """시스템 초기화"""
        pass
    
    @abstractmethod
    async def run_analysis_cycle(self) -> None:
        """분석 사이클 실행"""
        pass
    
    @abstractmethod
    def shutdown_system(self) -> None:
        """시스템 종료"""
        pass


# === S-Class 고급 인터페이스들 ===

class ISClassProcessor(IDataProcessor):
    """S-Class 프로세서 고급 인터페이스"""
    
    @abstractmethod
    async def process_data(self, data: Any, image: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """S-Class 프로세서는 이미지 데이터도 함께 처리"""
        pass
    
    @abstractmethod
    def get_advanced_capabilities(self) -> List[str]:
        """고급 기능 목록 반환"""
        pass
    
    @abstractmethod
    def get_processing_quality_score(self) -> float:
        """처리 품질 점수 반환"""
        pass


class IAdvancedFaceProcessor(ISClassProcessor):
    """고급 얼굴 처리 인터페이스 (rPPG, saccade 등)"""
    
    @abstractmethod
    def extract_rppg_signal(self, image: np.ndarray, landmarks: Any) -> Dict[str, Any]:
        """rPPG 신호 추출"""
        pass
    
    @abstractmethod
    def analyze_saccadic_movement(self, landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """안구 사케이드 분석"""
        pass
    
    @abstractmethod
    def analyze_pupil_dynamics(self, blendshapes: Any) -> Dict[str, Any]:
        """동공 역학 분석"""
        pass
    
    @abstractmethod
    def estimate_heart_rate(self) -> Dict[str, Any]:
        """심박수 추정"""
        pass


class IAdvancedPoseProcessor(ISClassProcessor):
    """고급 자세 처리 인터페이스 (생체역학, 척추 분석 등)"""
    
    @abstractmethod
    def analyze_spinal_alignment(self, landmarks: Any) -> Dict[str, Any]:
        """척추 정렬 분석"""
        pass
    
    @abstractmethod
    def measure_postural_sway(self, landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """자세 흔들림 측정"""
        pass
    
    @abstractmethod
    def detect_forward_head_posture(self, landmarks: Any) -> Dict[str, Any]:
        """전방 머리 자세 감지"""
        pass


class IAdvancedHandProcessor(ISClassProcessor):
    """고급 손 처리 인터페이스 (키네마틱스, 떨림 분석 등)"""
    
    @abstractmethod
    def analyze_hand_tremor(self, landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """손 떨림 분석"""
        pass
    
    @abstractmethod
    def evaluate_grip_quality(self, landmarks: Any) -> Dict[str, Any]:
        """그립 품질 평가"""
        pass
    
    @abstractmethod
    def assess_steering_skill(self, landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """조향 기술 평가"""
        pass


class IAdvancedObjectProcessor(ISClassProcessor):
    """고급 객체 처리 인터페이스 (행동 예측, 베이지안 추론 등)"""
    
    @abstractmethod
    def predict_driver_intent(self, detections: Any, hand_positions: List[Dict]) -> Dict[str, Any]:
        """운전자 의도 예측"""
        pass
    
    @abstractmethod
    def generate_attention_heatmap(self, detections: Any) -> Dict[str, Any]:
        """주의집중 히트맵 생성"""
        pass
    
    @abstractmethod
    def assess_contextual_risk(self, detections: Any, context: Dict) -> Dict[str, Any]:
        """상황적 위험 평가"""
        pass


# === 이벤트 시스템 인터페이스들 ===

class IEventHandler(ABC):
    """이벤트 핸들러 기본 인터페이스"""
    
    @abstractmethod
    async def handle_event(self, event: Any) -> bool:
        """이벤트 처리"""
        pass
    
    @abstractmethod
    def can_handle(self, event_type: Any) -> bool:
        """이벤트 처리 가능 여부"""
        pass
    
    @abstractmethod
    def get_handler_name(self) -> str:
        """핸들러 이름 반환"""
        pass


class IEventBus(ABC):
    """이벤트 버스 인터페이스"""
    
    @abstractmethod
    async def publish(self, event: Any) -> None:
        """이벤트 발행"""
        pass
    
    @abstractmethod
    def subscribe(self, handler: IEventHandler, event_types: Optional[List] = None) -> None:
        """핸들러 등록"""
        pass
    
    @abstractmethod
    def unsubscribe(self, handler: IEventHandler) -> None:
        """핸들러 등록 해제"""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """이벤트 통계 반환"""
        pass


# === 융합 엔진 고급 인터페이스들 ===

class IAdvancedMultiModalAnalyzer(IMultiModalAnalyzer):
    """고급 멀티모달 분석 인터페이스"""
    
    @abstractmethod
    def calculate_attention_weights(self, modality_data: List[Dict]) -> Dict[str, float]:
        """어텐션 가중치 계산"""
        pass
    
    @abstractmethod
    def assess_cognitive_load(self, signals: Dict[str, float]) -> Any:
        """인지 부하 평가"""
        pass
    
    @abstractmethod
    def calculate_multitasking_penalty(self, signals: Dict[str, float]) -> float:
        """멀티태스킹 페널티 계산"""
        pass
    
    @abstractmethod
    def get_fusion_confidence(self) -> Dict[str, Any]:
        """융합 신뢰도 반환"""
        pass


# === 오케스트레이터 인터페이스들 ===

class IAnalysisOrchestrator(ABC):
    """분석 오케스트레이터 인터페이스"""
    
    @abstractmethod
    async def process_frame_data(self, mediapipe_results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """프레임 데이터 처리"""
        pass
    
    @abstractmethod
    def get_system_health(self) -> Dict[str, Any]:
        """시스템 건강도 반환"""
        pass
    
    @abstractmethod
    def adapt_pipeline_mode(self, health_status: Dict[str, Any]) -> None:
        """파이프라인 모드 적응"""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 반환"""
        pass


# === 팩토리 시스템 인터페이스들 ===

class IAnalysisSystemFactory(ABC):
    """분석 시스템 팩토리 인터페이스"""
    
    @abstractmethod
    def create_system(self, metrics_updater: IMetricsUpdater, custom_config: Optional[Dict[str, Any]] = None) -> IAnalysisOrchestrator:
        """분석 시스템 생성"""
        pass
    
    @abstractmethod
    def get_system_configuration(self) -> Any:
        """시스템 구성 정보 반환"""
        pass
    
    @abstractmethod
    def get_supported_features(self) -> List[str]:
        """지원 기능 목록 반환"""
        pass


# === 리소스 관리 인터페이스들 ===

class IResourcePredictor(ABC):
    """리소스 예측기 인터페이스"""
    
    @abstractmethod
    def predict_next_frame_load(self, mediapipe_results: Dict, processor_performance: Dict) -> Dict[str, float]:
        """다음 프레임 부하 예측"""
        pass
    
    @abstractmethod
    def update_model(self, execution_report: Any) -> None:
        """예측 모델 업데이트"""
        pass


class IAdaptiveTimeoutManager(ABC):
    """적응형 타임아웃 관리 인터페이스"""
    
    @abstractmethod
    def adjust_timeouts(self, predicted_load: Dict[str, float]) -> None:
        """타임아웃 조정"""
        pass
    
    @abstractmethod
    def get_current_timeouts(self) -> Dict[str, float]:
        """현재 타임아웃 반환"""
        pass


# === 통합 시스템 인터페이스 ===

class IIntegratedDMSSystem(ABC):
    """통합 DMS 시스템 인터페이스"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """시스템 초기화"""
        pass
    
    @abstractmethod
    async def process_and_annotate_frame(self, frame_data: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """프레임 처리 및 주석"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """시스템 종료"""
        pass
    
    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        pass


# === 메트릭 업데이터 확장 ===

class IAdvancedMetricsUpdater(IMetricsUpdater):
    """고급 메트릭 업데이터 인터페이스"""
    
    @abstractmethod
    def update_saccade_metrics(self, saccade_data: Dict[str, Any]) -> None:
        """사케이드 메트릭 업데이트"""
        pass
    
    @abstractmethod
    def update_pupil_metrics(self, pupil_data: Dict[str, Any]) -> None:
        """동공 메트릭 업데이트"""
        pass
    
    @abstractmethod
    def update_rppg_metrics(self, rppg_data: Dict[str, Any]) -> None:
        """rPPG 메트릭 업데이트"""
        pass
    
    @abstractmethod
    def update_cognitive_load_metrics(self, cognitive_data: Dict[str, Any]) -> None:
        """인지 부하 메트릭 업데이트"""
        pass
    
    @abstractmethod
    def update_system_performance_metrics(self, performance_data: Dict[str, Any]) -> None:
        """시스템 성능 메트릭 업데이트"""
        pass


# 타입 힌트를 위한 타입 정의들
ProcessorResult = Dict[str, Any]
MetricsUpdate = Dict[str, Any]
SystemHealth = Dict[str, Any]
ConfigurationData = Dict[str, Any]
EventData = Dict[str, Any]
FusionResult = Dict[str, Any]
PerformanceReport = Dict[str, Any]