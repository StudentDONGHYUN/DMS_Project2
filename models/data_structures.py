"""
S-Class DMS v18+ Data Structures
UI-백엔드 간 데이터 계약 정의 (BFF 패턴)
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
import time


class UIMode(Enum):
    """적응형 UI 모드"""
    MINIMAL = "minimal"        # risk_score < 0.3: 핵심 정보만
    STANDARD = "standard"      # 0.3 <= risk_score < 0.7: 주요 분석 정보
    ALERT = "alert"            # risk_score >= 0.7: 위험 요소 강조


class AlertType(Enum):
    """경고 유형"""
    NONE = "none"
    FATIGUE = "fatigue"
    DISTRACTION = "distraction"
    STRESS = "stress"
    HEALTH_ALERT = "health_alert"
    CRITICAL = "critical"


class EmotionState(Enum):
    """감정 상태 (Emotion AI)"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    STRESSED = "stressed"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    FATIGUE = "fatigue"
    DROWSY = "drowsy"


@dataclass
class BiometricData:
    """생체 정보 데이터"""
    heart_rate: Optional[float] = None
    heart_rate_variability: Optional[float] = None
    breathing_rate: Optional[float] = None
    blood_pressure_systolic: Optional[float] = None
    blood_pressure_diastolic: Optional[float] = None
    stress_level: Optional[float] = None  # 0.0-1.0
    confidence: float = 0.0


@dataclass
class GazeData:
    """시선 데이터"""
    x: float = 0.0
    y: float = 0.0
    fixation_duration: Optional[float] = None
    saccade_velocity: Optional[float] = None
    attention_score: float = 0.0  # 0.0-1.0
    distraction_level: float = 0.0  # 0.0-1.0


@dataclass
class PostureData:
    """자세 데이터"""
    spinal_alignment_score: float = 0.0  # 0.0-1.0
    head_tilt_angle: float = 0.0
    shoulder_symmetry: float = 0.0
    posture_stability: float = 0.0
    fatigue_indicators: Dict[str, float] = None

    def __post_init__(self):
        if self.fatigue_indicators is None:
            self.fatigue_indicators = {}


@dataclass
class HandData:
    """손 데이터"""
    tremor_frequency: Optional[float] = None
    grip_stability: float = 0.0
    motor_control_score: float = 0.0
    hand_position_confidence: float = 0.0
    detected_hands: int = 0


@dataclass
class FaceData:
    """얼굴 데이터"""
    emotion_state: EmotionState = EmotionState.NEUTRAL
    emotion_confidence: float = 0.0
    eye_closure_rate: float = 0.0  # PERCLOS
    blink_frequency: float = 0.0
    yawn_detection: bool = False
    micro_expressions: Dict[str, float] = None

    def __post_init__(self):
        if self.micro_expressions is None:
            self.micro_expressions = {}


@dataclass
class PredictiveData:
    """예측 데이터 (Predictive Safety)"""
    predicted_event: Optional[str] = None  # 예: 'lane_departure_imminent'
    prediction_confidence: float = 0.0
    time_to_event: Optional[float] = None  # 초 단위
    risk_factors: List[str] = None
    intervention_suggestions: List[str] = None

    def __post_init__(self):
        if self.risk_factors is None:
            self.risk_factors = []
        if self.intervention_suggestions is None:
            self.intervention_suggestions = []


@dataclass
class SystemHealth:
    """시스템 상태"""
    overall_status: str = "healthy"  # healthy, degraded, error
    processing_fps: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_processors: List[str] = None
    error_messages: List[str] = None

    def __post_init__(self):
        if self.active_processors is None:
            self.active_processors = []
        if self.error_messages is None:
            self.error_messages = []


@dataclass
class UIState:
    """
    UI-백엔드 간 데이터 계약 정의
    UI 렌더링에 필요한 모든 데이터를 포함
    """
    # 핵심 위험 점수
    risk_score: float = 0.0  # 0.0-1.0
    overall_safety_status: str = "safe"  # safe, warning, danger, critical
    
    # UI 모드 제어
    ui_mode: UIMode = UIMode.STANDARD
    active_alert_type: AlertType = AlertType.NONE
    
    # 생체 정보
    biometrics: BiometricData = None
    
    # 행동 분석
    gaze: GazeData = None
    posture: PostureData = None
    hands: HandData = None
    face: FaceData = None
    
    # 예측 안전
    predictions: PredictiveData = None
    
    # 시스템 정보
    system_health: SystemHealth = None
    
    # 추가 메타데이터
    timestamp: float = 0.0
    session_duration: float = 0.0
    user_id: Optional[str] = None
    
    # 개인화 데이터
    personalization_active: bool = False
    user_baseline_established: bool = False
    adaptation_level: float = 0.0  # 시스템이 사용자에게 얼마나 적응했는지
    
    # 연구/디버그 정보
    debug_info: Dict[str, Any] = None
    raw_sensor_confidence: Dict[str, float] = None

    def __post_init__(self):
        """기본값 초기화"""
        if self.biometrics is None:
            self.biometrics = BiometricData()
        if self.gaze is None:
            self.gaze = GazeData()
        if self.posture is None:
            self.posture = PostureData()
        if self.hands is None:
            self.hands = HandData()
        if self.face is None:
            self.face = FaceData()
        if self.predictions is None:
            self.predictions = PredictiveData()
        if self.system_health is None:
            self.system_health = SystemHealth()
        if self.debug_info is None:
            self.debug_info = {}
        if self.raw_sensor_confidence is None:
            self.raw_sensor_confidence = {}
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def update_ui_mode_from_risk(self):
        """위험 점수에 따라 UI 모드 자동 조정"""
        if self.risk_score < 0.3:
            self.ui_mode = UIMode.MINIMAL
        elif self.risk_score < 0.7:
            self.ui_mode = UIMode.STANDARD
        else:
            self.ui_mode = UIMode.ALERT

    def get_primary_concern(self) -> str:
        """주요 우려사항 반환"""
        if self.face.emotion_state == EmotionState.FATIGUE:
            return "Fatigue Detection"
        elif self.gaze.distraction_level > 0.7:
            return "Attention Distraction"
        elif self.biometrics.stress_level and self.biometrics.stress_level > 0.8:
            return "High Stress Level"
        elif self.predictions.predicted_event:
            return f"Predicted: {self.predictions.predicted_event}"
        else:
            return "Normal Operation"

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (API 응답용)"""
        return {
            'risk_score': self.risk_score,
            'safety_status': self.overall_safety_status,
            'ui_mode': self.ui_mode.value,
            'alert_type': self.active_alert_type.value,
            'primary_concern': self.get_primary_concern(),
            'biometrics': {
                'heart_rate': self.biometrics.heart_rate,
                'stress_level': self.biometrics.stress_level,
                'confidence': self.biometrics.confidence
            },
            'gaze': {
                'attention_score': self.gaze.attention_score,
                'distraction_level': self.gaze.distraction_level
            },
            'predictions': {
                'event': self.predictions.predicted_event,
                'confidence': self.predictions.prediction_confidence
            },
            'system': {
                'status': self.system_health.overall_status,
                'fps': self.system_health.processing_fps
            },
            'timestamp': self.timestamp
        }


@dataclass
class ProcessorOutput:
    """프로세서별 출력 표준화"""
    processor_name: str
    confidence: float
    processing_time_ms: float
    data: Dict[str, Any]
    timestamp: float = 0.0
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class FusionEngineOutput:
    """MultiModalFusionEngine 출력"""
    ui_state: UIState
    individual_processors: List[ProcessorOutput]
    fusion_confidence: float
    processing_summary: Dict[str, Any]
    recommendations: List[str]