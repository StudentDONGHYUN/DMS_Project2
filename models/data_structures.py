"""
Data structures for the DMS system
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, Any


# UIMode, EmotionState, AlertType 등 필요한 Enum 정의
class UIMode(Enum):
    MINIMAL = auto()
    STANDARD = auto()
    ALERT = auto()


class UIState(Enum):
    IDLE = 0
    RUNNING = 1
    PAUSED = 2
    STOPPED = 3


class EmotionState(Enum):
    NEUTRAL = auto()
    HAPPY = auto()
    SAD = auto()
    ANGRY = auto()
    STRESSED = auto()
    FATIGUED = auto()
    # 필요시 추가


class AlertType(Enum):
    """경고 타입"""

    NONE = auto()
    DROWSINESS = auto()
    DISTRACTION = auto()
    PHONE_USAGE = auto()
    MICROSLEEP = auto()
    EMOTIONAL_STRESS = auto()
    # 필요시 추가


@dataclass
class BiometricData:
    """생체 측정 데이터"""

    heart_rate: Optional[float] = None
    stress_level: Optional[float] = None
    skin_temperature: Optional[float] = None


@dataclass
class GazeData:
    """시선 데이터"""

    attention_score: float = 1.0
    gaze_zone: str = "forward"
    duration: float = 0.0


@dataclass
class FaceData:
    """얼굴 분석 데이터"""

    emotion_state: EmotionState = EmotionState.NEUTRAL
    drowsiness_score: float = 0.0
    yawn_count: int = 0


@dataclass
class SystemHealth:
    """시스템 상태 데이터"""

    processing_fps: float = 30.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    overall_status: str = "healthy"


@dataclass
class UIState:
    """UI 상태 관리 클래스"""

    # 핵심 위험도 점수
    risk_score: float = 0.0
    overall_safety_status: str = "safe"

    # UI 모드
    ui_mode: UIMode = UIMode.STANDARD
    active_alert_type: AlertType = AlertType.NONE

    # 상세 데이터
    biometrics: BiometricData = None
    gaze: GazeData = None
    face: FaceData = None
    system_health: SystemHealth = None

    def __post_init__(self):
        if self.biometrics is None:
            self.biometrics = BiometricData()
        if self.gaze is None:
            self.gaze = GazeData()
        if self.face is None:
            self.face = FaceData()
        if self.system_health is None:
            self.system_health = SystemHealth()

    def update_ui_mode_from_risk(self):
        """위험도에 따라 UI 모드 자동 설정"""
        if self.risk_score < 0.3:
            self.ui_mode = UIMode.MINIMAL
        elif self.risk_score < 0.7:
            self.ui_mode = UIMode.STANDARD
        else:
            self.ui_mode = UIMode.ALERT

    def get_primary_concern(self) -> str:
        """주요 우려사항 반환"""
        if self.active_alert_type == AlertType.DROWSINESS:
            return "DROWSINESS DETECTED"
        elif self.active_alert_type == AlertType.DISTRACTION:
            return "DISTRACTION WARNING"
        elif self.active_alert_type == AlertType.PHONE_USAGE:
            return "PHONE USAGE DETECTED"
        elif self.active_alert_type == AlertType.MICROSLEEP:
            return "MICROSLEEP ALERT"
        elif self.active_alert_type == AlertType.EMOTIONAL_STRESS:
            return "EMOTIONAL STRESS"
        elif self.risk_score > 0.7:
            return "HIGH RISK"
        else:
            return "MONITORING"
