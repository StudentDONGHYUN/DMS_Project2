from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import List

class RiskLevel(IntEnum):
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class DriverState(Enum):
    SAFE = "SAFE"
    FATIGUE_LOW = "FATIGUE_LOW"
    FATIGUE_HIGH = "FATIGUE_HIGH"
    DISTRACTION_NORMAL = "DISTRACTION_NORMAL"
    DISTRACTION_DANGER = "DISTRACTION_DANGER"
    PHONE_USAGE = "PHONE_USAGE"
    MICROSLEEP = "MICROSLEEP"
    MULTIPLE_RISK = "MULTIPLE_RISK"
    EMOTIONAL_STRESS = "EMOTIONAL_STRESS"
    PREDICTIVE_WARNING = "PREDICTIVE_WARNING"


class AnalysisEvent(Enum):
    BLINK_DETECTED = "BLINK_DETECTED"
    YAWN_DETECTED = "YAWN_DETECTED"
    HEAD_NOD_DETECTED = "HEAD_NOD_DETECTED"
    GAZE_NORMAL = "GAZE_NORMAL"
    GAZE_MIRROR = "GAZE_MIRROR"
    GAZE_DASHBOARD = "GAZE_DASHBOARD"
    GAZE_DANGER = "GAZE_DANGER"
    HAND_OFF_WHEEL = "HAND_OFF_WHEEL"
    PHONE_USAGE_CONFIRMED = "PHONE_USAGE_CONFIRMED"
    FATIGUE_ACCUMULATION = "FATIGUE_ACCUMULATION"
    ATTENTION_DECLINE = "ATTENTION_DECLINE"
    NORMAL_BEHAVIOR = "NORMAL_BEHAVIOR"
    FACE_LOST = "FACE_LOST"
    POSE_LOST = "POSE_LOST"
    HAND_OUT_OF_BOUNDS = "HAND_OUT_OF_BOUNDS"
    INTERACTING_WITH_DASHBOARD = "INTERACTING_WITH_DASHBOARD"
    INTERACTING_WITH_GEAR = "INTERACTING_WITH_GEAR"
    INTERACTING_WITH_MIRROR = "INTERACTING_WITH_MIRROR"
    EMOTION_STRESS_DETECTED = "EMOTION_STRESS_DETECTED"
    DISTRACTION_OBJECT_DETECTED = "DISTRACTION_OBJECT_DETECTED"
    PREDICTIVE_RISK_HIGH = "PREDICTIVE_RISK_HIGH"
    DRIVER_IDENTIFIED = "DRIVER_IDENTIFIED"
    MICROSLEEP_PREDICTED = "MICROSLEEP_PREDICTED"


class GazeZone(Enum):
    FRONT = "FRONT"
    INSTRUMENT_CLUSTER = "INSTRUMENT_CLUSTER"
    CENTER_STACK = "CENTER_STACK"
    REARVIEW_MIRROR = "REARVIEW_MIRROR"
    LEFT_SIDE_MIRROR = "LEFT_SIDE_MIRROR"
    DRIVER_WINDOW = "DRIVER_WINDOW"
    RIGHT_SIDE_MIRROR = "RIGHT_SIDE_MIRROR"
    PASSENGER = "PASSENGER"
    ROOF = "ROOF"
    FLOOR = "FLOOR"
    BLIND_SPOT_LEFT = "BLIND_SPOT_LEFT"
    UNKNOWN = "UNKNOWN"


class EmotionState(Enum):
    NEUTRAL = "중립"
    HAPPINESS = "기쁨"
    SADNESS = "슬픔"
    ANGER = "분노"
    FEAR = "두려움"
    SURPRISE = "놀람"
    DISGUST = "혐오"
    STRESS = "스트레스"
    FATIGUE = "피로"


@dataclass
class TimeWindowConfig:
    blink_analysis: float = 1.0
    yawn_detection: float = 3.0
    fatigue_assessment: float = 30.0
    trend_analysis: float = 300.0
    gaze_tracking: float = 2.0
    emotion_analysis: float = 5.0
    prediction_window: float = 30.0


@dataclass
class AdvancedMetrics:
    yawn_score: float = 0.0
    left_eye_closure: float = 0.0
    right_eye_closure: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0
    shoulder_yaw: float = 0.0
    blink_count_1min: int = 0
    yawn_count_5min: int = 0
    head_nod_count_2min: int = 0
    gaze_deviation_count_1min: int = 0
    perclos: float = 0.0
    eyes_closed_duration: float = 0.0
    longest_gaze_away: float = 0.0
    current_gaze_zone: GazeZone = GazeZone.FRONT
    gaze_zone_duration: float = 0.0
    driving_context_score: float = 1.0
    left_hand_in_safe_zone: bool = True
    right_hand_in_safe_zone: bool = True
    hand_stability_score: float = 1.0
    phone_detected: bool = False
    phone_hand_proximity: float = 0.0
    fatigue_risk_score: float = 0.0
    distraction_risk_score: float = 0.0
    overall_risk_level: RiskLevel = RiskLevel.SAFE
    enhanced_ear: float = 0.0
    temporal_attention_score: float = 0.0
    emotion_state: EmotionState = EmotionState.NEUTRAL
    emotion_confidence: float = 0.0
    arousal_level: float = 0.0
    valence_level: float = 0.0
    distraction_objects: List[str] = field(default_factory=list)
    predictive_risk_score: float = 0.0
    driver_identity: str = "unknown"
    driver_confidence: float = 0.0
    personalized_threshold: float = 0.25
    pose_complexity_score: float = 0.0
    attention_focus_score: float = 1.0

class CameraPosition(Enum):
    REARVIEW_MIRROR = "백미러 위치"
    DASHBOARD_CENTER = "대시보드 중앙"
    A_PILLAR_LEFT = "A필러 좌측"
    SUN_VISOR_LEFT = "선바이저 좌측"

    def __str__(self):
        return self.value
