"""
통합 설정 관리 시스템
모든 시스템 설정값들을 중앙에서 관리합니다.
"""

from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path


@dataclass
class PerformanceConfig:
    """성능 관련 설정"""
    target_fps: int = 30
    min_fps: int = 15
    max_processing_time_ms: float = 200.0
    fps_avg_frame_count: int = 10
    performance_check_interval: int = 30  # frames
    optimization_threshold_multiplier: float = 1.2
    
    # 로깅 설정
    log_interval_ms: float = 30.0  # 30ms 간격
    max_log_entries: int = 300


@dataclass
class DrowsinessConfig:
    """졸음 감지 관련 설정"""
    # EAR 계산 설정 (RB2 최적화: 메모리 사용량 75% 감소)
    ear_history_size: int = 150  # 10초 @ 15fps (RB2 최적화)
    calibration_frames: int = 150  # 10초 보정 (RB2 최적화)
    default_ear_threshold: float = 0.25
    min_ear_threshold: float = 0.15
    max_ear_threshold: float = 0.35
    
    # 시간적 어텐션 모델
    temporal_window_size: int = 30  # 1초 윈도우
    attention_weight_decay: float = 2.0  # exp(-2 to 0)
    
    # PERCLOS 계산
    perclos_window_frames: int = 30  # 1초
    
    # 마이크로슬립 감지
    microsleep_threshold: float = 0.15
    microsleep_min_duration: float = 0.5  # 초
    microsleep_max_duration: float = 3.0  # 초
    microsleep_analysis_frames: int = 90  # 3초


@dataclass
class EmotionConfig:
    """감정 인식 관련 설정"""
    emotion_history_size: int = 75  # 5초 히스토리 (RB2 최적화)
    temporal_smoothing_window: int = 30  # 1초
    min_confidence_threshold: float = 0.3
    
    # 감정 분류 가중치
    emotion_weights: Dict[str, Dict[int, float]] = None
    
    def __post_init__(self):
        if self.emotion_weights is None:
            self.emotion_weights = {
                'happiness': {8: 1.0, 5: 0.8},  # AU12, AU6
                'sadness': {9: 1.0, 2: 0.6, 0: 0.4},  # AU15, AU4, AU1
                'anger': {2: 1.0, 6: 0.8, 12: 0.6},  # AU4, AU7, AU23
                'fear': {0: 1.0, 4: 0.8, 13: 0.6},  # AU1, AU5, AU25
                'surprise': {0: 1.0, 1: 0.8, 4: 0.8, 13: 0.6},
                'disgust': {7: 1.0, 6: 0.6},  # AU9, AU7
            }


@dataclass
class HandConfig:
    """손 분석 관련 설정"""
    # FFT 분석 설정 (RB2 최적화)
    fft_buffer_size: int = 75  # 5초 @ 15fps (RB2 최적화)
    fft_min_samples: int = 15   # 최소 FFT 샘플 수 (RB2 최적화)
    
    # 제스처 분석 설정
    gesture_buffer_size: int = 100  # 제스처 시퀀스 버퍼
    
    # 손 떨림 및 안정성 분석
    jerk_limit: float = 10.0  # 저크 한계값 (m/s³)
    tremor_analysis_window: int = 90  # 3초 떨림 분석 윈도우
    
    # 핸들 그립 분석
    grip_quality_threshold: float = 0.7
    precision_grip_threshold: float = 0.05  # 정밀 그립 거리 임계값
    
    # 주의산만 감지
    distraction_velocity_threshold: float = 0.25  # 빠른 움직임 임계값
    safe_zone_proximity_threshold: float = 0.02  # 안전 구역 근접 임계값
    
    # 핸들링 스킬 평가
    min_steering_smoothness: float = 0.6
    optimal_clock_positions: list = None  # 최적 핸들 그립 위치
    
    def __post_init__(self):
        if self.optimal_clock_positions is None:
            # 권장 핸들 그립 위치 (시계 방향)
            self.optimal_clock_positions = [(2, 10), (3, 9), (4, 8)]


@dataclass
class FaceConfig:
    """얼굴 분석 관련 설정"""
    # EMA 필터 설정
    ema_filter_size: int = 10
    ema_alpha: float = 0.3
    
    # Saccade 분석 설정
    saccade_history_size: int = 30
    saccade_min_samples: int = 10
    saccade_velocity_threshold: float = 0.05
    fixation_dispersion_max: float = 0.1


@dataclass
class RPPGConfig:
    """rPPG (원격 광혈류측정) 관련 설정"""
    fps: float = 30.0
    window_size_s: float = 10.0  # 10초 윈도우
    run_interval: int = 5  # 5프레임마다 실행
    
    # 신호 필터링
    low_cut_hz: float = 0.8  # 48 BPM
    high_cut_hz: float = 3.0  # 180 BPM
    
    # 품질 임계값
    snr_threshold: float = 2.0
    hrv_std_threshold: float = 100.0  # ms


@dataclass
class GazeConfig:
    """시선 분석 관련 설정"""
    gaze_history_size: int = 30  # 1초 히스토리
    min_stability_samples: int = 10
    min_focus_samples: int = 5
    
    # 시선 구역 정의 (각도 기반)
    zone_tolerances: Dict[str, float] = None
    
    # 주의집중 점수 계산
    front_gaze_weight: float = 1.0
    mirror_gaze_weight: float = 0.5
    mirror_ratio_range: tuple = (0.1, 0.3)  # 적절한 미러 체크 비율
    
    def __post_init__(self):
        if self.zone_tolerances is None:
            self.zone_tolerances = {
                'FRONT': 25.0,
                'REARVIEW_MIRROR': 15.0,
                'ROOF': 20.0,
                'INSTRUMENT_CLUSTER': 20.0,
                'CENTER_STACK': 18.0,
                'FLOOR': 25.0,
                'LEFT_SIDE_MIRROR': 12.0,
                'DRIVER_WINDOW': 30.0,
                'BLIND_SPOT_LEFT': 20.0,
                'RIGHT_SIDE_MIRROR': 12.0,
                'PASSENGER': 25.0
            }


@dataclass
class PredictionConfig:
    """예측적 안전 관련 설정"""
    prediction_window: float = 30.0  # 예측 윈도우 (초)
    feature_history_size: int = 150  # 10초 @ 15fps (RB2 최적화)
    min_prediction_samples: int = 75  # 최소 5초 데이터 (RB2 최적화)
    
    # 특징 가중치 (명시적으로 정의)
    feature_weights: Dict[str, float] = None
    
    # 위험도 임계값
    risk_thresholds: Dict[str, float] = None
    
    # 추세 분석 설정 (RB2 최적화)
    trend_window_size: int = 75  # 5초 (RB2 최적화)
    min_trend_samples: int = 5   # RB2 최적화
    trend_acceleration_threshold: int = 10  # 가속도 계산을 위한 최소 샘플 (RB2 최적화)
    
    def __post_init__(self):
        if self.feature_weights is None:
            self.feature_weights = {
                'fatigue_risk': 0.2,
                'distraction_risk': 0.2,
                'enhanced_ear': 0.15,
                'perclos': 0.15,
                'temporal_attention': 0.1,
                'head_yaw_normalized': 0.05,
                'head_pitch_normalized': 0.05,
                'arousal_level': 0.02,
                'attention_focus_inverted': 0.02,
                'phone_detected': 0.02,
                'distraction_objects_normalized': 0.02,
                'stress_confidence': 0.02
            }
        
        if self.risk_thresholds is None:
            self.risk_thresholds = {
                'low': 0.2,
                'medium': 0.4,
                'high': 0.6,
                'critical': 0.8
            }


@dataclass
class DistractionConfig:
    """주의산만 감지 관련 설정"""
    detection_history_size: int = 90  # 3초 히스토리
    
    # 주의산만 객체별 위험도
    object_risk_levels: Dict[str, Dict[str, any]] = None
    
    # 손과 객체 근접성 계산
    proximity_threshold: float = 0.5
    persistent_risk_frames: int = 30  # 1초
    persistent_risk_threshold: float = 0.5
    
    def __post_init__(self):
        if self.object_risk_levels is None:
            self.object_risk_levels = {
                "cell phone": {"risk_level": 0.9, "description": "휴대폰"},
                "cup": {"risk_level": 0.6, "description": "컵"},
                "bottle": {"risk_level": 0.6, "description": "물병"},
                "sandwich": {"risk_level": 0.7, "description": "음식"},
                "book": {"risk_level": 0.8, "description": "책"},
                "laptop": {"risk_level": 0.9, "description": "노트북"},
                "remote": {"risk_level": 0.5, "description": "리모컨"},
            }


@dataclass
class MultiModalConfig:
    """멀티모달 융합 관련 설정"""
    # 각 모달리티별 가중치
    modality_weights: Dict[str, float] = None
    
    # 졸음 융합 가중치
    drowsiness_fusion_weights: Dict[str, float] = None
    
    # 주의산만 융합 가중치
    distraction_fusion_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.modality_weights is None:
            self.modality_weights = {
                "face": 0.35,
                "pose": 0.25,
                "hand": 0.20,
                "object": 0.10,
                "emotion": 0.10
            }
        
        if self.drowsiness_fusion_weights is None:
            self.drowsiness_fusion_weights = {
                "perclos": 0.4,
                "enhanced_ear": 0.3,
                "temporal_attention": 0.3
            }
        
        if self.distraction_fusion_weights is None:
            self.distraction_fusion_weights = {
                "predictive_weight": 0.3,
                "emotion_stress_bonus": 0.2,
                "pose_complexity_bonus": 0.1,
                "head_roll_threshold": 25.0,
                "gaze_duration_threshold": 1.0
            }


@dataclass
class FeatureFlagConfig:
    """기능 플래그 설정 (상용화 전략)"""
    # 시스템 에디션 (상용화 버전 제어)
    system_edition: str = "RESEARCH"  # COMMUNITY, PRO, ENTERPRISE, RESEARCH
    
    # 기본 Expert Systems (모든 에디션)
    enable_face_processor: bool = True
    enable_pose_processor: bool = True
    enable_hand_processor: bool = True
    enable_object_processor: bool = True
    
    # S-Class Advanced Features (PRO 이상)
    enable_rppg_heart_rate: bool = True
    enable_saccade_analysis: bool = True
    enable_spinal_alignment: bool = True
    enable_fft_tremor_analysis: bool = True
    enable_bayesian_prediction: bool = True
    
    # Neural AI Features (ENTERPRISE/RESEARCH)
    enable_emotion_ai: bool = True
    enable_predictive_safety: bool = True
    enable_biometric_fusion: bool = True
    enable_uncertainty_quantification: bool = True
    
    # Research Features (RESEARCH 전용)
    enable_digital_twin_simulation: bool = True
    enable_advanced_cognitive_modeling: bool = True
    enable_experimental_algorithms: bool = True
    enable_detailed_logging: bool = True
    
    # 혁신 기능들 (제안서 기반)
    enable_mental_wellness_monitoring: bool = True
    enable_edge_vision_transformer: bool = True
    enable_multimodal_sensor_fusion: bool = True
    enable_adaptive_ui_modes: bool = True
    enable_context_aware_ar_hud: bool = False  # 하드웨어 의존
    enable_v2d_healthcare_platform: bool = True
    
    def __post_init__(self):
        """에디션별 기능 제한 적용 - Bug fix: Add missing properties"""
        # Bug fix: Define convenient property aggregates for main.py compatibility
        self.basic_expert_systems = (
            self.enable_face_processor and 
            self.enable_pose_processor and 
            self.enable_hand_processor and 
            self.enable_object_processor
        )
        
        self.s_class_advanced_features = (
            self.enable_rppg_heart_rate and
            self.enable_saccade_analysis and
            self.enable_spinal_alignment and
            self.enable_fft_tremor_analysis and
            self.enable_bayesian_prediction
        )
        
        self.neural_ai_features = (
            self.enable_emotion_ai and
            self.enable_predictive_safety and
            self.enable_biometric_fusion and
            self.enable_uncertainty_quantification
        )
        
        self.innovation_research_features = (
            self.enable_digital_twin_simulation and
            self.enable_advanced_cognitive_modeling and
            self.enable_experimental_algorithms and
            self.enable_detailed_logging
        )
        
        if self.system_edition == "COMMUNITY":
            # 커뮤니티 에디션: 기본 기능만
            self._disable_advanced_features()
            self._disable_neural_ai_features()
            self._disable_research_features()
            self._disable_innovation_features()
        
        elif self.system_edition == "PRO":
            # 프로 에디션: S-Class 고급 기능 포함
            self._disable_neural_ai_features()
            self._disable_research_features()
            self._disable_innovation_features()
        
        elif self.system_edition == "ENTERPRISE":
            # 엔터프라이즈 에디션: Neural AI 포함, 연구 기능 제외
            self._disable_research_features()
        
        # Bug fix: Recalculate property aggregates after edition-based disabling
        self._update_property_aggregates()
    
    def _disable_advanced_features(self):
        """고급 기능 비활성화"""
        self.enable_rppg_heart_rate = False
        self.enable_saccade_analysis = False
        self.enable_spinal_alignment = False
        self.enable_fft_tremor_analysis = False
        self.enable_bayesian_prediction = False
    
    def _disable_neural_ai_features(self):
        """Neural AI 기능 비활성화"""
        self.enable_emotion_ai = False
        self.enable_predictive_safety = False
        self.enable_biometric_fusion = False
        self.enable_uncertainty_quantification = False
    
    def _disable_research_features(self):
        """연구 기능 비활성화"""
        self.enable_digital_twin_simulation = False
        self.enable_advanced_cognitive_modeling = False
        self.enable_experimental_algorithms = False
        self.enable_detailed_logging = False
    
    def _disable_innovation_features(self):
        """혁신 기능 비활성화"""
        self.enable_mental_wellness_monitoring = False
        self.enable_edge_vision_transformer = False
        self.enable_multimodal_sensor_fusion = False
        self.enable_adaptive_ui_modes = False
        self.enable_v2d_healthcare_platform = False
    
    def _update_property_aggregates(self):
        """Bug fix: Update property aggregates after feature changes"""
        self.basic_expert_systems = (
            self.enable_face_processor and 
            self.enable_pose_processor and 
            self.enable_hand_processor and 
            self.enable_object_processor
        )
        
        self.s_class_advanced_features = (
            self.enable_rppg_heart_rate and
            self.enable_saccade_analysis and
            self.enable_spinal_alignment and
            self.enable_fft_tremor_analysis and
            self.enable_bayesian_prediction
        )
        
        self.neural_ai_features = (
            self.enable_emotion_ai and
            self.enable_predictive_safety and
            self.enable_biometric_fusion and
            self.enable_uncertainty_quantification
        )
        
        self.innovation_research_features = (
            self.enable_digital_twin_simulation and
            self.enable_advanced_cognitive_modeling and
            self.enable_experimental_algorithms and
            self.enable_detailed_logging
        )
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """특정 기능의 활성화 상태 확인"""
        return getattr(self, f'enable_{feature_name}', False)
    
    def get_edition_features(self) -> dict:
        """현재 에디션에서 사용 가능한 기능 목록 반환"""
        features = {}
        for attr_name in dir(self):
            if attr_name.startswith('enable_') and isinstance(getattr(self, attr_name), bool):
                feature_name = attr_name.replace('enable_', '')
                features[feature_name] = getattr(self, attr_name)
        return features


@dataclass
class SystemConfig:
    """전체 시스템 설정을 통합하는 메인 설정 클래스"""
    
    # 하위 설정 그룹들
    performance: PerformanceConfig = None
    drowsiness: DrowsinessConfig = None
    emotion: EmotionConfig = None
    face: FaceConfig = None
    rppg: RPPGConfig = None
    hand: HandConfig = None
    gaze: GazeConfig = None
    prediction: PredictionConfig = None
    distraction: DistractionConfig = None
    multimodal: MultiModalConfig = None
    feature_flags: FeatureFlagConfig = None
    
    # 전역 설정
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # 파일 경로 설정
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")
    profiles_dir: Path = Path("profiles")
    performance_logs_dir: Path = Path("performance_logs")
    
    # MediaPipe 관련 설정
    mediapipe_confidence_thresholds: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        # 하위 설정 객체들 초기화
        if self.performance is None:
            self.performance = PerformanceConfig()
        if self.drowsiness is None:
            self.drowsiness = DrowsinessConfig()
        if self.emotion is None:
            self.emotion = EmotionConfig()
        if self.face is None:
            self.face = FaceConfig()
        if self.rppg is None:
            self.rppg = RPPGConfig()
        if self.hand is None:
            self.hand = HandConfig()
        if self.gaze is None:
            self.gaze = GazeConfig()
        if self.prediction is None:
            self.prediction = PredictionConfig()
        if self.distraction is None:
            self.distraction = DistractionConfig()
        if self.multimodal is None:
            self.multimodal = MultiModalConfig()
        if self.feature_flags is None:
            self.feature_flags = FeatureFlagConfig()
        
        # MediaPipe 임계값 기본값 설정
        if self.mediapipe_confidence_thresholds is None:
            self.mediapipe_confidence_thresholds = {
                'pose': {
                    'detection_confidence': 0.7,
                    'presence_confidence': 0.9,
                    'tracking_confidence': 0.8
                },
                'hand': {
                    'detection_confidence': 0.4,
                    'presence_confidence': 0.5,
                    'tracking_confidence': 0.7
                },
                'object': {
                    'score_threshold': 0.3
                }
            }
        
        # 디렉토리 생성
        for dir_path in [self.models_dir, self.logs_dir, self.profiles_dir, self.performance_logs_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def validate(self) -> bool:
        """설정값들의 유효성을 검증합니다."""
        try:
            # 성능 설정 검증
            assert 0 < self.performance.min_fps <= self.performance.target_fps
            assert self.performance.max_processing_time_ms > 0
            
            # 졸음 설정 검증
            assert 0 < self.drowsiness.min_ear_threshold < self.drowsiness.max_ear_threshold
            assert self.drowsiness.calibration_frames > 0
            
            # 예측 설정 검증
            assert sum(self.prediction.feature_weights.values()) > 0.99  # 가중치 합이 1에 가까워야 함
            
            # 멀티모달 설정 검증
            assert abs(sum(self.multimodal.modality_weights.values()) - 1.0) < 0.01
            
            return True
        except AssertionError as e:
            print(f"설정 검증 실패: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> 'SystemConfig':
        """파일에서 설정을 로드합니다. (향후 확장 가능)"""
        # 현재는 기본 설정 반환, 나중에 JSON/YAML 로딩 구현 가능
        return cls()
    
    def save_to_file(self, config_path: Path):
        """설정을 파일에 저장합니다. (향후 확장 가능)"""
        # 현재는 구현하지 않음, 나중에 JSON/YAML 저장 구현 가능
        pass


# 전역 설정 인스턴스 (싱글톤 패턴)
_global_config = None

def get_config() -> SystemConfig:
    """전역 설정 인스턴스를 반환합니다."""
    global _global_config
    if _global_config is None:
        _global_config = SystemConfig()
        if not _global_config.validate():
            raise ValueError("시스템 설정 검증에 실패했습니다.")
    return _global_config

def set_config(config: SystemConfig):
    """전역 설정을 업데이트합니다."""
    global _global_config
    if not config.validate():
        raise ValueError("유효하지 않은 설정입니다.")
    _global_config = config