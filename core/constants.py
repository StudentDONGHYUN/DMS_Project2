"""
시스템 상수 정의
매직 넘버들을 의미있는 이름으로 정의하여 코드 가독성을 향상시킵니다.
"""

import numpy as np
from typing import List, Tuple
from core.definitions import GazeZone


class MediaPipeConstants:
    """MediaPipe 관련 상수들"""
    
    # Face Landmark 인덱스 (468개 랜드마크 기준)
    class FaceLandmarks:
        # 눈 관련 랜드마크
        LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153]  # 6점 EAR 계산용
        RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373]
        
        # 주요 얼굴 특징점 (운전자 식별용)
        LEFT_EYE_CENTER = 33
        RIGHT_EYE_CENTER = 263
        NOSE_TIP = 1
        MOUTH_LEFT = 61
        MOUTH_RIGHT = 291
        CHIN = 175
        
        # 머리 자세 추정용
        NOSE_CENTER = 1
        LEFT_EAR_POINT = 234
        RIGHT_EAR_POINT = 454
    
    # Pose Landmark 인덱스 (33개 키포인트 기준)
    class PoseLandmarks:
        # 상체 주요 포인트
        NOSE = 0
        LEFT_EYE_INNER = 1
        LEFT_EYE = 2
        LEFT_EYE_OUTER = 3
        RIGHT_EYE_INNER = 4
        RIGHT_EYE = 5
        RIGHT_EYE_OUTER = 6
        LEFT_EAR = 7
        RIGHT_EAR = 8
        MOUTH_LEFT = 9
        MOUTH_RIGHT = 10
        
        # 어깨와 팔
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        
        # 몸통
        LEFT_HIP = 23
        RIGHT_HIP = 24
        
        # 상체 분석용 그룹
        TORSO_POINTS = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
        FACE_POSE_POINTS = [NOSE, LEFT_EAR, RIGHT_EAR, MOUTH_LEFT, MOUTH_RIGHT]


class AnalysisConstants:
    """분석 관련 상수들"""
    
    # 시간 기반 상수 (모두 초 단위)
    class TimeWindows:
        BLINK_ANALYSIS = 1.0
        YAWN_DETECTION = 3.0
        FATIGUE_ASSESSMENT = 30.0
        TREND_ANALYSIS = 300.0  # 5분
        GAZE_TRACKING = 2.0
        EMOTION_ANALYSIS = 5.0
        PREDICTION_WINDOW = 30.0
    
    # 프레임 속도 관련
    class FrameRates:
        TARGET_FPS = 30
        MIN_ACCEPTABLE_FPS = 15
        CALIBRATION_SECONDS = 10
        CALIBRATION_FRAMES = TARGET_FPS * CALIBRATION_SECONDS
    
    # 임계값들
    class Thresholds:
        # 졸음 감지
        DEFAULT_EAR_THRESHOLD = 0.25
        BLINK_EAR_THRESHOLD = 0.8
        YAWN_SCORE_THRESHOLD = 0.6
        PERCLOS_CRITICAL = 0.7
        MICROSLEEP_EAR = 0.15
        
        # 머리 자세
        HEAD_YAW_DANGER = 45.0  # 도
        HEAD_YAW_EXTREME = 60.0  # 도
        HEAD_ROLL_EXTREME = 25.0  # 도
        HEAD_PITCH_LIMIT = 60.0  # 도
        
        # 위험도 분류
        RISK_LOW = 0.2
        RISK_MEDIUM = 0.4
        RISK_HIGH = 0.6
        RISK_CRITICAL = 0.8
        
        # 감정 인식
        EMOTION_CONFIDENCE_MIN = 0.3
        EMOTION_CONFIDENCE_HIGH = 0.7
        STRESS_DETECTION_THRESHOLD = 0.7
        
        # 주의 집중
        ATTENTION_FOCUS_LOW = 0.3
        MIRROR_CHECK_MIN_RATIO = 0.1
        MIRROR_CHECK_MAX_RATIO = 0.3
        
        # 성능 최적화
        PROCESSING_TIME_LIMIT = 200.0  # ms
        MEMORY_USAGE_WARNING = 85.0  # %
    
    # 시선 구역별 위험도 정의 (face_processor에서 사용)
    GazeZoneRisk = {
        GazeZone.FRONT: 0.0,
        GazeZone.REARVIEW_MIRROR: 0.2,
        GazeZone.LEFT_SIDE_MIRROR: 0.2,
        GazeZone.RIGHT_SIDE_MIRROR: 0.2,
        GazeZone.INSTRUMENT_CLUSTER: 0.1,
        GazeZone.CENTER_STACK: 0.4,
        GazeZone.FLOOR: 0.8,
        GazeZone.ROOF: 0.6,
        GazeZone.PASSENGER: 0.7,
        GazeZone.DRIVER_WINDOW: 0.5,
        GazeZone.BLIND_SPOT_LEFT: 0.9,
        GazeZone.UNKNOWN: 0.5
    }


class VehicleConstants:
    """차량 내부 영역 정의"""
    
    # 정규화된 좌표계 (0.0 ~ 1.0)
    class Zones:
        STEERING_WHEEL = {"x1": 0.3, "y1": 0.4, "x2": 0.7, "y2": 0.8}
        DASHBOARD_AREA = {"x1": 0.2, "y1": 0.0, "x2": 0.8, "y2": 0.3}
        GEAR_LEVER = {"x1": 0.4, "y1": 0.7, "x2": 0.6, "y2": 0.9}
        SIDE_MIRROR_LEFT = {"x1": 0.0, "y1": 0.2, "x2": 0.2, "y2": 0.4}
        CENTER_CONSOLE = {"x1": 0.45, "y1": 0.5, "x2": 0.55, "y2": 0.7}
    
    # 시선 구역별 3D 벡터와 허용 각도
    class GazeZones:
        DEFINITIONS = [
            ("FRONT", [0.0, 0.0, 1.0], 25.0),
            ("REARVIEW_MIRROR", [0.0, 0.5, 1.0], 15.0),
            ("ROOF", [0.0, 1.0, 0.3], 20.0),
            ("INSTRUMENT_CLUSTER", [0.0, -0.3, 1.0], 20.0),
            ("CENTER_STACK", [0.3, -0.2, 0.8], 18.0),
            ("FLOOR", [0.0, -1.0, 0.5], 25.0),
            ("LEFT_SIDE_MIRROR", [-0.8, 0.2, 0.6], 12.0),
            ("DRIVER_WINDOW", [-1.0, 0.0, 0.2], 30.0),
            ("BLIND_SPOT_LEFT", [-0.6, -0.2, -0.8], 20.0),
            ("RIGHT_SIDE_MIRROR", [0.8, 0.2, 0.6], 12.0),
            ("PASSENGER", [1.0, 0.0, 0.5], 25.0),
        ]


class ColorConstants:
    """UI 색상 정의 (BGR 형식)"""
    
    # 위험도별 색상
    SAFE = (0, 200, 0)
    LOW_RISK = (0, 255, 255)
    MEDIUM_RISK = (0, 165, 255)
    HIGH_RISK = (0, 100, 255)
    CRITICAL = (0, 0, 255)
    
    # 일반 UI 색상
    TEXT = (255, 255, 255)
    BACKGROUND = (0, 0, 0)
    BACKUP_MODE = (0, 165, 255)
    CALIBRATION = (255, 255, 0)
    
    # 감정 상태별 색상
    EMOTION_POSITIVE = (0, 255, 0)
    EMOTION_NEGATIVE = (0, 0, 255)
    EMOTION_NEUTRAL = (128, 128, 128)
    
    # 특수 상태 색상
    PREDICTION_WARNING = (255, 165, 0)
    DRIVER_IDENTIFIED = (255, 255, 0)
    MICROSLEEP_ALERT = (128, 0, 128)


class MathConstants:
    """수학적 계산 관련 상수"""
    
    # 각도 변환
    DEG_TO_RAD = np.pi / 180.0
    RAD_TO_DEG = 180.0 / np.pi
    
    # 정규화 관련
    HEAD_YAW_NORMALIZATION = 90.0  # ±90도 범위로 정규화
    HEAD_PITCH_NORMALIZATION = 60.0  # ±60도 범위로 정규화
    
    # 벡터 계산
    EPSILON = 1e-6  # 0으로 나누기 방지
    VECTOR_NORM_MIN = 1e-8  # 최소 벡터 크기
    
    # 필터링 및 평활화
    TEMPORAL_DECAY_RATE = 2.0  # 시간적 어텐션 가중치 감쇠율
    SIMILARITY_THRESHOLD = 0.7  # 운전자 식별 임계값
    
    # 통계적 계산
    PERCENTILE_EAR_THRESHOLD = 5  # EAR 임계값 계산용 백분위수
    OUTLIER_DETECTION_FACTOR = 2.0  # 이상치 감지 표준편차 배수


class SystemConstants:
    """시스템 운영 관련 상수"""
    
    # 타이밍 관련
    class Timing:
        EXPAND_ANALYSIS_FACE_LOST = 2.0  # 초
        EXPAND_ANALYSIS_POSE_LOST = 2.0  # 초
        EXPAND_ANALYSIS_HAND_OUT = 1.0  # 초
        NORMAL_STATE_DURATION = 5.0  # 정상 상태로 복귀하기 위한 최소 시간
        BUFFER_CLEANUP_INTERVAL = 2.0  # 초
    
    # 큐 및 버퍼 크기
    class BufferSizes:
        FRAME_BUFFER_MAX = 5
        RESULT_BUFFER_TIMEOUT = 2000  # ms
        STATE_HISTORY_MAX = 100
        INTERACTION_STATES_MAX = 20
    
    # 로그 관리
    class Logging:
        MAX_LOG_COUNT = 500
        MIN_CLEAR_INTERVAL = 300  # 초
        MEMORY_CLEANUP_INTERVAL = 300  # 초
    
    # 파일 시스템
    class FileSystem:
        CAPTURE_DIR = "captures"
        LOG_DIR = "logs"
        MODELS_DIR = "models"
        PROFILES_DIR = "profiles"
        PERFORMANCE_LOGS_DIR = "performance_logs"
        
        # 필수 모델 파일들
        REQUIRED_MODELS = [
            "face_landmarker.task",
            "pose_landmarker_full.task", 
            "hand_landmarker.task",
            "efficientdet_lite0.tflite"
        ]
    
    # 백업 시스템
    class Backup:
        QUALITY_FACE_FROM_POSE = 0.6
        QUALITY_POSE_FROM_FACE = 0.4
        QUALITY_HAND_FROM_POSE = 0.8


class MessageConstants:
    """사용자 메시지 상수"""
    
    # 상태 전환 메시지
    STATE_ALERTS = {
        "FATIGUE_HIGH": "FATIGUE DETECTED!",
        "DISTRACTION_DANGER": "DANGEROUS DISTRACTION!",
        "PHONE_USAGE": "PHONE USAGE!",
        "MULTIPLE_RISK": "MULTIPLE RISKS!",
        "MICROSLEEP": "MICROSLEEP DETECTED!",
        "EMOTIONAL_STRESS": "EMOTIONAL STRESS!",
        "PREDICTIVE_WARNING": "PREDICTIVE WARNING!",
    }
    
    # 시스템 메시지
    SYSTEM_MESSAGES = {
        "INITIALIZATION_COMPLETE": "🚀 고도화된 DMS 시스템 v18 (연구 결과 통합) 초기화 완료",
        "NEW_FEATURES": "📊 새로운 기능: 향상된 EAR, 감정 인식, 예측적 안전, 운전자 식별",
        "CONTROLS": "🎯 DMS v18 시스템 시작. 'q'를 눌러 종료, 스페이스바로 일시정지, 's'로 스크린샷 저장",
        "TERMINAL_CLEARED": "=== 터미널 로그 정리됨 (메모리 관리) ===",
        "S_CLASS_ACTIVATED": "🔬 S-Class 고급 분석 모드 활성화",
        "EVENT_SYSTEM_READY": "📡 이벤트 시스템 가동 - 실시간 모니터링 시작",
        "FUSION_ENGINE_READY": "🧠 고급 융합 엔진 준비 완료 - 멀티모달 분석 시작",
    }
    
    # 오류 메시지
    ERROR_MESSAGES = {
        "INPUT_SOURCE_FAILED": "입력 소스 열기 실패",
        "TASK_INIT_FAILED": "MediaPipe Task 초기화 실패",
        "MODEL_FILE_MISSING": "다음 모델 파일이 없어 프로그램을 시작할 수 없습니다",
        "PROFILE_LOAD_FAILED": "프로필 로드 실패",
        "PROFILE_SAVE_FAILED": "프로필 저장 실패",
    }


# 편의 함수들
def normalize_angle(angle: float, max_angle: float) -> float:
    """각도를 0-1 범위로 정규화"""
    return abs(angle) / max_angle

def create_zone_bounds(x1: float, y1: float, x2: float, y2: float) -> dict:
    """구역 경계 딕셔너리 생성"""
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

def is_point_in_zone(x: float, y: float, zone: dict) -> bool:
    """점이 구역 내에 있는지 확인"""
    return zone["x1"] <= x <= zone["x2"] and zone["y1"] <= y <= zone["y2"]

def calculate_distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """2D 거리 계산"""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def normalize_vector(vector: List[float]) -> np.ndarray:
    """벡터 정규화"""
    v = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > MathConstants.VECTOR_NORM_MIN else v


# === S-Class 고급 기능 상수들 ===

class SClassConstants:
    """S-Class 프로세서들의 고급 기능 관련 상수"""
    
    # rPPG (원격 광혈류측정) 관련
    class RPPG:
        # 생리학적 범위
        MIN_HEART_RATE_BPM = 50
        MAX_HEART_RATE_BPM = 180
        NORMAL_HEART_RATE_MIN = 60
        NORMAL_HEART_RATE_MAX = 100
        
        # 신호 처리
        SAMPLING_RATE = 30.0  # FPS와 동일
        WINDOW_SIZE_SECONDS = 10.0  # 10초 윈도우
        FILTER_LOW_CUT_HZ = 0.8  # 48 BPM
        FILTER_HIGH_CUT_HZ = 3.0  # 180 BPM
        
        # 품질 검증
        MIN_SNR_THRESHOLD = 2.0  # Signal-to-Noise Ratio
        MAX_HRV_STD_THRESHOLD = 200.0  # ms
        MIN_SIGNAL_QUALITY = 0.3
        
        # ROI 정의 (이마 영역)
        FOREHEAD_ROI_INDICES = [10, 151, 9, 8, 107, 55, 8]
    
    # 사케이드 (안구 도약 운동) 관련
    class Saccade:
        # 운동학적 임계값
        VELOCITY_THRESHOLD = 30.0  # deg/sec
        MIN_AMPLITUDE = 1.0  # degrees
        MAX_AMPLITUDE = 50.0  # degrees
        
        # 분석 파라미터
        MIN_SAMPLES_FOR_ANALYSIS = 10
        PEAK_DETECTION_DISTANCE = 3  # frames
        NORMAL_SACCADE_RATE = 3.0  # per second
        
        # 고정시선 (Fixation) 분석
        FIXATION_VELOCITY_THRESHOLD = 10.0  # deg/sec
        MAX_FIXATION_DISPERSION = 2.0  # degrees
        MIN_FIXATION_DURATION = 0.1  # seconds
    
    # 동공 역학 관련
    class Pupil:
        # 정상 범위
        NORMAL_DIAMETER_MIN_MM = 2.0
        NORMAL_DIAMETER_MAX_MM = 8.0
        
        # 인지 부하 지표
        COGNITIVE_LOAD_DILATION = 0.5  # mm increase
        STRESS_CONSTRICTION_RATE = 2.0  # mm/sec
        
        # 변화율 분석
        MAX_NORMAL_VARIABILITY = 0.3  # relative to mean
        RAPID_CHANGE_THRESHOLD = 1.0  # mm/sec
    
    # 생체역학 분석 관련
    class Biomechanics:
        # 척추 정렬
        NORMAL_CERVICAL_LORDOSIS = 20.0  # degrees
        FORWARD_HEAD_THRESHOLD = 15.0  # degrees
        SLOUCHING_THRESHOLD = 10.0  # degrees
        
        # 자세 흔들림
        NORMAL_SWAY_AMPLITUDE = 2.0  # cm
        FATIGUE_SWAY_THRESHOLD = 5.0  # cm
        SWAY_FREQUENCY_NORMAL = 0.5  # Hz
        
        # 근육 긴장도
        TENSION_SCALE_MIN = 0.0
        TENSION_SCALE_MAX = 1.0
        HIGH_TENSION_THRESHOLD = 0.7
    
    # 손 키네마틱스 관련
    class HandKinematics:
        # 떨림 분석
        TREMOR_FREQUENCY_MIN = 4.0  # Hz
        TREMOR_FREQUENCY_MAX = 12.0  # Hz
        PATHOLOGICAL_TREMOR_AMPLITUDE = 2.0  # mm
        
        # 그립 품질
        OPTIMAL_GRIP_FORCE = 0.6  # normalized
        GRIP_STABILITY_THRESHOLD = 0.1  # variation coefficient
        
        # 조향 기술
        SMOOTH_STEERING_SCORE_MIN = 0.7
        REACTION_TIME_NORMAL = 0.3  # seconds
        OVER_CORRECTION_THRESHOLD = 15.0  # degrees


class EventSystemConstants:
    """이벤트 시스템 관련 상수"""
    
    # 큐 크기 설정
    class QueueSizes:
        EMERGENCY_QUEUE = 100
        CRITICAL_QUEUE = 500
        HIGH_QUEUE = 1000
        MEDIUM_QUEUE = 3000
        LOW_QUEUE = 5000
        MAX_TOTAL_EVENTS = 10000
    
    # 타임아웃 설정 (초)
    class Timeouts:
        HANDLER_CRITICAL = 5.0
        HANDLER_NORMAL = 10.0
        EVENT_PROCESSING = 1.0
        QUEUE_WAIT = 1.0
    
    # 재시도 및 복구
    class Recovery:
        MAX_HANDLER_RETRIES = 3
        HANDLER_FAILURE_COOLDOWN = 30.0  # seconds
        DEAD_REFERENCE_CLEANUP_INTERVAL = 30.0  # seconds
        STATISTICS_REPORT_INTERVAL = 300.0  # seconds
    
    # 성능 임계값
    class Performance:
        MAX_PROCESSING_TIME_MS = 100.0
        QUEUE_CONGESTION_THRESHOLD = 1000
        HANDLER_FAILURE_RATE_THRESHOLD = 0.1  # 10%


class FusionEngineConstants:
    """멀티모달 융합 엔진 관련 상수"""
    
    # 어텐션 메커니즘
    class Attention:
        # 트랜스포머 스타일 파라미터
        DEFAULT_ATTENTION_HEADS = 4
        HIGH_PERFORMANCE_HEADS = 8
        RESEARCH_HEADS = 16
        
        # 어텐션 스케일링
        TEMPERATURE_SCALING = 0.1
        SOFTMAX_TEMPERATURE = 1.0
        
        # 가중치 정규화
        MIN_WEIGHT_THRESHOLD = 0.01
        MAX_WEIGHT_CLIPPING = 0.9
    
    # 인지 부하 모델링
    class CognitiveLoad:
        # 부하 수준 임계값
        MINIMAL_THRESHOLD = 0.2
        LIGHT_THRESHOLD = 0.4
        MODERATE_THRESHOLD = 0.6
        HIGH_THRESHOLD = 0.8
        
        # 멀티태스킹 계수
        TASK_INTERFERENCE_BASE = 0.2
        TASK_INTERFERENCE_EXPONENT = 1.5
        MAX_MULTITASKING_PENALTY = 0.5
        
        # 어텐션 용량
        DEFAULT_ATTENTION_CAPACITY = 1.0
        DEGRADED_CAPACITY_MULTIPLIER = 0.7
    
    # 시간적 모델링
    class Temporal:
        DEFAULT_WINDOW_SIZE = 30  # frames (1 second at 30fps)
        HIGH_PERFORMANCE_WINDOW = 60  # frames (2 seconds)
        RESEARCH_WINDOW = 120  # frames (4 seconds)
        
        # 가중치 감쇠
        EXPONENTIAL_DECAY_RATE = 0.1
        LINEAR_DECAY_RATE = 0.05
        
        # 상관관계 분석
        MIN_CORRELATION_STRENGTH = 0.5
        MAX_CORRELATION_BOOST = 0.3
    
    # 신뢰도 정량화
    class Confidence:
        # 융합 신뢰도 가중치
        CONSISTENCY_WEIGHT = 0.6
        MODALITY_RELIABILITY_WEIGHT = 0.4
        
        # 일관성 측정
        TEMPORAL_CONSISTENCY_WINDOW = 10
        MAX_ALLOWED_VARIANCE = 0.3
        
        # 불확실성 임계값
        LOW_UNCERTAINTY_THRESHOLD = 0.1
        HIGH_UNCERTAINTY_THRESHOLD = 0.3


class OrchestratorConstants:
    """분석 오케스트레이터 관련 상수"""
    
    # 파이프라인 모드별 설정
    class PipelineModes:
        # 건강도 임계값
        FULL_PARALLEL_THRESHOLD = 0.8
        SELECTIVE_PARALLEL_THRESHOLD = 0.6
        SEQUENTIAL_SAFE_THRESHOLD = 0.3
        
        # 타임아웃 설정 (초)
        FULL_PARALLEL_TIMEOUT = 0.15
        SELECTIVE_TIMEOUT = 0.20
        SEQUENTIAL_TIMEOUT = 0.30
        EMERGENCY_TIMEOUT = 0.10
    
    # 프로세서 성능 모니터링
    class ProcessorHealth:
        # 상태 전환 임계값
        HEALTHY_SUCCESS_RATE = 0.95
        DEGRADED_SUCCESS_RATE = 0.80
        FAILING_SUCCESS_RATE = 0.60
        
        # 연속 실패 임계값
        DEGRADED_FAILURE_COUNT = 1
        FAILED_FAILURE_COUNT = 3
        RECOVERY_SUCCESS_COUNT = 5
        
        # 성능 메트릭
        TARGET_PROCESSING_TIME = 0.1  # seconds
        WARNING_PROCESSING_TIME = 0.15
        CRITICAL_PROCESSING_TIME = 0.25
    
    # 적응형 관리
    class AdaptiveManagement:
        # 타임아웃 조정 범위
        MIN_TIMEOUT = 0.05  # seconds
        MAX_TIMEOUT = 0.30  # seconds
        TIMEOUT_ADJUSTMENT_FACTOR = 1.2
        
        # 부하 예측
        LOAD_PREDICTION_WINDOW = 10  # frames
        BASE_LOAD_FACTOR = 0.5
        COMPLEXITY_MULTIPLIER = 0.01
        
        # 성능 이력 관리
        MAX_PERFORMANCE_HISTORY = 1000
        PERFORMANCE_SMOOTHING_ALPHA = 0.1
    
    # 리소스 관리
    class ResourceManagement:
        # 메모리 임계값
        MEMORY_WARNING_MB = 500
        MEMORY_CRITICAL_MB = 800
        
        # CPU 사용률 임계값
        CPU_WARNING_PERCENT = 75
        CPU_CRITICAL_PERCENT = 90
        
        # 긴급 최적화 트리거
        EMERGENCY_OPTIMIZATION_THRESHOLD = 0.9


class FactorySystemConstants:
    """팩토리 시스템 관련 상수"""
    
    # 시스템 타입별 기본 설정
    class SystemTypes:
        # 저사양 시스템
        LOW_RESOURCE_MAX_FPS = 15
        LOW_RESOURCE_MAX_RESOLUTION = 480
        LOW_RESOURCE_MAX_MEMORY_MB = 200
        LOW_RESOURCE_TIMEOUT_MULTIPLIER = 0.8
        
        # 표준 시스템
        STANDARD_MAX_FPS = 30
        STANDARD_MAX_RESOLUTION = 720
        STANDARD_MAX_MEMORY_MB = 800
        STANDARD_TIMEOUT_MULTIPLIER = 1.0
        
        # 고성능 시스템
        HIGH_PERFORMANCE_MAX_FPS = 60
        HIGH_PERFORMANCE_MAX_RESOLUTION = 1080
        HIGH_PERFORMANCE_MAX_MEMORY_MB = 2000
        HIGH_PERFORMANCE_TIMEOUT_MULTIPLIER = 1.3
        
        # 연구용 시스템
        RESEARCH_MAX_FPS = 120
        RESEARCH_MAX_RESOLUTION = 2160  # 4K
        RESEARCH_MAX_MEMORY_MB = 8000
        RESEARCH_TIMEOUT_MULTIPLIER = 2.0
    
    # 프로세서 우선순위
    class ProcessorPriority:
        FACE_PRIORITY = 0.9  # 가장 중요
        OBJECT_PRIORITY = 0.8
        HAND_PRIORITY = 0.7
        POSE_PRIORITY = 0.6
        
        # 중요 프로세서 선별 임계값
        CRITICAL_PRIORITY_THRESHOLD = 0.7
    
    # 품질 설정
    class QualitySettings:
        # 신뢰도 임계값
        LOW_RESOURCE_CONFIDENCE = 0.6
        STANDARD_CONFIDENCE = 0.7
        HIGH_PERFORMANCE_CONFIDENCE = 0.8
        RESEARCH_CONFIDENCE = 0.9
        
        # 처리 품질 점수
        MINIMUM_QUALITY_SCORE = 0.3
        TARGET_QUALITY_SCORE = 0.8
        EXCELLENT_QUALITY_SCORE = 0.95


class MediaPipeExtendedConstants:
    """MediaPipe 확장 상수 (S-Class 기능용)"""
    
    # 얼굴 ROI 정의
    class FaceROIs:
        # rPPG를 위한 이마 영역
        FOREHEAD_ROI = [10, 151, 9, 8, 107, 55, 8, 9]
        
        # 좌우 뺨 영역 (보조 rPPG)
        LEFT_CHEEK_ROI = [116, 117, 118, 119, 120, 121]
        RIGHT_CHEEK_ROI = [345, 346, 347, 348, 349, 350]
        
        # 눈 영역 (사케이드 분석용)
        LEFT_EYE_DETAILED = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        RIGHT_EYE_DETAILED = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    # 동공 추적을 위한 홍채 중심점
    class EyeLandmarks:
        LEFT_IRIS_CENTER = 468
        RIGHT_IRIS_CENTER = 473
        
        # 눈꺼풀 랜드마크 (깜빡임 분석용)
        LEFT_UPPER_EYELID = [159, 158, 157, 173]
        LEFT_LOWER_EYELID = [144, 145, 153, 154]
        RIGHT_UPPER_EYELID = [386, 387, 388, 466]
        RIGHT_LOWER_EYELID = [374, 373, 380, 381]
    
    # 자세 분석을 위한 확장 랜드마크
    class PoseExtended:
        # 척추 정렬 분석용
        CERVICAL_SPINE_POINTS = [0, 11, 12]  # 목-어깨 라인
        THORACIC_SPINE_POINTS = [11, 12, 23, 24]  # 어깨-엉덩이 라인
        
        # 머리 자세 분석용
        HEAD_ORIENTATION_POINTS = [0, 7, 8]  # 코-귀 삼각형
        
        # 어깨 수평 분석용
        SHOULDER_LINE_POINTS = [11, 12]
    
    # 손 랜드마크 확장
    class HandExtended:
        # 손가락별 관절점
        THUMB_JOINTS = [1, 2, 3, 4]
        INDEX_FINGER_JOINTS = [5, 6, 7, 8]
        MIDDLE_FINGER_JOINTS = [9, 10, 11, 12]
        RING_FINGER_JOINTS = [13, 14, 15, 16]
        PINKY_JOINTS = [17, 18, 19, 20]
        
        # 그립 분석용 주요 점들
        GRIP_ANALYSIS_POINTS = [0, 4, 8, 12, 16, 20]  # 손목 + 손가락 끝
        
        # 떨림 분석용 안정성 점들
        STABILITY_POINTS = [0, 9]  # 손목과 중지 기저부


# === 성능 벤치마크 상수 ===

class PerformanceBenchmarks:
    """시스템 성능 벤치마크 및 목표값"""
    
    # 처리 시간 목표 (밀리초)
    class ProcessingTime:
        FACE_PROCESSOR_TARGET = 50
        POSE_PROCESSOR_TARGET = 40
        HAND_PROCESSOR_TARGET = 30
        OBJECT_PROCESSOR_TARGET = 35
        FUSION_ENGINE_TARGET = 20
        TOTAL_PIPELINE_TARGET = 150
    
    # 메모리 사용량 목표 (MB)
    class MemoryUsage:
        FACE_PROCESSOR_TARGET = 100
        POSE_PROCESSOR_TARGET = 80
        HAND_PROCESSOR_TARGET = 60
        OBJECT_PROCESSOR_TARGET = 70
        FUSION_ENGINE_TARGET = 50
        TOTAL_SYSTEM_TARGET = 500
    
    # 정확도 목표 (0.0 - 1.0)
    class AccuracyTargets:
        DROWSINESS_DETECTION = 0.85
        DISTRACTION_DETECTION = 0.80
        EMOTION_RECOGNITION = 0.75
        GAZE_CLASSIFICATION = 0.78
        DRIVER_IDENTIFICATION = 0.90
        OVERALL_SYSTEM_CONFIDENCE = 0.80
    
    # 성능 등급 임계값
    class PerformanceGrades:
        EXCELLENT_THRESHOLD = 0.90
        GOOD_THRESHOLD = 0.75
        ACCEPTABLE_THRESHOLD = 0.60
        POOR_THRESHOLD = 0.45