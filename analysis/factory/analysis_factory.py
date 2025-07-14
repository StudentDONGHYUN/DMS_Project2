import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass

from config.settings import get_config
from core.interfaces import IMetricsUpdater
from analysis.orchestrator.orchestrator_advanced import AnalysisOrchestrator
from analysis.fusion.fusion_engine_advanced import MultiModalFusionEngine

logger = logging.getLogger(__name__)


class AnalysisSystemType(Enum):
    """
    분석 시스템 타입 정의
    
    각 타입은 서로 다른 사용 사례와 성능 요구사항을 가집니다.
    마치 스마트폰의 성능 모드 선택과 같습니다.
    """
    LOW_RESOURCE = "low_resource"        # 절전모드 - 최소 자원으로 기본 기능
    STANDARD = "standard"                # 균형모드 - 성능과 효율의 균형
    HIGH_PERFORMANCE = "high_performance" # 고성능모드 - 최대 정확도와 기능
    RESEARCH = "research"                # 연구모드 - 모든 실험 기능 활성화


@dataclass
class SystemConfiguration:
    """
    시스템 구성 정보를 담는 데이터 클래스
    
    비유: 자동차의 사양서
    - 엔진 종류, 기어박스, 옵션 등이 명시되어 있듯이
    - 여기에는 어떤 프로세서를 사용할지, 어떤 설정값을 적용할지 정의
    """
    system_type: AnalysisSystemType
    enabled_processors: List[str]
    fusion_config: Dict[str, Any]
    performance_settings: Dict[str, Any]
    timeout_settings: Dict[str, float]
    quality_settings: Dict[str, Any]
    experimental_features: List[str]
    resource_limits: Dict[str, Any]
    
    def __post_init__(self):
        """구성 검증 및 기본값 설정"""
        self._validate_configuration()
        self._apply_compatibility_fixes()
    
    def _validate_configuration(self):
        """구성의 유효성을 검증합니다"""
        # 필수 프로세서 확인
        required_processors = ['face', 'pose', 'hand', 'object']
        for processor in required_processors:
            if processor not in self.enabled_processors:
                logger.warning(f"필수 프로세서 누락: {processor} - 기본 설정으로 추가")
                self.enabled_processors.append(processor)
        
        # 타임아웃 설정 검증
        for processor, timeout in self.timeout_settings.items():
            if timeout <= 0 or timeout > 1.0:  # 1초 초과는 비현실적
                logger.warning(f"비정상적인 타임아웃 설정: {processor}={timeout} - 기본값으로 조정")
                self.timeout_settings[processor] = 0.1  # 100ms 기본값
    
    def _apply_compatibility_fixes(self):
        """시스템 타입에 따른 호환성 수정"""
        if self.system_type == AnalysisSystemType.LOW_RESOURCE:
            # 저사양 시스템에서는 실험적 기능 비활성화
            self.experimental_features = []
            
            # 품질 설정을 낮춤
            self.quality_settings['max_resolution'] = min(
                self.quality_settings.get('max_resolution', 640), 480
            )
        
        elif self.system_type == AnalysisSystemType.HIGH_PERFORMANCE:
            # 고성능 시스템에서는 모든 기능 활성화
            all_experimental = [
                'rppg_heart_rate', 'saccade_analysis', 'pupil_dynamics',
                'predictive_analysis', 'behavior_modeling', 'stress_detection'
            ]
            self.experimental_features.extend(all_experimental)
            self.experimental_features = list(set(self.experimental_features))  # 중복 제거


class IAnalysisSystemFactory(ABC):
    """분석 시스템 팩토리 인터페이스"""
    
    @abstractmethod
    def create_system(
        self, 
        metrics_updater: IMetricsUpdater,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> AnalysisOrchestrator:
        """분석 시스템을 생성합니다"""
        pass
    
    @abstractmethod
    def get_system_configuration(self) -> SystemConfiguration:
        """시스템 구성 정보를 반환합니다"""
        pass
    
    @abstractmethod
    def get_supported_features(self) -> List[str]:
        """지원하는 기능 목록을 반환합니다"""
        pass


class LowResourceSystemFactory(IAnalysisSystemFactory):
    """
    저사양 시스템용 팩토리
    
    비유: 경차 제조라인
    - 연비 최우선: CPU/메모리 사용량 최소화
    - 기본 안전장치: 핵심 안전 기능만 제공
    - 단순한 구조: 복잡한 알고리즘 제외
    - 빠른 응답: 실시간성 우선
    
    사용 사례: 
    - 임베디드 시스템 (라즈베리파이 등)
    - 저사양 차량용 컴퓨터
    - 배터리 수명이 중요한 환경
    """
    
    def get_system_configuration(self) -> SystemConfiguration:
        return SystemConfiguration(
            system_type=AnalysisSystemType.LOW_RESOURCE,
            enabled_processors=['face', 'pose'],  # 기본 프로세서만
            fusion_config={
                'fusion_method': 'simple_weighted',  # 단순 가중평균
                'enable_attention': False,           # 어텐션 메커니즘 비활성화
                'temporal_smoothing': False          # 시간적 평활화 비활성화
            },
            performance_settings={
                'max_fps': 15,                       # 낮은 프레임레이트
                'parallel_processing': False,       # 순차 처리로 메모리 절약
                'cache_size': 10,                   # 작은 캐시
                'optimization_level': 'speed'        # 속도 우선 최적화
            },
            timeout_settings={
                'face': 0.08,    # 80ms 타임아웃 (빠른 응답)
                'pose': 0.06,    # 60ms 타임아웃
                'total': 0.15    # 전체 150ms 제한
            },
            quality_settings={
                'max_resolution': 480,               # 낮은 해상도
                'detection_confidence': 0.6,        # 낮은 신뢰도 임계값
                'landmark_smoothing': False         # 랜드마크 평활화 비활성화
            },
            experimental_features=[],               # 실험 기능 없음
            resource_limits={
                'max_memory_mb': 200,               # 200MB 메모리 제한
                'max_cpu_percent': 60,              # CPU 60% 제한
                'max_gpu_memory_mb': 100            # GPU 메모리 100MB 제한
            }
        )
    
    def create_system(
        self, 
        metrics_updater: IMetricsUpdater,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> AnalysisOrchestrator:
        """저사양 최적화된 분석 시스템 생성"""
        
        config = self.get_system_configuration()
        
        # 사용자 정의 설정 적용
        if custom_config:
            self._apply_custom_config(config, custom_config)
        
        logger.info("저사양 최적화 DMS 시스템 생성 중...")
        
        # 단순한 융합 엔진 생성 (내부 설정 사용)
        fusion_engine = MultiModalFusionEngine()
        
        # 최소 기능 오케스트레이터 생성
        from analysis.processors.face_processor import FaceDataProcessor
        from analysis.processors.pose_processor import PoseDataProcessor
        from analysis.processors.hand_processor import HandDataProcessor
        from analysis.processors.object_processor import ObjectDataProcessor
        from analysis.drowsiness import EnhancedDrowsinessDetector
        from analysis.emotion import EmotionRecognitionSystem
        from analysis.gaze import GazeZoneClassifier
        from analysis.identity import DriverIdentificationSystem

        drowsiness_detector = EnhancedDrowsinessDetector()
        emotion_recognizer = EmotionRecognitionSystem()
        gaze_classifier = GazeZoneClassifier()
        driver_identifier = DriverIdentificationSystem()

        face_processor = FaceDataProcessor(
            metrics_updater=metrics_updater,
            drowsiness_detector=drowsiness_detector,
            emotion_recognizer=emotion_recognizer,
            gaze_classifier=gaze_classifier,
            driver_identifier=driver_identifier,
        )

        orchestrator = AnalysisOrchestrator(
            metrics_updater=metrics_updater,
            face_processor=face_processor,
            pose_processor=PoseDataProcessor(metrics_updater),
            hand_processor=HandDataProcessor(metrics_updater),
            object_processor=ObjectDataProcessor(metrics_updater),
            fusion_engine=fusion_engine,
        )
        
        logger.info("저사양 DMS 시스템 생성 완료 - 절전 모드 활성화")
        return orchestrator
    
    def get_supported_features(self) -> List[str]:
        return [
            'basic_drowsiness_detection',
            'basic_distraction_detection', 
            'head_pose_tracking',
            'simple_gaze_estimation',
            'basic_emotion_recognition'
        ]
    
    def _apply_custom_config(self, config: SystemConfiguration, custom_config: Dict[str, Any]):
        """사용자 정의 설정을 안전하게 적용"""
        # 저사양 시스템에서는 성능에 해로운 설정은 무시
        safe_overrides = ['max_fps', 'detection_confidence', 'cache_size']
        
        for key, value in custom_config.items():
            if key in safe_overrides:
                if key == 'max_fps' and value <= 20:  # FPS는 20 이하로 제한
                    config.performance_settings[key] = value
                elif key == 'detection_confidence' and 0.4 <= value <= 0.8:
                    config.quality_settings[key] = value
                elif key == 'cache_size' and value <= 20:
                    config.performance_settings[key] = value
                else:
                    logger.warning(f"저사양 시스템에서 안전하지 않은 설정 무시: {key}={value}")


class StandardSystemFactory(IAnalysisSystemFactory):
    """
    표준 시스템용 팩토리
    
    비유: 중형차 제조라인
    - 균형잡힌 성능: 효율성과 기능성의 적절한 조화
    - 완전한 안전장치: 모든 기본 안전 기능 + 일부 고급 기능
    - 적응형 구조: 상황에 따라 성능 조절
    - 안정적 운영: 검증된 알고리즘 위주
    
    사용 사례:
    - 일반적인 승용차
    - 상용 운전자 모니터링 시스템
    - 플릿 관리 시스템
    """
    
    def get_system_configuration(self) -> SystemConfiguration:
        return SystemConfiguration(
            system_type=AnalysisSystemType.STANDARD,
            enabled_processors=['face', 'pose', 'hand', 'object'],  # 모든 기본 프로세서
            fusion_config={
                'fusion_method': 'attention_weighted',  # 어텐션 기반 융합
                'enable_attention': True,               # 어텐션 메커니즘 활성화
                'temporal_smoothing': True,             # 시간적 평활화 활성화
                'confidence_weighting': True            # 신뢰도 기반 가중치
            },
            performance_settings={
                'max_fps': 30,                          # 표준 프레임레이트
                'parallel_processing': True,            # 병렬 처리 활성화
                'cache_size': 50,                      # 적당한 캐시 크기
                'optimization_level': 'balanced',       # 균형잡힌 최적화
                'adaptive_quality': True               # 적응형 품질 조절
            },
            timeout_settings={
                'face': 0.12,    # 120ms 타임아웃
                'pose': 0.08,    # 80ms 타임아웃  
                'hand': 0.10,    # 100ms 타임아웃
                'object': 0.06,  # 60ms 타임아웃
                'total': 0.25    # 전체 250ms 제한
            },
            quality_settings={
                'max_resolution': 720,               # HD 해상도
                'detection_confidence': 0.7,        # 표준 신뢰도
                'landmark_smoothing': True,         # 랜드마크 평활화
                'temporal_consistency': True        # 시간적 일관성 유지
            },
            experimental_features=[                 # 일부 실험 기능
                'basic_rppg', 'simple_saccade_analysis', 'attention_modeling'
            ],
            resource_limits={
                'max_memory_mb': 800,               # 800MB 메모리 한계
                'max_cpu_percent': 75,              # CPU 75% 제한
                'max_gpu_memory_mb': 500            # GPU 메모리 500MB 제한
            }
        )
    
    def create_system(
        self, 
        metrics_updater: IMetricsUpdater,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> AnalysisOrchestrator:
        """표준 성능의 분석 시스템 생성"""
        
        config = self.get_system_configuration()
        
        if custom_config:
            self._apply_custom_config(config, custom_config)
        
        logger.info("표준 성능 DMS 시스템 생성 중...")
        
        # 어텐션 기반 융합 엔진 (내부 설정 사용)
        fusion_engine = MultiModalFusionEngine()
        
        # 적응형 오케스트레이터
        from analysis.processors.face_processor import FaceDataProcessor
        from analysis.processors.pose_processor import PoseDataProcessor
        from analysis.processors.hand_processor import HandDataProcessor
        from analysis.processors.object_processor import ObjectDataProcessor
        from analysis.drowsiness import EnhancedDrowsinessDetector
        from analysis.emotion import EmotionRecognitionSystem
        from analysis.gaze import GazeZoneClassifier
        from analysis.identity import DriverIdentificationSystem

        drowsiness_detector = EnhancedDrowsinessDetector()
        emotion_recognizer = EmotionRecognitionSystem()
        gaze_classifier = GazeZoneClassifier()
        driver_identifier = DriverIdentificationSystem()

        face_processor = FaceDataProcessor(
            metrics_updater=metrics_updater,
            drowsiness_detector=drowsiness_detector,
            emotion_recognizer=emotion_recognizer,
            gaze_classifier=gaze_classifier,
            driver_identifier=driver_identifier,
        )

        orchestrator = AnalysisOrchestrator(
            metrics_updater=metrics_updater,
            face_processor=face_processor,
            pose_processor=PoseDataProcessor(metrics_updater),
            hand_processor=HandDataProcessor(metrics_updater),
            object_processor=ObjectDataProcessor(metrics_updater),
            fusion_engine=fusion_engine,
        )
        
        logger.info("표준 DMS 시스템 생성 완료 - 균형 모드 활성화")
        return orchestrator
    
    def get_supported_features(self) -> List[str]:
        return [
            'advanced_drowsiness_detection',
            'multi_level_distraction_detection',
            'precise_head_pose_tracking', 
            'gaze_zone_classification',
            'emotion_recognition',
            'basic_behavior_prediction',
            'driver_identification',
            'attention_modeling',
            'basic_vital_signs_monitoring'
        ]
    
    def _apply_custom_config(self, config: SystemConfiguration, custom_config: Dict[str, Any]):
        """표준 시스템용 설정 적용 - 더 유연함"""
        # 표준 시스템에서는 대부분의 설정 변경 허용
        for key, value in custom_config.items():
            if key in config.performance_settings:
                config.performance_settings[key] = value
            elif key in config.quality_settings:
                config.quality_settings[key] = value
            elif key in config.timeout_settings:
                config.timeout_settings[key] = value
            else:
                logger.debug(f"알 수 없는 설정 무시: {key}={value}")


class HighPerformanceSystemFactory(IAnalysisSystemFactory):
    """
    고성능 시스템용 팩토리
    
    비유: 럭셔리 스포츠카 제조라인
    - 최고 성능: 최신 기술과 알고리즘 총동원
    - 완벽한 안전: 모든 가능한 안전 기능 탑재
    - 지능형 구조: AI 기반 적응형 시스템
    - 무제한 기능: 실험적 기능까지 포함
    
    사용 사례:
    - 자율주행 차량
    - 연구용 차량
    - 프리미엄 안전 시스템
    - 실시간 테스트 환경
    """
    
    def get_system_configuration(self) -> SystemConfiguration:
        return SystemConfiguration(
            system_type=AnalysisSystemType.HIGH_PERFORMANCE,
            enabled_processors=['face', 'pose', 'hand', 'object', 'environment'],  # 모든 프로세서
            fusion_config={
                'fusion_method': 'neural_attention',    # 신경망 기반 융합
                'enable_attention': True,
                'temporal_smoothing': True,
                'confidence_weighting': True,
                'multimodal_correlation': True,         # 모달리티간 상관관계 분석
                'uncertainty_quantification': True     # 불확실성 정량화
            },
            performance_settings={
                'max_fps': 60,                          # 고프레임레이트
                'parallel_processing': True,
                'cache_size': 200,                     # 대용량 캐시
                'optimization_level': 'accuracy',      # 정확도 우선 최적화
                'adaptive_quality': True,
                'predictive_resource_management': True  # 예측적 리소스 관리
            },
            timeout_settings={
                'face': 0.15,    # 더 긴 타임아웃으로 정확도 향상
                'pose': 0.12,
                'hand': 0.15,
                'object': 0.10,
                'environment': 0.08,
                'total': 0.40    # 전체 400ms 허용
            },
            quality_settings={
                'max_resolution': 1080,              # Full HD 해상도
                'detection_confidence': 0.8,        # 높은 신뢰도 요구
                'landmark_smoothing': True,
                'temporal_consistency': True,
                'sub_pixel_accuracy': True,          # 서브픽셀 정확도
                'multi_scale_analysis': True         # 다중 스케일 분석
            },
            experimental_features=[                  # 모든 실험 기능 활성화
                'advanced_rppg', 'full_saccade_analysis', 'pupil_dynamics',
                'stress_detection', 'cognitive_load_estimation', 
                'behavior_prediction', 'personality_profiling',
                'micro_expression_analysis', 'breathing_pattern_analysis'
            ],
            resource_limits={
                'max_memory_mb': 2000,              # 2GB 메모리
                'max_cpu_percent': 90,              # CPU 90% 사용 가능
                'max_gpu_memory_mb': 1500           # GPU 메모리 1.5GB
            }
        )
    
    def create_system(
        self, 
        metrics_updater: IMetricsUpdater,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> AnalysisOrchestrator:
        """최고 성능의 분석 시스템 생성"""
        
        config = self.get_system_configuration()
        
        if custom_config:
            self._apply_custom_config(config, custom_config)
        
        logger.info("고성능 DMS 시스템 생성 중... 모든 고급 기능 활성화")
        
        # 최고급 신경망 융합 엔진 (내부 설정 사용)
        fusion_engine = MultiModalFusionEngine()
        
        # 최고 성능 오케스트레이터
        from analysis.processors.face_processor import FaceDataProcessor
        from analysis.processors.pose_processor import PoseDataProcessor
        from analysis.processors.hand_processor import HandDataProcessor
        from analysis.processors.object_processor import ObjectDataProcessor
        from analysis.drowsiness import EnhancedDrowsinessDetector
        from analysis.emotion import EmotionRecognitionSystem
        from analysis.gaze import GazeZoneClassifier
        from analysis.identity import DriverIdentificationSystem

        drowsiness_detector = EnhancedDrowsinessDetector()
        emotion_recognizer = EmotionRecognitionSystem()
        gaze_classifier = GazeZoneClassifier()
        driver_identifier = DriverIdentificationSystem()

        face_processor = FaceDataProcessor(
            metrics_updater=metrics_updater,
            drowsiness_detector=drowsiness_detector,
            emotion_recognizer=emotion_recognizer,
            gaze_classifier=gaze_classifier,
            driver_identifier=driver_identifier,
        )

        orchestrator = AnalysisOrchestrator(
            metrics_updater=metrics_updater,
            face_processor=face_processor,
            pose_processor=PoseDataProcessor(metrics_updater),
            hand_processor=HandDataProcessor(metrics_updater),
            object_processor=ObjectDataProcessor(metrics_updater),
            fusion_engine=fusion_engine,
        )
        
        logger.info("고성능 DMS 시스템 생성 완료 - 최대 성능 모드 활성화")
        return orchestrator
    
    def get_supported_features(self) -> List[str]:
        return [
            'advanced_drowsiness_detection',
            'multi_modal_distraction_detection',
            'precision_head_pose_tracking',
            'advanced_gaze_analysis',
            'comprehensive_emotion_recognition',
            'behavioral_pattern_analysis',
            'predictive_risk_assessment',
            'driver_profiling',
            'advanced_attention_modeling',
            'vital_signs_monitoring',
            'stress_level_detection',
            'cognitive_load_estimation',
            'micro_expression_analysis',
            'breathing_pattern_analysis'
            'personality_assessment',
            'fatigue_prediction',
            'distraction_prediction'
        ]
    
    def _apply_custom_config(self, config: SystemConfiguration, custom_config: Dict[str, Any]):
        """고성능 시스템용 설정 - 모든 설정 허용"""
        # 고성능 시스템에서는 거의 모든 설정 변경 허용
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif key in config.performance_settings:
                config.performance_settings[key] = value
            elif key in config.quality_settings:
                config.quality_settings[key] = value
            elif key in config.timeout_settings:
                config.timeout_settings[key] = value
            elif key in config.fusion_config:
                config.fusion_config[key] = value
            else:
                logger.info(f"고성능 시스템에 사용자 정의 설정 적용: {key}={value}")


class ResearchSystemFactory(IAnalysisSystemFactory):
    """
    연구용 시스템 팩토리
    
    비유: 포뮬러 1 경주차 제조라인
    - 실험적 기술: 아직 검증되지 않은 최신 알고리즘
    - 극한 성능: 성능 제약 없음
    - 데이터 수집: 모든 가능한 데이터 수집 및 분석
    - 유연한 구조: 실시간 알고리즘 변경 가능
    
    사용 사례:
    - 대학 연구실
    - 자동차 회사 R&D 부서
    - 알고리즘 개발 및 검증
    - 데이터 수집 및 분석
    """
    
    def get_system_configuration(self) -> SystemConfiguration:
        return SystemConfiguration(
            system_type=AnalysisSystemType.RESEARCH,
            enabled_processors=['face', 'pose', 'hand', 'object', 'environment', 'audio'],
            fusion_config={
                'fusion_method': 'experimental_ensemble',  # 실험적 앙상블 방법
                'enable_attention': True,
                'temporal_smoothing': True,
                'confidence_weighting': True,
                'multimodal_correlation': True,
                'uncertainty_quantification': True,
                'causal_inference': True,               # 인과관계 추론
                'bayesian_fusion': True                # 베이지안 융합
            },
            performance_settings={
                'max_fps': 120,                         # 극고속 프레임레이트
                'parallel_processing': True,
                'cache_size': 1000,                    # 초대용량 캐시
                'optimization_level': 'experimental',   # 실험적 최적화
                'adaptive_quality': True,
                'predictive_resource_management': True,
                'data_logging': True,                  # 모든 데이터 로깅
                'algorithm_profiling': True            # 알고리즘 프로파일링
            },
            timeout_settings={
                'face': 0.30,    # 매우 긴 타임아웃으로 극한 정확도
                'pose': 0.25,
                'hand': 0.30,
                'object': 0.20,
                'environment': 0.15,
                'audio': 0.10,
                'total': 1.0     # 전체 1초 허용
            },
            quality_settings={
                'max_resolution': 2160,              # 4K 해상도
                'detection_confidence': 0.9,        # 매우 높은 신뢰도
                'landmark_smoothing': True,
                'temporal_consistency': True,
                'sub_pixel_accuracy': True,
                'multi_scale_analysis': True,
                'ensemble_validation': True          # 앙상블 검증
            },
            experimental_features=[                  # 모든 실험 기능 + 추가 연구 기능
                'advanced_rppg', 'full_saccade_analysis', 'pupil_dynamics',
                'stress_detection', 'cognitive_load_estimation', 
                'behavior_prediction', 'personality_profiling',
                'micro_expression_analysis', 'breathing_pattern_analysis',
                'eeg_signal_processing', 'galvanic_skin_response',
                'voice_stress_analysis', 'thermal_imaging_analysis',
                'gait_analysis', 'social_signal_processing'
            ],
            resource_limits={
                'max_memory_mb': 8000,              # 8GB 메모리
                'max_cpu_percent': 95,              # CPU 95% 사용
                'max_gpu_memory_mb': 4000           # GPU 메모리 4GB
            }
        )
    
    def create_system(
        self, 
        metrics_updater: IMetricsUpdater,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> AnalysisOrchestrator:
        """연구용 최고사양 분석 시스템 생성"""
        
        config = self.get_system_configuration()
        
        if custom_config:
            self._apply_custom_config(config, custom_config)
        
        logger.info("연구용 DMS 시스템 생성 중... 모든 실험적 기능 활성화")
        
        # 최첨단 실험적 융합 엔진 (내부 설정 사용)
        fusion_engine = MultiModalFusionEngine()
        
        # 연구용 특수 오케스트레이터
        from analysis.processors.face_processor import FaceDataProcessor
        from analysis.processors.pose_processor import PoseDataProcessor
        from analysis.processors.hand_processor import HandDataProcessor
        from analysis.processors.object_processor import ObjectDataProcessor
        from analysis.drowsiness import EnhancedDrowsinessDetector
        from analysis.emotion import EmotionRecognitionSystem
        from analysis.gaze import GazeZoneClassifier
        from analysis.identity import DriverIdentificationSystem

        drowsiness_detector = EnhancedDrowsinessDetector()
        emotion_recognizer = EmotionRecognitionSystem()
        gaze_classifier = GazeZoneClassifier()
        driver_identifier = DriverIdentificationSystem()

        face_processor = FaceDataProcessor(
            metrics_updater=metrics_updater,
            drowsiness_detector=drowsiness_detector,
            emotion_recognizer=emotion_recognizer,
            gaze_classifier=gaze_classifier,
            driver_identifier=driver_identifier,
        )

        orchestrator = AnalysisOrchestrator(
            metrics_updater=metrics_updater,
            face_processor=face_processor,
            pose_processor=PoseDataProcessor(metrics_updater),
            hand_processor=HandDataProcessor(metrics_updater),
            object_processor=ObjectDataProcessor(metrics_updater),
            fusion_engine=fusion_engine,
        )
        
        logger.info("연구용 DMS 시스템 생성 완료 - 실험 모드 활성화")
        return orchestrator
    
    def get_supported_features(self) -> List[str]:
        return [
            # 모든 기존 기능 + 실험적 기능들
            'advanced_drowsiness_detection', 'multi_modal_distraction_detection',
            'precision_head_pose_tracking', 'advanced_gaze_analysis',
            'comprehensive_emotion_recognition', 'behavioral_pattern_analysis',
            'predictive_risk_assessment',
            'driver_profiling',
            'advanced_attention_modeling',
            'vital_signs_monitoring',
            'stress_level_detection',
            'cognitive_load_estimation',
            'micro_expression_analysis',
            'breathing_pattern_analysis',
            'personality_assessment',
            'fatigue_prediction',
            'distraction_prediction',
            # 연구 전용 기능들
            'eeg_signal_analysis', 'galvanic_skin_response_analysis',
            'voice_stress_analysis', 'thermal_imaging_analysis',
            'gait_pattern_analysis', 'social_signal_processing',
            'multimodal_biometric_fusion', 'real_time_algorithm_switching',
            'continuous_learning', 'transfer_learning',
            'few_shot_adaptation', 'meta_learning'
        ]
    
    def _apply_custom_config(self, config: SystemConfiguration, custom_config: Dict[str, Any]):
        """연구용 시스템 - 모든 설정 허용 및 동적 변경 지원"""
        # 연구 시스템에서는 모든 설정 변경 허용, 심지어 런타임 변경도 가능
        for key, value in custom_config.items():
            try:
                if hasattr(config, key):
                    setattr(config, key, value)
                    logger.info(f"연구용 시스템 설정 변경: {key}={value}")
                else:
                    # 새로운 실험적 설정도 허용
                    if not hasattr(config, 'experimental_settings'):
                        config.experimental_settings = {}
                    config.experimental_settings[key] = value
                    logger.info(f"실험적 설정 추가: {key}={value}")
            except Exception as e:
                logger.warning(f"설정 적용 실패: {key}={value}, 오류: {e}")


# === 팩토리 레지스트리 및 생성 함수 ===

# 팩토리 타입 매핑
_FACTORY_REGISTRY: Dict[AnalysisSystemType, Type[IAnalysisSystemFactory]] = {
    AnalysisSystemType.LOW_RESOURCE: LowResourceSystemFactory,
    AnalysisSystemType.STANDARD: StandardSystemFactory,
    AnalysisSystemType.HIGH_PERFORMANCE: HighPerformanceSystemFactory,
    AnalysisSystemType.RESEARCH: ResearchSystemFactory,
}


def create_analysis_system(
    system_type: AnalysisSystemType,
    metrics_updater: IMetricsUpdater,
    custom_config: Optional[Dict[str, Any]] = None
) -> AnalysisOrchestrator:
    """
    지정된 타입의 분석 시스템을 생성합니다.
    
    이 함수는 팩토리 패턴의 핵심입니다. 마치 자동차 딜러에서
    고객이 "경제형/표준형/프리미엄/연구용 모델을 원합니다"라고 하면
    적절한 제조라인에서 해당 사양의 차량을 만들어주는 것과 같습니다.
    
    Args:
        system_type: 생성할 시스템 타입
        metrics_updater: 메트릭 업데이터 인스턴스
        custom_config: 사용자 정의 설정 (선택사항)
    
    Returns:
        생성된 분석 오케스트레이터 인스턴스
    
    Raises:
        ValueError: 지원하지 않는 시스템 타입인 경우
    """
    
    # 팩토리 클래스 조회
    factory_class = _FACTORY_REGISTRY.get(system_type)
    if factory_class is None:
        available_types = list(_FACTORY_REGISTRY.keys())
        raise ValueError(f"지원하지 않는 시스템 타입: {system_type}. 사용 가능한 타입: {available_types}")
    
    # 팩토리 인스턴스 생성 및 시스템 생성
    factory = factory_class()
    
    logger.info(f"DMS 분석 시스템 생성 시작: {system_type.value}")
    logger.info(f"지원 기능: {', '.join(factory.get_supported_features()[:5])}...")  # 처음 5개만 표시
    
    try:
        system = factory.create_system(metrics_updater, custom_config)
        logger.info(f"DMS 분석 시스템 생성 완료: {system_type.value}")
        return system
        
    except Exception as e:
        logger.error(f"DMS 시스템 생성 실패: {e}")
        
        # 실패시 폴백으로 가장 단순한 시스템 생성 시도
        if system_type != AnalysisSystemType.LOW_RESOURCE:
            logger.info("폴백으로 저사양 시스템 생성 시도...")
            fallback_factory = LowResourceSystemFactory()
            return fallback_factory.create_system(metrics_updater, None)
        else:
            raise  # 저사양 시스템도 실패하면 예외 발생


def get_system_info(system_type: AnalysisSystemType) -> Dict[str, Any]:
    """
    지정된 시스템 타입의 상세 정보를 반환합니다.
    
    비유: 자동차 카탈로그
    각 모델의 사양, 성능, 옵션, 가격 등이 자세히 나와있듯이
    여기서는 각 시스템의 기능, 성능, 리소스 요구사항 등을 제공합니다.
    """
    
    factory_class = _FACTORY_REGISTRY.get(system_type)
    if factory_class is None:
        return {"error": f"지원하지 않는 시스템 타입: {system_type}"}
    
    factory = factory_class()
    config = factory.get_system_configuration()
    features = factory.get_supported_features()
    
    return {
        "system_type": system_type.value,
        "description": _get_system_description(system_type),
        "configuration": {
            "enabled_processors": config.enabled_processors,
            "max_fps": config.performance_settings.get('max_fps', 'N/A'),
            "max_resolution": config.quality_settings.get('max_resolution', 'N/A'),
            "parallel_processing": config.performance_settings.get('parallel_processing', False),
            "experimental_features_count": len(config.experimental_features)
        },
        "resource_requirements": config.resource_limits,
        "supported_features": features,
        "use_cases": _get_use_cases(system_type),
        "performance_profile": _get_performance_profile(system_type)
    }


def _get_system_description(system_type: AnalysisSystemType) -> str:
    """시스템 타입별 설명"""
    descriptions = {
        AnalysisSystemType.LOW_RESOURCE: "저사양 하드웨어를 위한 효율성 중심 시스템. 기본적인 안전 기능을 저전력으로 제공합니다.",
        AnalysisSystemType.STANDARD: "일반적인 상용 환경을 위한 균형잡힌 시스템. 성능과 효율성을 적절히 조화시켰습니다.",
        AnalysisSystemType.HIGH_PERFORMANCE: "최고 성능을 요구하는 환경을 위한 프리미엄 시스템. 모든 고급 기능과 최대 정확도를 제공합니다.",
        AnalysisSystemType.RESEARCH: "연구개발을 위한 실험적 시스템. 최신 알고리즘과 데이터 수집 기능을 포함합니다."
    }
    return descriptions.get(system_type, "설명이 없습니다.")


def _get_use_cases(system_type: AnalysisSystemType) -> List[str]:
    """시스템 타입별 사용 사례"""
    use_cases = {
        AnalysisSystemType.LOW_RESOURCE: [
            "임베디드 시스템 (라즈베리파이, NVIDIA Jetson Nano)",
            "저사양 차량용 컴퓨터", "배터리 구동 장치", "IoT 디바이스"
        ],
        AnalysisSystemType.STANDARD: [
            "일반 승용차 안전 시스템", "상용 운전자 모니터링",
            "플릿 관리 시스템", "운전 교육 시스템"
        ],
        AnalysisSystemType.HIGH_PERFORMANCE: [
            "자율주행 차량", "프리미엄 안전 시스템",
            "실시간 테스트 환경", "고급 운전자 지원 시스템"
        ],
        AnalysisSystemType.RESEARCH: [
            "대학 연구실", "자동차 회사 R&D 부서",
            "알고리즘 개발 및 검증", "데이터 수집 및 분석"
        ]
    }
    return use_cases.get(system_type, [])


def _get_performance_profile(system_type: AnalysisSystemType) -> Dict[str, str]:
    """시스템 타입별 성능 프로필"""
    profiles = {
        AnalysisSystemType.LOW_RESOURCE: {
            "처리속도": "빠름 (단순 알고리즘)", "정확도": "기본", 
            "메모리사용": "낮음", "전력소모": "낮음"
        },
        AnalysisSystemType.STANDARD: {
            "처리속도": "보통", "정확도": "높음",
            "메모리사용": "보통", "전력소모": "보통"
        },
        AnalysisSystemType.HIGH_PERFORMANCE: {
            "처리속도": "빠름 (병렬처리)", "정확도": "매우 높음",
            "메모리사용": "높음", "전력소모": "높음"
        },
        AnalysisSystemType.RESEARCH: {
            "처리속도": "가변 (실험적)", "정확도": "최고",
            "메모리사용": "매우 높음", "전력소모": "매우 높음"
        }
    }
    return profiles.get(system_type, {})


def list_available_systems() -> List[Dict[str, Any]]:
    """사용 가능한 모든 시스템 타입의 정보를 반환"""
    return [get_system_info(system_type) for system_type in AnalysisSystemType]


# === 사용 예시 ===
if __name__ == "__main__":
    # 사용 가능한 시스템들 출력
    print("=== 사용 가능한 DMS 시스템들 ===")
    for system_info in list_available_systems():
        print(f"\n{system_info['system_type'].upper()}:")
        print(f"  설명: {system_info['description']}")
        print(f"  사용사례: {', '.join(system_info['use_cases'][:2])}...")
        print(f"  주요기능: {len(system_info['supported_features'])}개 기능 지원")