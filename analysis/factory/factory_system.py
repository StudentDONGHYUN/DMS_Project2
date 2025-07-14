"""
Analysis Factory System
각 분석 모듈의 생성과 의존성 주입을 체계적으로 관리하는 팩토리 시스템
- [Factory] 추상 팩토리 패턴으로 다양한 구성의 분석 시스템 생성 지원
- [DI] 의존성 주입(Dependency Injection)을 통한 느슨한 결합
- [Config] 설정 기반 모듈 선택 및 구성
- [Lifecycle] 모듈 생명주기 관리 (초기화, 해제, 리소스 정리)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, List
from enum import Enum
import logging

from config.settings import get_config
from core.interfaces import (
    IMetricsUpdater, IFaceDataProcessor, IPoseDataProcessor,
    IHandDataProcessor, IObjectDataProcessor, IMultiModalAnalyzer,
    IDrowsinessDetector, IEmotionRecognizer, IGazeClassifier, IDriverIdentifier
)

# 실제 구현 클래스들 import (예시)
from analysis.processors.face_processor_s_class import FaceDataProcessor
from analysis.processors.pose_processor_s_class import PoseDataProcessor
from analysis.processors.hand_processor_s_class import HandDataProcessor
from analysis.processors.object_processor_s_class import ObjectDataProcessor
from analysis.fusion.multimodal_fusion import MultiModalFusionEngine
from analysis.orchestrator.analysis_orchestrator import AnalysisOrchestrator

logger = logging.getLogger(__name__)


class AnalysisSystemType(Enum):
    """분석 시스템 타입"""
    STANDARD = "standard"           # 표준 구성
    HIGH_PERFORMANCE = "high_perf"  # 고성능 구성
    LOW_RESOURCE = "low_resource"   # 저사양 구성
    RESEARCH = "research"           # 연구용 구성 (모든 기능 활성화)
    MINIMAL = "minimal"             # 최소 구성 (핵심 기능만)


class ProcessorVariant(Enum):
    """프로세서 변형"""
    BASIC = "basic"
    ADVANCED = "advanced"
    S_CLASS = "s_class"


class IAnalysisSystemFactory(ABC):
    """
    분석 시스템 팩토리 인터페이스
    
    이 인터페이스는 마치 건축 설계사가 다양한 타입의 건물을 설계하는 것처럼,
    각기 다른 요구사항에 맞는 분석 시스템을 생성하는 청사진을 제공합니다.
    """
    
    @abstractmethod
    def create_face_processor(self, metrics_updater: IMetricsUpdater) -> IFaceDataProcessor:
        """얼굴 분석 프로세서 생성"""
        pass
    
    @abstractmethod
    def create_pose_processor(self, metrics_updater: IMetricsUpdater) -> IPoseDataProcessor:
        """자세 분석 프로세서 생성"""
        pass
    
    @abstractmethod
    def create_hand_processor(self, metrics_updater: IMetricsUpdater) -> IHandDataProcessor:
        """손 분석 프로세서 생성"""
        pass
    
    @abstractmethod
    def create_object_processor(self, metrics_updater: IMetricsUpdater) -> IObjectDataProcessor:
        """객체 분석 프로세서 생성"""
        pass
    
    @abstractmethod
    def create_fusion_engine(self) -> IMultiModalAnalyzer:
        """융합 엔진 생성"""
        pass
    
    @abstractmethod
    def create_orchestrator(
        self, 
        metrics_updater: IMetricsUpdater,
        face_processor: IFaceDataProcessor,
        pose_processor: IPoseDataProcessor,
        hand_processor: IHandDataProcessor,
        object_processor: IObjectDataProcessor,
        fusion_engine: IMultiModalAnalyzer
    ) -> AnalysisOrchestrator:
        """분석 오케스트레이터 생성"""
        pass


class StandardAnalysisFactory(IAnalysisSystemFactory):
    """
    표준 분석 시스템 팩토리
    
    일반적인 사용 환경에 최적화된 균형잡힌 구성의 분석 시스템을 생성합니다.
    마치 일반 승용차처럼 성능과 효율성의 적절한 균형을 제공합니다.
    """
    
    def __init__(self, processor_variant: ProcessorVariant = ProcessorVariant.ADVANCED):
        self.processor_variant = processor_variant
        self.config = get_config()
        
        logger.info(f"StandardAnalysisFactory 초기화 - 변형: {processor_variant.value}")
    
    def create_face_processor(self, metrics_updater: IMetricsUpdater) -> IFaceDataProcessor:
        """표준 얼굴 분석 프로세서 생성"""
        
        # 보조 분석 모듈들 생성 (의존성 주입)
        drowsiness_detector = self._create_drowsiness_detector()
        emotion_recognizer = self._create_emotion_recognizer()
        gaze_classifier = self._create_gaze_classifier()
        driver_identifier = self._create_driver_identifier()
        
        # 메인 프로세서 생성 및 의존성 주입
        face_processor = FaceDataProcessor(
            metrics_updater=metrics_updater,
            drowsiness_detector=drowsiness_detector,
            emotion_recognizer=emotion_recognizer,
            gaze_classifier=gaze_classifier,
            driver_identifier=driver_identifier
        )
        
        logger.info("표준 얼굴 분석 프로세서 생성 완료")
        return face_processor
    
    def create_pose_processor(self, metrics_updater: IMetricsUpdater) -> IPoseDataProcessor:
        """표준 자세 분석 프로세서 생성"""
        return PoseDataProcessor(metrics_updater=metrics_updater)
    
    def create_hand_processor(self, metrics_updater: IMetricsUpdater) -> IHandDataProcessor:
        """표준 손 분석 프로세서 생성"""
        return HandDataProcessor(metrics_updater=metrics_updater)
    
    def create_object_processor(self, metrics_updater: IMetricsUpdater) -> IObjectDataProcessor:
        """표준 객체 분석 프로세서 생성"""
        return ObjectDataProcessor(metrics_updater=metrics_updater)
    
    def create_fusion_engine(self) -> IMultiModalAnalyzer:
        """표준 융합 엔진 생성"""
        return MultiModalFusionEngine()
    
    def create_orchestrator(
        self,
        metrics_updater: IMetricsUpdater,
        face_processor: IFaceDataProcessor,
        pose_processor: IPoseDataProcessor,
        hand_processor: IHandDataProcessor,
        object_processor: IObjectDataProcessor,
        fusion_engine: IMultiModalAnalyzer
    ) -> AnalysisOrchestrator:
        """표준 분석 오케스트레이터 생성"""
        return AnalysisOrchestrator(
            metrics_updater=metrics_updater,
            face_processor=face_processor,
            pose_processor=pose_processor,
            hand_processor=hand_processor,
            object_processor=object_processor,
            fusion_engine=fusion_engine
        )
    
    def _create_drowsiness_detector(self) -> IDrowsinessDetector:
        """졸음 감지기 생성 (설정에 따라 다른 구현체 선택)"""
        # 실제 구현에서는 설정 파일의 drowsiness_detector_type에 따라 
        # 다른 구현체를 선택할 수 있습니다
        detector_type = self.config.analysis.drowsiness_detector_type
        
        if detector_type == "enhanced":
            from analysis.drowsiness import EnhancedDrowsinessDetector
            return EnhancedDrowsinessDetector()
        else:
            from analysis.drowsiness import StandardDrowsinessDetector
            return StandardDrowsinessDetector()
    
    def _create_emotion_recognizer(self) -> IEmotionRecognizer:
        """감정 인식기 생성"""
        from analysis.emotion import EmotionRecognitionSystem
        return EmotionRecognitionSystem()
    
    def _create_gaze_classifier(self) -> IGazeClassifier:
        """시선 분류기 생성"""
        from analysis.gaze import GazeTrackingSystem
        return GazeTrackingSystem()
    
    def _create_driver_identifier(self) -> IDriverIdentifier:
        """운전자 식별기 생성"""
        from analysis.identity import DriverIdentificationSystem
        return DriverIdentificationSystem()


class HighPerformanceAnalysisFactory(IAnalysisSystemFactory):
    """
    고성능 분석 시스템 팩토리
    
    최고 성능을 요구하는 환경을 위한 분석 시스템을 생성합니다.
    마치 슈퍼카처럼 최고의 성능을 제공하지만 더 많은 리소스를 사용합니다.
    """
    
    def __init__(self):
        self.config = get_config()
        logger.info("HighPerformanceAnalysisFactory 초기화 - 최고 성능 모드")
    
    def create_face_processor(self, metrics_updater: IMetricsUpdater) -> IFaceDataProcessor:
        """고성능 얼굴 분석 프로세서 생성 (S-Class 변형)"""
        
        # 고성능 보조 모듈들 생성
        drowsiness_detector = self._create_enhanced_drowsiness_detector()
        emotion_recognizer = self._create_advanced_emotion_recognizer()
        gaze_classifier = self._create_precision_gaze_classifier()
        driver_identifier = self._create_ai_driver_identifier()
        
        # S-Class 프로세서 생성
        face_processor = FaceDataProcessor(
            metrics_updater=metrics_updater,
            drowsiness_detector=drowsiness_detector,
            emotion_recognizer=emotion_recognizer,
            gaze_classifier=gaze_classifier,
            driver_identifier=driver_identifier
        )
        
        # 고성능 모드 설정
        face_processor.enable_advanced_features()
        face_processor.set_precision_mode(True)
        
        logger.info("고성능 얼굴 분석 프로세서 생성 완료")
        return face_processor
    
    def create_pose_processor(self, metrics_updater: IMetricsUpdater) -> IPoseDataProcessor:
        """고성능 자세 분석 프로세서 생성"""
        pose_processor = PoseDataProcessor(metrics_updater=metrics_updater)
        pose_processor.enable_3d_analysis()  # 3D 분석 활성화
        pose_processor.enable_biomechanics_analysis()  # 생체역학 분석 활성화
        return pose_processor
    
    def create_hand_processor(self, metrics_updater: IMetricsUpdater) -> IHandDataProcessor:
        """고성능 손 분석 프로세서 생성"""
        hand_processor = HandDataProcessor(metrics_updater=metrics_updater)
        hand_processor.enable_kinematics_analysis()  # 운동학 분석 활성화
        hand_processor.enable_gesture_prediction()   # 제스처 예측 활성화
        return hand_processor
    
    def create_object_processor(self, metrics_updater: IMetricsUpdater) -> IObjectDataProcessor:
        """고성능 객체 분석 프로세서 생성"""
        object_processor = ObjectDataProcessor(metrics_updater=metrics_updater)
        object_processor.enable_behavior_prediction()  # 행동 예측 활성화
        object_processor.enable_attention_heatmap()    # 어텐션 히트맵 활성화
        return object_processor
    
    def create_fusion_engine(self) -> IMultiModalAnalyzer:
        """고성능 융합 엔진 생성"""
        fusion_engine = MultiModalFusionEngine()
        fusion_engine.enable_attention_mechanism()     # 어텐션 메커니즘 활성화
        fusion_engine.enable_uncertainty_quantification()  # 불확실성 정량화 활성화
        return fusion_engine
    
    def create_orchestrator(
        self,
        metrics_updater: IMetricsUpdater,
        face_processor: IFaceDataProcessor,
        pose_processor: IPoseDataProcessor,
        hand_processor: IHandDataProcessor,
        object_processor: IObjectDataProcessor,
        fusion_engine: IMultiModalAnalyzer
    ) -> AnalysisOrchestrator:
        """고성능 분석 오케스트레이터 생성"""
        orchestrator = AnalysisOrchestrator(
            metrics_updater=metrics_updater,
            face_processor=face_processor,
            pose_processor=pose_processor,
            hand_processor=hand_processor,
            object_processor=object_processor,
            fusion_engine=fusion_engine
        )
        
        # 고성능 모드 설정
        orchestrator.enable_adaptive_pipeline()        # 적응형 파이프라인 활성화
        orchestrator.enable_predictive_scheduling()    # 예측적 스케줄링 활성화
        orchestrator.set_max_parallel_tasks(8)         # 최대 병렬 작업 수 증가
        
        return orchestrator
    
    def _create_enhanced_drowsiness_detector(self) -> IDrowsinessDetector:
        """향상된 졸음 감지기 생성"""
        from analysis.drowsiness import AIEnhancedDrowsinessDetector
        return AIEnhancedDrowsinessDetector()
    
    def _create_advanced_emotion_recognizer(self) -> IEmotionRecognizer:
        """고급 감정 인식기 생성"""
        from analysis.emotion import DeepEmotionRecognizer
        return DeepEmotionRecognizer()
    
    def _create_precision_gaze_classifier(self) -> IGazeClassifier:
        """정밀 시선 분류기 생성"""
        from analysis.gaze import PrecisionGazeTracker
        return PrecisionGazeTracker()
    
    def _create_ai_driver_identifier(self) -> IDriverIdentifier:
        """AI 운전자 식별기 생성"""
        from analysis.identity import AIDriverIdentifier
        return AIDriverIdentifier()


class LowResourceAnalysisFactory(IAnalysisSystemFactory):
    """
    저사양 분석 시스템 팩토리
    
    제한된 리소스 환경에서 최적화된 분석 시스템을 생성합니다.
    마치 경차처럼 효율성을 최우선으로 하면서도 필수 기능은 모두 제공합니다.
    """
    
    def __init__(self):
        self.config = get_config()
        logger.info("LowResourceAnalysisFactory 초기화 - 저사양 최적화 모드")
    
    def create_face_processor(self, metrics_updater: IMetricsUpdater) -> IFaceDataProcessor:
        """경량화된 얼굴 분석 프로세서 생성"""
        
        # 경량화된 보조 모듈들
        drowsiness_detector = self._create_lightweight_drowsiness_detector()
        emotion_recognizer = self._create_basic_emotion_recognizer()
        gaze_classifier = self._create_simple_gaze_classifier()
        driver_identifier = self._create_fast_driver_identifier()
        
        face_processor = FaceDataProcessor(
            metrics_updater=metrics_updater,
            drowsiness_detector=drowsiness_detector,
            emotion_recognizer=emotion_recognizer,
            gaze_classifier=gaze_classifier,
            driver_identifier=driver_identifier
        )
        
        # 경량화 모드 설정
        face_processor.enable_lightweight_mode()
        face_processor.set_processing_interval(2)  # 2프레임마다 처리
        
        return face_processor
    
    def create_pose_processor(self, metrics_updater: IMetricsUpdater) -> IPoseDataProcessor:
        """경량화된 자세 분석 프로세서 생성"""
        pose_processor = PoseDataProcessor(metrics_updater=metrics_updater)
        pose_processor.enable_2d_only_mode()    # 2D 분석만 사용
        pose_processor.reduce_landmark_points()  # 랜드마크 포인트 수 감소
        return pose_processor
    
    def create_hand_processor(self, metrics_updater: IMetricsUpdater) -> IHandDataProcessor:
        """경량화된 손 분석 프로세서 생성"""
        hand_processor = HandDataProcessor(metrics_updater=metrics_updater)
        hand_processor.enable_basic_tracking_only()  # 기본 추적만 활성화
        return hand_processor
    
    def create_object_processor(self, metrics_updater: IMetricsUpdater) -> IObjectDataProcessor:
        """경량화된 객체 분석 프로세서 생성"""
        object_processor = ObjectDataProcessor(metrics_updater=metrics_updater)
        object_processor.limit_object_types(['cell_phone', 'cup'])  # 핵심 객체만 추적
        return object_processor
    
    def create_fusion_engine(self) -> IMultiModalAnalyzer:
        """경량화된 융합 엔진 생성"""
        fusion_engine = MultiModalFusionEngine()
        fusion_engine.enable_simple_fusion_mode()  # 단순 융합 모드
        return fusion_engine
    
    def create_orchestrator(
        self,
        metrics_updater: IMetricsUpdater,
        face_processor: IFaceDataProcessor,
        pose_processor: IPoseDataProcessor,
        hand_processor: IHandDataProcessor,
        object_processor: IObjectDataProcessor,
        fusion_engine: IMultiModalAnalyzer
    ) -> AnalysisOrchestrator:
        """경량화된 분석 오케스트레이터 생성"""
        orchestrator = AnalysisOrchestrator(
            metrics_updater=metrics_updater,
            face_processor=face_processor,
            pose_processor=pose_processor,
            hand_processor=hand_processor,
            object_processor=object_processor,
            fusion_engine=fusion_engine
        )
        
        # 경량화 설정
        orchestrator.set_max_parallel_tasks(2)      # 병렬 작업 수 제한
        orchestrator.enable_sequential_fallback()   # 순차 실행 우선
        orchestrator.set_processing_fps_limit(15)   # FPS 제한
        
        return orchestrator
    
    def _create_lightweight_drowsiness_detector(self) -> IDrowsinessDetector:
        """경량화된 졸음 감지기 생성"""
        from analysis.drowsiness import LightweightDrowsinessDetector
        return LightweightDrowsinessDetector()
    
    def _create_basic_emotion_recognizer(self) -> IEmotionRecognizer:
        """기본 감정 인식기 생성"""
        from analysis.emotion import BasicEmotionRecognizer
        return BasicEmotionRecognizer()
    
    def _create_simple_gaze_classifier(self) -> IGazeClassifier:
        """단순 시선 분류기 생성"""
        from analysis.gaze import SimpleGazeClassifier
        return SimpleGazeClassifier()
    
    def _create_fast_driver_identifier(self) -> IDriverIdentifier:
        """빠른 운전자 식별기 생성"""
        from analysis.identity import FastDriverIdentifier
        return FastDriverIdentifier()


class AnalysisSystemBuilder:
    """
    분석 시스템 빌더
    
    이 클래스는 마치 숙련된 시스템 엔지니어처럼 요구사항에 맞는 
    최적의 분석 시스템을 구성하고 생성하는 역할을 합니다.
    """
    
    def __init__(self):
        self.config = get_config()
        self.factory_registry = {
            AnalysisSystemType.STANDARD: StandardAnalysisFactory,
            AnalysisSystemType.HIGH_PERFORMANCE: HighPerformanceAnalysisFactory,
            AnalysisSystemType.LOW_RESOURCE: LowResourceAnalysisFactory,
            # 필요시 추가 팩토리들을 등록할 수 있습니다
        }
        
        logger.info("AnalysisSystemBuilder 초기화 완료")
    
    def build_analysis_system(
        self, 
        system_type: AnalysisSystemType,
        metrics_updater: IMetricsUpdater,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> AnalysisOrchestrator:
        """
        분석 시스템 빌드
        
        요구사항에 맞는 완전한 분석 시스템을 구성하고 반환합니다.
        마치 자동차 공장에서 주문에 맞는 차량을 조립하는 과정과 같습니다.
        """
        
        logger.info(f"{system_type.value} 타입의 분석 시스템 빌드 시작")
        
        # 1. 적절한 팩토리 선택
        factory_class = self.factory_registry.get(system_type)
        if not factory_class:
            raise ValueError(f"지원하지 않는 시스템 타입: {system_type}")
        
        factory = factory_class()
        
        # 2. 각 구성 요소들을 순서대로 생성 (의존성 고려)
        try:
            # 개별 프로세서들 생성
            face_processor = factory.create_face_processor(metrics_updater)
            pose_processor = factory.create_pose_processor(metrics_updater)
            hand_processor = factory.create_hand_processor(metrics_updater)
            object_processor = factory.create_object_processor(metrics_updater)
            
            # 융합 엔진 생성
            fusion_engine = factory.create_fusion_engine()
            
            # 오케스트레이터 생성 (모든 구성요소 주입)
            orchestrator = factory.create_orchestrator(
                metrics_updater=metrics_updater,
                face_processor=face_processor,
                pose_processor=pose_processor,
                hand_processor=hand_processor,
                object_processor=object_processor,
                fusion_engine=fusion_engine
            )
            
            # 3. 사용자 정의 설정 적용
            if custom_config:
                self._apply_custom_configuration(orchestrator, custom_config)
            
            # 4. 시스템 검증
            self._validate_system_integrity(orchestrator)
            
            logger.info(f"{system_type.value} 분석 시스템 빌드 완료")
            return orchestrator
            
        except Exception as e:
            logger.error(f"분석 시스템 빌드 중 오류 발생: {e}")
            raise
    
    def register_custom_factory(
        self, 
        system_type: AnalysisSystemType, 
        factory_class: Type[IAnalysisSystemFactory]
    ):
        """사용자 정의 팩토리 등록"""
        self.factory_registry[system_type] = factory_class
        logger.info(f"사용자 정의 팩토리 등록: {system_type.value}")
    
    def _apply_custom_configuration(
        self, 
        orchestrator: AnalysisOrchestrator, 
        custom_config: Dict[str, Any]
    ):
        """사용자 정의 설정 적용"""
        
        # 성능 관련 설정
        if 'max_fps' in custom_config:
            orchestrator.set_max_fps(custom_config['max_fps'])
        
        if 'timeout_settings' in custom_config:
            orchestrator.update_timeouts(custom_config['timeout_settings'])
        
        # 기능 활성화/비활성화
        if 'disabled_features' in custom_config:
            for feature in custom_config['disabled_features']:
                orchestrator.disable_feature(feature)
        
        logger.info("사용자 정의 설정 적용 완료")
    
    def _validate_system_integrity(self, orchestrator: AnalysisOrchestrator):
        """시스템 무결성 검증"""
        
        # 모든 프로세서가 올바르게 주입되었는지 확인
        if not all([
            orchestrator.processors.get('face'),
            orchestrator.processors.get('pose'), 
            orchestrator.processors.get('hand'),
            orchestrator.processors.get('object'),
            orchestrator.fusion_engine
        ]):
            raise RuntimeError("일부 필수 구성요소가 누락되었습니다")
        
        # 기본 설정 값들이 유효한지 확인
        if any(timeout <= 0 for timeout in orchestrator.adaptive_timeouts.values()):
            raise RuntimeError("잘못된 타임아웃 설정값이 발견되었습니다")
        
        logger.info("시스템 무결성 검증 완료")


# 편의를 위한 전역 빌더 인스턴스
system_builder = AnalysisSystemBuilder()


def create_analysis_system(
    system_type: AnalysisSystemType,
    metrics_updater: IMetricsUpdater,
    custom_config: Optional[Dict[str, Any]] = None
) -> AnalysisOrchestrator:
    """
    분석 시스템 생성 편의 함수
    
    사용 예시:
    >>> from systems.factory import create_analysis_system, AnalysisSystemType
    >>> from systems.metrics import MetricsManager
    >>> 
    >>> metrics = MetricsManager()
    >>> system = create_analysis_system(AnalysisSystemType.HIGH_PERFORMANCE, metrics)
    """
    return system_builder.build_analysis_system(system_type, metrics_updater, custom_config)