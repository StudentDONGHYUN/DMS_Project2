"""
Legacy System Integration Adapter
기존 EnhancedAnalysisEngine과 새로운 MetricsManager 시스템 간의 브리지 역할

이 모듈은 두 개의 서로 다른 아키텍처를 연결하는 어댑터입니다.
마치 구형 전자제품과 신형 전자제품을 연결하는 변환기와 같은 역할을 합니다.

주요 기능:
1. 기존 엔진의 메트릭을 새로운 MetricsManager로 전달
2. 이벤트 시스템 통합 (기존 직접 호출 -> 새로운 이벤트 버스)
3. 상태 관리 시스템 연동
4. 점진적 마이그레이션 지원

설계 원칙:
- 기존 코드 변경 최소화
- 새로운 시스템의 장점 활용
- 안전한 점진적 전환
- 성능 영향 최소화
"""

import logging
import time
from typing import Dict, Any, Optional
from dataclasses import asdict

from core.interfaces import IMetricsUpdater
from core.definitions import AdvancedMetrics, AnalysisEvent
from systems.metrics_manager import MetricsManager
from events.event_bus import get_event_bus, publish_safety_event, EventType, EventPriority

logger = logging.getLogger(__name__)


class LegacySystemAdapter:
    """
    기존 시스템과 새로운 시스템 간의 어댑터
    
    이 클래스는 마치 번역가와 같은 역할을 합니다.
    기존 시스템이 사용하는 "언어"를 새로운 시스템이 이해할 수 있는 
    "언어"로 번역해주고, 그 반대도 수행합니다.
    
    비유: 국제회의의 동시통역사
    - 한국어로 말하는 사람의 내용을 영어로 번역
    - 영어로 말하는 사람의 내용을 한국어로 번역
    - 양쪽 모두가 자연스럽게 소통할 수 있도록 도움
    """
    
    def __init__(self, metrics_manager: MetricsManager, enable_event_bridge: bool = True):
        """
        어댑터 초기화
        
        Args:
            metrics_manager: 새로운 메트릭 관리자
            enable_event_bridge: 이벤트 브리지 활성화 여부
        """
        self.metrics_manager = metrics_manager
        self.enable_event_bridge = enable_event_bridge
        
        # 마지막 동기화 시간 추적
        self.last_sync_time = time.time()
        self.sync_interval = 0.1  # 100ms 간격으로 동기화
        
        # 이벤트 매핑 테이블 (기존 이벤트 -> 새로운 이벤트)
        self.event_mapping = self._create_event_mapping()
        
        # 상태 추적 (중복 이벤트 방지)
        self.last_events = {}
        self.event_debounce_time = 1.0  # 1초 디바운스
        
        logger.info("기존 시스템 어댑터 초기화 완료")
    
    def _create_event_mapping(self) -> Dict[AnalysisEvent, EventType]:
        """기존 이벤트와 새로운 이벤트 타입 간의 매핑 테이블 생성"""
        return {
            AnalysisEvent.FATIGUE_ACCUMULATION: EventType.DROWSINESS_DETECTED,
            AnalysisEvent.ATTENTION_DECLINE: EventType.DISTRACTION_DETECTED,
            AnalysisEvent.EMOTION_STRESS_DETECTED: EventType.STRESS_DETECTED,
            AnalysisEvent.DISTRACTION_OBJECT_DETECTED: EventType.DISTRACTION_DETECTED,
            AnalysisEvent.MICROSLEEP_PREDICTED: EventType.CRITICAL_DROWSINESS,
            AnalysisEvent.PHONE_USAGE_CONFIRMED: EventType.PHONE_USAGE_DETECTED,
            AnalysisEvent.PREDICTIVE_RISK_HIGH: EventType.RISK_LEVEL_CHANGED,
            AnalysisEvent.NORMAL_BEHAVIOR: EventType.NORMAL_STATE_RESTORED
        }
    
    def sync_metrics_from_legacy(self, legacy_metrics: AdvancedMetrics) -> None:
        """
        기존 시스템의 메트릭을 새로운 MetricsManager로 동기화
        
        이 메서드는 마치 두 개의 서로 다른 회계 시스템 간에
        재무 정보를 동기화하는 것과 같습니다.
        
        Args:
            legacy_metrics: 기존 시스템에서 생성된 메트릭
        """
        current_time = time.time()
        
        # 너무 자주 동기화하지 않도록 제한
        if current_time - self.last_sync_time < self.sync_interval:
            return
        
        try:
            # 졸음 관련 메트릭 동기화
            self._sync_drowsiness_metrics(legacy_metrics)
            
            # 감정 관련 메트릭 동기화
            self._sync_emotion_metrics(legacy_metrics)
            
            # 시선 관련 메트릭 동기화
            self._sync_gaze_metrics(legacy_metrics)
            
            # 주의산만 관련 메트릭 동기화
            self._sync_distraction_metrics(legacy_metrics)
            
            # 예측 관련 메트릭 동기화
            self._sync_prediction_metrics(legacy_metrics)
            
            # 고급 메트릭 동기화 (새로운 기능)
            self._sync_advanced_metrics(legacy_metrics)
            
            self.last_sync_time = current_time
            
            logger.debug("기존 시스템 메트릭 동기화 완료")
            
        except Exception as e:
            logger.error(f"메트릭 동기화 중 오류: {e}")
    
    def _sync_drowsiness_metrics(self, metrics: AdvancedMetrics) -> None:
        """졸음 관련 메트릭 동기화"""
        drowsiness_data = {
            'fatigue_score': metrics.fatigue_risk_score,
            'confidence': metrics.drowsiness_confidence,
            'ear_value': metrics.enhanced_ear,
            'perclos': metrics.perclos,
            'temporal_attention_score': metrics.temporal_attention_score,
            'microsleep_detected': False  # 기존 시스템에서는 직접 제공하지 않음
        }
        
        self.metrics_manager.update_drowsiness_metrics(drowsiness_data)
    
    def _sync_emotion_metrics(self, metrics: AdvancedMetrics) -> None:
        """감정 관련 메트릭 동기화"""
        emotion_data = {
            'dominant_emotion': metrics.emotion_state.value if metrics.emotion_state else 'neutral',
            'arousal_level': metrics.arousal_level,
            'stress_level': 1.0 if metrics.emotion_state and 'stress' in metrics.emotion_state.value.lower() else 0.0,
            'confidence': metrics.emotion_confidence
        }
        
        self.metrics_manager.update_emotion_metrics(emotion_data)
    
    def _sync_gaze_metrics(self, metrics: AdvancedMetrics) -> None:
        """시선 관련 메트릭 동기화"""
        gaze_data = {
            'primary_zone': metrics.current_gaze_zone.value if metrics.current_gaze_zone else 'FRONT',
            'attention_focus': metrics.attention_focus_score,
            'gaze_stability': 1.0 - (abs(metrics.head_yaw) + abs(metrics.head_pitch)) / 180.0,  # 간단한 추정
            'off_road_duration': metrics.gaze_zone_duration if metrics.current_gaze_zone and 'FRONT' not in metrics.current_gaze_zone.value else 0.0
        }
        
        self.metrics_manager.update_gaze_metrics(gaze_data)
    
    def _sync_distraction_metrics(self, metrics: AdvancedMetrics) -> None:
        """주의산만 관련 메트릭 동기화"""
        distraction_data = {
            'distraction_risk_score': metrics.distraction_risk_score,
            'phone_detected': metrics.phone_detected,
            'hand_stability_score': 1.0,  # 기존 시스템에서는 직접 제공하지 않음, 기본값 사용
            'left_hand_in_safe_zone': True,  # 기본값
            'right_hand_in_safe_zone': True  # 기본값
        }
        
        self.metrics_manager.update_distraction_metrics(distraction_data)
    
    def _sync_prediction_metrics(self, metrics: AdvancedMetrics) -> None:
        """예측 관련 메트릭 동기화"""
        prediction_data = {
            'risk_score': metrics.predictive_risk_score,
            'risk_factors': [],  # 기존 시스템에서는 상세 정보 제공하지 않음
            'confidence': 0.7,   # 기본 신뢰도
            'time_horizon': 30.0  # 30초 예측 윈도우
        }
        
        self.metrics_manager.update_prediction_metrics(prediction_data)
    
    def _sync_advanced_metrics(self, metrics: AdvancedMetrics) -> None:
        """고급 메트릭 동기화 (새로운 기능들)"""
        # 기존 시스템에서 제공하지 않는 고급 메트릭들을 기본값으로 설정
        
        # 사케이드 메트릭 (기본값)
        saccade_data = {
            'frequency': 0.0,
            'amplitude': 0.0,
            'fixation_stability': 1.0
        }
        self.metrics_manager.update_saccade_metrics(saccade_data)
        
        # 동공 메트릭 (기본값)
        pupil_data = {
            'diameter': 0.0,
            'light_response': 1.0
        }
        self.metrics_manager.update_pupil_metrics(pupil_data)
        
        # rPPG 메트릭 (기본값)
        rppg_data = {
            'heart_rate': 0.0,
            'hrv_score': 0.0,
            'signal_quality': 0.0
        }
        self.metrics_manager.update_rppg_metrics(rppg_data)
        
        # 인지 부하 메트릭 (기존 메트릭에서 추정)
        cognitive_load = (metrics.fatigue_risk_score + metrics.distraction_risk_score) / 2.0
        cognitive_data = {
            'cognitive_load': cognitive_load,
            'mental_workload': cognitive_load
        }
        self.metrics_manager.update_cognitive_load_metrics(cognitive_data)
    
    async def bridge_event_to_new_system(self, legacy_event: AnalysisEvent, event_data: Dict[str, Any] = None) -> None:
        """
        기존 시스템의 이벤트를 새로운 이벤트 시스템으로 브리지
        
        이 메서드는 마치 두 개의 서로 다른 라디오 주파수 간에
        신호를 중계하는 중계소와 같은 역할을 합니다.
        
        Args:
            legacy_event: 기존 시스템에서 발생한 이벤트
            event_data: 이벤트와 함께 전달할 데이터
        """
        if not self.enable_event_bridge:
            return
        
        # 이벤트 매핑 확인
        new_event_type = self.event_mapping.get(legacy_event)
        if not new_event_type:
            logger.debug(f"매핑되지 않은 기존 이벤트 무시: {legacy_event}")
            return
        
        # 중복 이벤트 방지 (디바운싱)
        current_time = time.time()
        last_event_time = self.last_events.get(legacy_event, 0)
        
        if current_time - last_event_time < self.event_debounce_time:
            logger.debug(f"중복 이벤트 방지: {legacy_event}")
            return
        
        self.last_events[legacy_event] = current_time
        
        try:
            # 이벤트 데이터 준비
            if event_data is None:
                event_data = {}
            
            # 이벤트 발행
            if new_event_type in [EventType.CRITICAL_DROWSINESS, EventType.DROWSINESS_DETECTED, 
                                 EventType.DISTRACTION_DETECTED, EventType.STRESS_DETECTED]:
                # 안전 관련 이벤트는 높은 우선순위로 발행
                priority = EventPriority.CRITICAL if 'CRITICAL' in new_event_type.value else EventPriority.HIGH
                
                await publish_safety_event(
                    new_event_type,
                    event_data,
                    source='legacy_adapter',
                    priority=priority
                )
            else:
                # 일반 이벤트는 이벤트 버스를 통해 발행
                event_bus = get_event_bus()
                if event_bus:
                    await event_bus.publish({
                        'type': new_event_type,
                        'data': event_data,
                        'source': 'legacy_adapter',
                        'timestamp': current_time
                    })
            
            logger.debug(f"이벤트 브리지 완료: {legacy_event} -> {new_event_type}")
            
        except Exception as e:
            logger.error(f"이벤트 브리지 중 오류: {e}")
    
    def get_sync_statistics(self) -> Dict[str, Any]:
        """동기화 통계 정보 반환"""
        return {
            'last_sync_time': self.last_sync_time,
            'sync_interval': self.sync_interval,
            'event_bridge_enabled': self.enable_event_bridge,
            'mapped_events': len(self.event_mapping),
            'recent_events': len(self.last_events),
            'metrics_manager_stats': self.metrics_manager.get_debug_info()
        }
    
    def configure_sync_interval(self, interval: float) -> None:
        """동기화 간격 조정"""
        if 0.05 <= interval <= 1.0:  # 50ms ~ 1초 사이만 허용
            self.sync_interval = interval
            logger.info(f"동기화 간격 변경: {interval}초")
        else:
            logger.warning(f"유효하지 않은 동기화 간격: {interval}")
    
    def enable_event_bridge(self, enable: bool = True) -> None:
        """이벤트 브리지 활성화/비활성화"""
        self.enable_event_bridge = enable
        logger.info(f"이벤트 브리지 {'활성화' if enable else '비활성화'}")
    
    def reset_event_history(self) -> None:
        """이벤트 히스토리 리셋 (디바운싱 초기화)"""
        self.last_events.clear()
        logger.info("이벤트 히스토리 리셋 완료")


class EnhancedAnalysisEngineWrapper:
    """
    기존 EnhancedAnalysisEngine을 새로운 인터페이스에 맞게 감싸는 래퍼
    
    이 클래스는 마치 전기 어댑터와 같은 역할을 합니다.
    기존의 전자제품(EnhancedAnalysisEngine)을 새로운 콘센트(새로운 인터페이스)에
    연결할 수 있도록 중간에서 변환해주는 역할을 담당합니다.
    
    주요 기능:
    1. 기존 엔진을 새로운 IAnalysisOrchestrator 인터페이스로 노출
    2. 메트릭 동기화를 자동으로 처리
    3. 이벤트 시스템 통합
    4. 성능 모니터링 및 최적화
    """
    
    def __init__(self, legacy_engine, metrics_manager: MetricsManager):
        """
        래퍼 초기화
        
        Args:
            legacy_engine: 기존 EnhancedAnalysisEngine 인스턴스
            metrics_manager: 새로운 MetricsManager 인스턴스
        """
        self.legacy_engine = legacy_engine
        self.metrics_manager = metrics_manager
        
        # 어댑터 생성
        self.adapter = LegacySystemAdapter(metrics_manager)
        
        # 성능 추적
        self.frame_count = 0
        self.total_processing_time = 0.0
        
        logger.info("기존 엔진 래퍼 초기화 완료")
    
    async def process_frame_data(self, mediapipe_results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """
        프레임 데이터 처리 (새로운 인터페이스)
        
        이 메서드는 새로운 시스템의 인터페이스를 사용하면서,
        내부적으로는 기존 엔진을 호출하는 방식으로 작동합니다.
        """
        processing_start = time.time()
        
        try:
            # 기존 엔진의 형식에 맞게 데이터 변환
            # (실제 구현에서는 mediapipe_results를 기존 엔진이 기대하는 형식으로 변환해야 함)
            
            # 기존 엔진 호출 (임시로 더미 프레임 사용)
            dummy_frame = None  # 실제로는 적절한 프레임 데이터 필요
            perf_stats = {}
            playback_info = {}
            
            # 주의: 이 부분은 실제 구현에서 수정이 필요합니다
            # annotated_frame = await self.legacy_engine.process_and_annotate_frame(
            #     dummy_frame, mediapipe_results, perf_stats, playback_info
            # )
            
            # 기존 엔진에서 최신 메트릭 가져오기
            legacy_metrics = self.legacy_engine.get_latest_metrics()
            
            # 메트릭을 새로운 시스템으로 동기화
            self.adapter.sync_metrics_from_legacy(legacy_metrics)
            
            # 처리 시간 계산
            processing_time = time.time() - processing_start
            self.frame_count += 1
            self.total_processing_time += processing_time
            
            # 성능 메트릭 업데이트
            self.metrics_manager.update_system_performance_metrics({
                'processing_time_ms': processing_time * 1000,
                'fps': 1.0 / processing_time if processing_time > 0 else 0,
                'memory_mb': 0  # 실제 메모리 사용량으로 교체 필요
            })
            
            # 결과 반환 (새로운 형식)
            return {
                'processed_data': {
                    'face': mediapipe_results.get('face'),
                    'pose': mediapipe_results.get('pose'),
                    'hand': mediapipe_results.get('hand'),
                    'object': mediapipe_results.get('object')
                },
                'fused_risks': {
                    'fatigue_score': legacy_metrics.fatigue_risk_score,
                    'distraction_score': legacy_metrics.distraction_risk_score
                },
                'execution_quality': {
                    'confidence_score': (legacy_metrics.drowsiness_confidence + legacy_metrics.emotion_confidence) / 2,
                    'degraded_performance': processing_time > 0.2,  # 200ms 초과시 성능 저하로 간주
                    'processing_time': processing_time
                },
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.error(f"래퍼에서 프레임 처리 중 오류: {e}")
            
            # 오류 상황에서도 기본 응답 제공
            return {
                'processed_data': {},
                'fused_risks': {'fatigue_score': 0.0, 'distraction_score': 0.0},
                'execution_quality': {'confidence_score': 0.0, 'degraded_performance': True},
                'timestamp': timestamp,
                'error': str(e)
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """시스템 건강도 반환"""
        avg_processing_time = (
            self.total_processing_time / self.frame_count 
            if self.frame_count > 0 else 0
        )
        
        return {
            'system_type': 'legacy_wrapped',
            'frames_processed': self.frame_count,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'health_status': 'healthy' if avg_processing_time < 0.2 else 'degraded',
            'adapter_stats': self.adapter.get_sync_statistics()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 반환"""
        return {
            'total_frames': self.frame_count,
            'total_processing_time': self.total_processing_time,
            'average_fps': self.frame_count / self.total_processing_time if self.total_processing_time > 0 else 0,
            'legacy_engine_type': type(self.legacy_engine).__name__
        }
    
    async def shutdown(self):
        """시스템 종료"""
        logger.info("기존 엔진 래퍼 종료 중...")
        
        # 최종 통계 출력
        if self.frame_count > 0:
            avg_time = self.total_processing_time / self.frame_count
            logger.info(f"래퍼 최종 통계 - 총 {self.frame_count}프레임, 평균 {avg_time*1000:.1f}ms")
        
        logger.info("기존 엔진 래퍼 종료 완료")
