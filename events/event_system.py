"""
Event System (Advanced)
이벤트 기반 아키텍처를 통한 모듈 간 느슨한 결합 및 확장성 제공
- [Event] 이벤트 기반 통신으로 모듈 간 의존성 최소화
- [Pattern] 옵저버 패턴과 발행-구독 패턴의 하이브리드 구현
- [Priority] 이벤트 우선순위 기반 처리
- [History] 이벤트 이력 관리 및 패턴 분석
"""

import asyncio
import time
from typing import Dict, Any, List, Callable, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from collections import deque, defaultdict
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EventPriority(IntEnum):
    """이벤트 우선순위"""
    CRITICAL = 1    # 즉시 처리 필요 (안전 관련)
    HIGH = 2        # 높은 우선순위 (위험 감지)
    NORMAL = 3      # 일반 우선순위 (상태 업데이트)
    LOW = 4         # 낮은 우선순위 (로깅, 통계)


class EventType(Enum):
    """이벤트 타입 분류"""
    # 안전 관련 이벤트 (최고 우선순위)
    EMERGENCY_STOP = "emergency_stop"
    CRITICAL_DROWSINESS = "critical_drowsiness"
    IMMEDIATE_DANGER = "immediate_danger"

    # 위험 감지 이벤트
    DROWSINESS_DETECTED = "drowsiness_detected"
    DISTRACTION_DETECTED = "distraction_detected"
    PHONE_USAGE_DETECTED = "phone_usage_detected"
    HANDS_OFF_WHEEL = "hands_off_wheel"

    # 상태 변화 이벤트
    DRIVER_STATE_CHANGED = "driver_state_changed"
    EMOTION_STATE_CHANGED = "emotion_state_changed"
    GAZE_ZONE_CHANGED = "gaze_zone_changed"

    # 시스템 이벤트
    PROCESSOR_PERFORMANCE_CHANGED = "processor_performance_changed"
    FUSION_CONFIDENCE_CHANGED = "fusion_confidence_changed"
    SYSTEM_HEALTH_CHANGED = "system_health_changed"

    # 예측 이벤트
    RISK_PREDICTION = "risk_prediction"
    BEHAVIOR_PREDICTION = "behavior_prediction"

    # 분석 완료 이벤트
    FRAME_ANALYSIS_COMPLETE = "frame_analysis_complete"
    FUSION_ANALYSIS_COMPLETE = "fusion_analysis_complete"


@dataclass
class Event:
    """이벤트 데이터 클래스"""
    event_type: EventType
    source: str  # 이벤트 발생원
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None  # 관련 이벤트들을 그룹화
    expires_at: Optional[float] = None    # 이벤트 만료 시간

    def is_expired(self) -> bool:
        """이벤트 만료 여부 확인"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def get_age(self) -> float:
        """이벤트 생성 후 경과 시간"""
        return time.time() - self.timestamp


class IEventHandler(ABC):
    """이벤트 핸들러 인터페이스"""

    @abstractmethod
    async def handle_event(self, event: Event) -> bool:
        """
        이벤트 처리

        Returns:
            bool: 처리 성공 여부
        """
        pass

    @abstractmethod
    def get_handled_event_types(self) -> Set[EventType]:
        """처리 가능한 이벤트 타입들"""
        pass

    @abstractmethod
    def get_handler_name(self) -> str:
        """핸들러 이름"""
        pass


@dataclass
class EventSubscription:
    """이벤트 구독 정보"""
    handler: IEventHandler
    event_types: Set[EventType]
    priority_filter: Optional[Set[EventPriority]] = None
    condition_filter: Optional[Callable[[Event], bool]] = None
    max_age_seconds: Optional[float] = None


class EventBus:
    """
    Event Bus - 이벤트 기반 통신의 중앙 허브

    이 클래스는 마치 도시의 교통 관제센터처럼 모든 이벤트의
    흐름을 관리하고 적절한 수신자에게 전달합니다.
    """

    def __init__(self, max_event_history: int = 1000):
        self.subscriptions: List[EventSubscription] = []
        self.event_queue = asyncio.PriorityQueue()
        self.event_history = deque(maxlen=max_event_history)
        self.event_stats = defaultdict(int)
        self.processing_task: Optional[asyncio.Task] = None
        self.is_running = False

        # 성능 메트릭
        self.total_events_processed = 0
        self.failed_events = 0
        self.avg_processing_time = 0.0

        logger.info("EventBus 초기화 완료")

    async def start(self):
        """이벤트 버스 시작"""
        if self.is_running:
            return

        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_events())
        logger.info("EventBus 시작됨")

    async def stop(self):
        """이벤트 버스 중지"""
        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        logger.info("EventBus 중지됨")

    def subscribe(
        self,
        handler: IEventHandler,
        event_types: Optional[Set[EventType]] = None,
        priority_filter: Optional[Set[EventPriority]] = None,
        condition_filter: Optional[Callable[[Event], bool]] = None,
        max_age_seconds: Optional[float] = None
    ):
        """이벤트 구독"""
        if event_types is None:
            event_types = handler.get_handled_event_types()

        subscription = EventSubscription(
            handler=handler,
            event_types=event_types,
            priority_filter=priority_filter,
            condition_filter=condition_filter,
            max_age_seconds=max_age_seconds
        )

        self.subscriptions.append(subscription)
        logger.info(f"이벤트 구독 추가: {handler.get_handler_name()} -> {len(event_types)}개 이벤트 타입")

    def unsubscribe(self, handler: IEventHandler):
        """이벤트 구독 해제"""
        self.subscriptions = [
            sub for sub in self.subscriptions
            if sub.handler != handler
        ]
        logger.info(f"이벤트 구독 해제: {handler.get_handler_name()}")

    async def publish(self, event: Event):
        """이벤트 발행"""
        if event.is_expired():
            logger.warning(f"만료된 이벤트 무시: {event.event_type.value}")
            return

        # 우선순위 기반으로 큐에 추가 (우선순위 값이 낮을수록 높은 우선순위)
        priority_value = event.priority.value
        await self.event_queue.put((priority_value, time.time(), event))

        # 통계 업데이트
        self.event_stats[event.event_type] += 1

        logger.debug(f"이벤트 발행: {event.event_type.value} (우선순위: {event.priority.name})")

    async def publish_immediate(self, event: Event):
        """즉시 처리 이벤트 발행 (큐를 우회)"""
        logger.warning(f"즉시 처리 이벤트: {event.event_type.value}")
        await self._deliver_event(event)

    async def _process_events(self):
        """이벤트 처리 루프"""
        while self.is_running:
            try:
                # 큐에서 이벤트 가져오기 (타임아웃 설정)
                priority, queued_time, event = await asyncio.wait_for(
                    self.event_queue.get(), timeout=0.1
                )

                # 이벤트 처리
                start_time = time.time()
                await self._deliver_event(event)
                processing_time = time.time() - start_time

                # 성능 메트릭 업데이트
                self._update_performance_metrics(processing_time)

                # 이벤트 이력에 추가
                self.event_history.append(event)

            except (asyncio.CancelledError, asyncio.TimeoutError) as e:
                logger.debug(f"이벤트 처리 루프 중단: {e}")
                break
            except Exception as e:
                self.failed_events += 1
                if not hasattr(self, '_process_events_fail_count'):
                    self._process_events_fail_count = 0
                self._process_events_fail_count += 1
                if self._process_events_fail_count >= 3:
                    global safe_mode
                    safe_mode = True  # 시스템 전체 안전 모드 진입

    async def _deliver_event(self, event: Event):
        """이벤트를 구독자들에게 전달"""
        delivery_tasks = []

        for subscription in self.subscriptions:
            if self._should_deliver(event, subscription):
                task = asyncio.create_task(
                    self._safe_deliver_to_handler(event, subscription.handler)
                )
                delivery_tasks.append(task)

        # 모든 핸들러에게 동시에 전달
        if delivery_tasks:
            await asyncio.gather(*delivery_tasks, return_exceptions=True)

    def _should_deliver(self, event: Event, subscription: EventSubscription) -> bool:
        """이벤트를 특정 구독에 전달해야 하는지 판단"""

        # 이벤트 타입 확인
        if event.event_type not in subscription.event_types:
            return False

        # 우선순위 필터 확인
        if subscription.priority_filter and event.priority not in subscription.priority_filter:
            return False

        # 나이 필터 확인
        if subscription.max_age_seconds and event.get_age() > subscription.max_age_seconds:
            return False

        # 조건 필터 확인
        if subscription.condition_filter and not subscription.condition_filter(event):
            return False

        return True

    async def _safe_deliver_to_handler(self, event: Event, handler: IEventHandler):
        """안전한 핸들러 전달 (예외 처리 포함)"""
        try:
            await asyncio.wait_for(handler.handle_event(event), timeout=1.0)
        except asyncio.TimeoutError:
            logger.warning(f"핸들러 타임아웃: {handler.get_handler_name()}")
        except Exception as e:
            logger.error(f"핸들러 오류 {handler.get_handler_name()}: {e}", exc_info=True)

    def _update_performance_metrics(self, processing_time: float):
        """성능 메트릭 업데이트"""
        self.total_events_processed += 1

        # 지수 이동 평균으로 평균 처리 시간 계산
        alpha = 0.1
        self.avg_processing_time = (
            self.avg_processing_time * (1 - alpha) + processing_time * alpha
        )

    def get_event_statistics(self) -> Dict[str, Any]:
        """이벤트 통계 조회"""
        return {
            'total_processed': self.total_events_processed,
            'failed_events': self.failed_events,
            'avg_processing_time_ms': self.avg_processing_time * 1000,
            'queue_size': self.event_queue.qsize(),
            'subscription_count': len(self.subscriptions),
            'event_type_stats': dict(self.event_stats),
            'success_rate': (
                (self.total_events_processed - self.failed_events) /
                max(1, self.total_events_processed)
            )
        }

    def analyze_event_patterns(self, time_window_seconds: float = 60.0) -> Dict[str, Any]:
        """이벤트 패턴 분석"""
        current_time = time.time()
        recent_events = [
            event for event in self.event_history
            if current_time - event.timestamp <= time_window_seconds
        ]

        if not recent_events:
            return {'pattern': 'no_activity', 'event_count': 0}

        # 이벤트 빈도 분석
        event_frequencies = defaultdict(int)
        for event in recent_events:
            event_frequencies[event.event_type] += 1

        # 위험 이벤트 비율
        risk_events = [
            event for event in recent_events
            if event.event_type in [
                EventType.DROWSINESS_DETECTED,
                EventType.DISTRACTION_DETECTED,
                EventType.CRITICAL_DROWSINESS
            ]
        ]
        risk_ratio = len(risk_events) / len(recent_events)

        # 패턴 분류
        if risk_ratio > 0.5:
            pattern = 'high_risk_activity'
        elif risk_ratio > 0.2:
            pattern = 'moderate_risk_activity'
        elif len(recent_events) > 50:
            pattern = 'high_activity'
        else:
            pattern = 'normal_activity'

        return {
            'pattern': pattern,
            'event_count': len(recent_events),
            'risk_ratio': risk_ratio,
            'most_frequent_events': sorted(
                event_frequencies.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


class SafetyEventHandler(IEventHandler):
    """
    안전 이벤트 핸들러

    안전과 관련된 중요한 이벤트들을 처리하여 즉각적인 대응을 수행합니다.
    마치 응급실의 의료진처럼 생명과 직결된 상황에 최우선으로 대응합니다.
    """

    def __init__(self, alert_system):
        self.alert_system = alert_system
        self.emergency_protocols = {
            EventType.CRITICAL_DROWSINESS: self._handle_critical_drowsiness,
            EventType.IMMEDIATE_DANGER: self._handle_immediate_danger,
            EventType.EMERGENCY_STOP: self._handle_emergency_stop
        }

    async def handle_event(self, event: Event) -> bool:
        """안전 이벤트 처리"""
        try:
            handler = self.emergency_protocols.get(event.event_type)
            if handler:
                await handler(event)
                return True

            # 일반적인 위험 이벤트 처리
            if event.priority == EventPriority.CRITICAL:
                await self._handle_general_critical_event(event)
                return True

            return False

        except Exception as e:
            logger.error(f"안전 이벤트 처리 중 오류: {e}", exc_info=True)
            return False

    def get_handled_event_types(self) -> Set[EventType]:
        """처리 가능한 이벤트 타입들"""
        return {
            EventType.CRITICAL_DROWSINESS,
            EventType.IMMEDIATE_DANGER,
            EventType.EMERGENCY_STOP,
            EventType.DROWSINESS_DETECTED,
            EventType.DISTRACTION_DETECTED
        }

    def get_handler_name(self) -> str:
        return "SafetyEventHandler"

    async def _handle_critical_drowsiness(self, event: Event):
        """치명적 졸음 상태 처리"""
        logger.critical("치명적 졸음 상태 감지 - 응급 프로토콜 활성화")

        # 즉각적인 경고 발생
        await self.alert_system.trigger_emergency_alert(
            "운전자가 심각한 졸음 상태입니다. 즉시 안전한 곳에 정차하세요!",
            alert_type="critical_drowsiness"
        )

        # 자동 대응 시스템이 있다면 활성화
        # (예: 자동 비상등 점멸, 속도 제한 등)

    async def _handle_immediate_danger(self, event: Event):
        """즉각적 위험 상황 처리"""
        logger.critical("즉각적 위험 상황 감지")

        danger_type = event.data.get('danger_type', 'unknown')
        await self.alert_system.trigger_emergency_alert(
            f"위험 상황 감지: {danger_type}",
            alert_type="immediate_danger"
        )

    async def _handle_emergency_stop(self, event: Event):
        """응급 정지 처리"""
        logger.critical("응급 정지 신호 수신")

        # 모든 시스템에 응급 정지 신호 전파
        # 실제 구현에서는 차량 제어 시스템과 연동

    async def _handle_general_critical_event(self, event: Event):
        """일반적인 중요 이벤트 처리"""
        logger.warning(f"중요 이벤트 처리: {event.event_type.value}")

        await self.alert_system.trigger_warning_alert(
            f"주의: {event.event_type.value}",
            data=event.data
        )


class AnalyticsEventHandler(IEventHandler):
    """
    분석 이벤트 핸들러

    시스템 성능과 사용자 행동 패턴을 분석하여
    지속적인 개선과 최적화를 수행합니다.
    """

    def __init__(self, analytics_engine):
        self.analytics_engine = analytics_engine
        self.behavior_patterns = deque(maxlen=1000)
        self.performance_metrics = deque(maxlen=500)

    async def handle_event(self, event: Event) -> bool:
        """분석 이벤트 처리"""
        try:
            # 행동 패턴 분석
            if event.event_type in [
                EventType.GAZE_ZONE_CHANGED,
                EventType.EMOTION_STATE_CHANGED,
                EventType.DRIVER_STATE_CHANGED
            ]:
                self.behavior_patterns.append({
                    'timestamp': event.timestamp,
                    'event_type': event.event_type,
                    'data': event.data
                })
                await self._analyze_behavior_patterns()

            # 시스템 성능 분석
            if event.event_type in [
                EventType.PROCESSOR_PERFORMANCE_CHANGED,
                EventType.FUSION_CONFIDENCE_CHANGED
            ]:
                self.performance_metrics.append({
                    'timestamp': event.timestamp,
                    'event_type': event.event_type,
                    'data': event.data
                })
                await self._analyze_system_performance()

            return True

        except Exception as e:
            logger.error(f"분석 이벤트 처리 중 오류: {e}", exc_info=True)
            return False

    def get_handled_event_types(self) -> Set[EventType]:
        return {
            EventType.GAZE_ZONE_CHANGED,
            EventType.EMOTION_STATE_CHANGED,
            EventType.DRIVER_STATE_CHANGED,
            EventType.PROCESSOR_PERFORMANCE_CHANGED,
            EventType.FUSION_CONFIDENCE_CHANGED,
            EventType.FRAME_ANALYSIS_COMPLETE
        }

    def get_handler_name(self) -> str:
        return "AnalyticsEventHandler"

    async def _analyze_behavior_patterns(self):
        """행동 패턴 분석"""
        if len(self.behavior_patterns) < 10:
            return

        # 최근 패턴 분석 로직
        recent_patterns = list(self.behavior_patterns)[-50:]

        # 패턴 감지 (예: 반복적인 주의산만 행동)
        pattern_analysis = await self.analytics_engine.analyze_patterns(recent_patterns)

        if pattern_analysis.get('needs_intervention'):
            # 개입이 필요한 패턴 감지시 이벤트 발행
            logger.info("행동 패턴 분석: 개입 필요한 패턴 감지")

    async def _analyze_system_performance(self):
        """시스템 성능 분석"""
        if len(self.performance_metrics) < 5:
            return

        # 성능 트렌드 분석
        performance_analysis = await self.analytics_engine.analyze_performance(
            list(self.performance_metrics)
        )

        if performance_analysis.get('degradation_detected'):
            logger.warning("시스템 성능 저하 감지")


# 전역 이벤트 버스 인스턴스
global_event_bus = EventBus()


async def initialize_event_system():
    """이벤트 시스템 초기화"""
    await global_event_bus.start()
    logger.info("전역 이벤트 시스템 초기화 완료")


async def shutdown_event_system():
    """이벤트 시스템 종료"""
    await global_event_bus.stop()
    logger.info("전역 이벤트 시스템 종료 완료")


def get_event_bus() -> EventBus:
    """전역 이벤트 버스 조회"""
    return global_event_bus


# 편의 함수들
async def publish_safety_event(event_type: EventType, data: Dict[str, Any], source: str = "system"):
    """안전 이벤트 발행 편의 함수"""
    event = Event(
        event_type=event_type,
        source=source,
        data=data,
        priority=EventPriority.CRITICAL
    )
    await global_event_bus.publish(event)


async def publish_analysis_event(event_type: EventType, data: Dict[str, Any], source: str = "analysis"):
    """분석 이벤트 발행 편의 함수"""
    event = Event(
        event_type=event_type,
        source=source,
        data=data,
        priority=EventPriority.NORMAL
    )
    await global_event_bus.publish(event)