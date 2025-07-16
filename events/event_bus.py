"""
Event System Core - DMS 시스템의 중추 신경계

이 모듈은 시스템 전체의 이벤트 흐름을 관리하는 핵심 인프라입니다.

비유: 도시의 종합상황실
- 각 구역(모듈)에서 발생하는 모든 사건들을 실시간으로 수집
- 사건의 중요도와 유형에 따라 적절한 대응팀에 즉시 전달
- 대응 과정과 결과를 모니터링하여 시스템 전체의 건강도 관리
- 패턴 분석을 통한 예방적 조치 및 시스템 최적화

예를 들어:
- 졸음 감지 센서 → 이벤트 버스 → 안전팀 + 분석팀 → 즉시 경고 + 데이터 수집
- 시스템 과부하 → 이벤트 버스 → 성능팀 → 자동 최적화 수행
- 프로세서 실패 → 이벤트 버스 → 복구팀 → 백업 시스템 활성화
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Callable, Set
import threading
import weakref

logger = logging.getLogger(__name__)


class EventType(Enum):
    """
    시스템에서 발생할 수 있는 모든 이벤트 타입들

    비유: 도시의 비상 코드들
    - 화재 (안전 위험) → 빨간색
    - 교통 체증 (성능 저하) → 노란색
    - 정전 (시스템 장애) → 검은색
    - 축제 (정상 운영) → 녹색
    """

    # === 안전 관련 이벤트 (최우선) ===
    DROWSINESS_DETECTED = "drowsiness_detected"
    CRITICAL_DROWSINESS = "critical_drowsiness"
    DISTRACTION_DETECTED = "distraction_detected"
    CRITICAL_DISTRACTION = "critical_distraction"
    DANGEROUS_BEHAVIOR = "dangerous_behavior"
    EMERGENCY_SITUATION = "emergency_situation"

    # === 시스템 성능 이벤트 ===
    FRAME_ANALYSIS_COMPLETE = "frame_analysis_complete"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SYSTEM_OVERLOAD = "system_overload"
    RESOURCE_EXHAUSTION = "resource_exhaustion"

    # === 컴포넌트 상태 이벤트 ===
    MODULE_FAILURE = "module_failure"
    MODULE_RECOVERY = "module_recovery"
    PROCESSOR_TIMEOUT = "processor_timeout"
    FUSION_ENGINE_ERROR = "fusion_engine_error"

    # === 데이터 품질 이벤트 ===
    LOW_CONFIDENCE_DETECTION = "low_confidence_detection"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    CALIBRATION_REQUIRED = "calibration_required"
    SENSOR_MALFUNCTION = "sensor_malfunction"

    # === 운영 및 분석 이벤트 ===
    DRIVER_BEHAVIOR_CHANGE = "driver_behavior_change"
    PATTERN_ANOMALY_DETECTED = "pattern_anomaly_detected"
    SYSTEM_OPTIMIZATION_TRIGGERED = "system_optimization_triggered"
    MAINTENANCE_REQUIRED = "maintenance_required"


class EventPriority(IntEnum):
    """이벤트 우선순위 - 응급실의 트리아지 시스템과 같음"""

    LOW = 1  # 정보성 (로깅만)
    MEDIUM = 2  # 모니터링 필요
    HIGH = 3  # 즉시 조치 필요
    CRITICAL = 4  # 긴급 대응 필요
    EMERGENCY = 5  # 최우선 처리


@dataclass
class Event:
    """
    이벤트 데이터 구조

    비유: 911 신고서
    - 언제 발생했나? (timestamp)
    - 무슨 일인가? (event_type)
    - 얼마나 심각한가? (priority)
    - 상세 내용은? (data)
    - 누가 신고했나? (source)
    - 처리되었나? (handled)
    """

    event_type: EventType
    priority: EventPriority
    timestamp: datetime
    data: Dict[str, Any]
    source: str
    event_id: str = ""
    handled: bool = False
    handlers_notified: Set[str] = None

    def __post_init__(self):
        if not self.event_id:
            # 고유 ID 생성 (timestamp + source + event_type)
            time_str = str(int(self.timestamp.timestamp() * 1000))[-8:]  # 마지막 8자리
            self.event_id = f"{self.source[:3]}-{self.event_type.value[:8]}-{time_str}"

        if self.handlers_notified is None:
            self.handlers_notified = set()


class IEventHandler(ABC):
    """이벤트 핸들러 인터페이스"""

    @abstractmethod
    async def handle_event(self, event: Event) -> bool:
        """이벤트 처리"""
        pass

    @abstractmethod
    def can_handle(self, event_type: EventType) -> bool:
        """이 핸들러가 해당 이벤트를 처리할 수 있는지"""
        pass

    @abstractmethod
    def get_handler_name(self) -> str:
        """핸들러 식별 이름"""
        pass


class EventBus:
    """
    이벤트 버스 - 시스템의 중추 신경계

    비유: 도시의 종합상황실 + 통신 센터
    - 모든 사건 접수 및 분류
    - 적절한 대응팀에 즉시 전파
    - 대응 과정 모니터링
    - 상황 통계 및 패턴 분석

    특징:
    - 비동기 처리로 실시간 성능 보장
    - 우선순위 큐로 중요한 이벤트 우선 처리
    - 핸들러 실패시에도 다른 핸들러는 정상 동작
    - 메모리 효율적인 약한 참조 사용
    """

    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size

        # 핸들러 등록 (약한 참조로 메모리 누수 방지)
        self._handlers: Dict[EventType, List[weakref.ref]] = defaultdict(list)
        self._global_handlers: List[weakref.ref] = []  # 모든 이벤트 처리

        # 이벤트 큐들 (우선순위별)
        self._priority_queues = {
            EventPriority.EMERGENCY: asyncio.Queue(maxsize=100),
            EventPriority.CRITICAL: asyncio.Queue(maxsize=500),
            EventPriority.HIGH: asyncio.Queue(maxsize=1000),
            EventPriority.MEDIUM: asyncio.Queue(maxsize=3000),
            EventPriority.LOW: asyncio.Queue(maxsize=5000),
        }

        # 처리 통계
        self._event_statistics = {
            "total_events": 0,
            "events_by_type": defaultdict(int),
            "events_by_priority": defaultdict(int),
            "handler_performance": defaultdict(lambda: {"success": 0, "failure": 0}),
            "average_processing_time": 0.0,
        }

        # 이벤트 이력 (최근 1000개만 보관)
        self._event_history = deque(maxlen=1000)

        # 백그라운드 처리 태스크들
        self._processor_tasks = []
        self._running = False

        # 성능 모니터링
        self._processing_times = deque(maxlen=100)

        logger.info("EventBus 초기화 완료 - 중추 신경계 가동 시작")

    async def start(self):
        """이벤트 버스 시작"""
        if self._running:
            logger.warning("EventBus가 이미 실행 중입니다")
            return

        self._running = True

        # 우선순위별 처리 태스크 시작
        for priority in EventPriority:
            task = asyncio.create_task(self._process_priority_queue(priority))
            self._processor_tasks.append(task)

        # 정리 태스크 시작 (주기적으로 끊어진 약한 참조 정리)
        cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self._processor_tasks.append(cleanup_task)

        logger.info("EventBus 시작됨 - 모든 우선순위 큐 처리 시작")

    async def stop(self):
        """이벤트 버스 종료"""
        if not self._running:
            return

        self._running = False
        logger.info("EventBus 종료 중...")

        # 모든 처리 태스크 취소
        for task in self._processor_tasks:
            task.cancel()

        # 태스크 완료 대기
        if self._processor_tasks:
            await asyncio.gather(*self._processor_tasks, return_exceptions=True)

        self._processor_tasks.clear()
        logger.info("EventBus 종료 완료")

    def subscribe(
        self, handler: IEventHandler, event_types: Optional[List[EventType]] = None
    ):
        """
        핸들러 등록

        Args:
            handler: 등록할 핸들러
            event_types: 처리할 이벤트 타입들 (None이면 모든 이벤트)
        """
        handler_ref = weakref.ref(handler)

        if event_types is None:
            # 모든 이벤트 처리
            self._global_handlers.append(handler_ref)
            logger.info(f"글로벌 핸들러 등록: {handler.get_handler_name()}")
        else:
            # 특정 이벤트만 처리
            for event_type in event_types:
                self._handlers[event_type].append(handler_ref)
            logger.info(
                f"핸들러 등록: {handler.get_handler_name()} - {len(event_types)}개 이벤트 타입"
            )

    def unsubscribe(self, handler: IEventHandler):
        """핸들러 등록 해제"""
        handler_name = handler.get_handler_name()

        # 글로벌 핸들러에서 제거
        self._global_handlers = [
            ref for ref in self._global_handlers if ref() is not handler
        ]

        # 특정 이벤트 핸들러에서 제거
        for event_type in list(self._handlers.keys()):
            self._handlers[event_type] = [
                ref for ref in self._handlers[event_type] if ref() is not handler
            ]
            if not self._handlers[event_type]:
                del self._handlers[event_type]

        logger.info(f"핸들러 등록 해제: {handler_name}")

    async def publish(self, event: Event):
        """
        이벤트 발행

        비유: 911 센터에 신고 접수
        - 신고 내용 확인 및 분류
        - 우선순위에 따른 대기열 배정
        - 관련 부서들에 즉시 통보
        """
        if not self._running:
            logger.error("EventBus가 실행되지 않은 상태에서 이벤트 발행 시도")
            return

        try:
            # 통계 업데이트
            self._update_statistics(event)

            # 이력에 추가
            self._event_history.append(event)

            # 우선순위에 따른 큐에 추가
            queue = self._priority_queues[event.priority]

            # 큐가 가득 찬 경우 (시스템 과부하)
            if queue.full():
                # 낮은 우선순위 이벤트는 드롭
                if event.priority in [EventPriority.LOW, EventPriority.MEDIUM]:
                    logger.warning(f"큐 포화로 이벤트 드롭: {event.event_type.value}")
                    return
                else:
                    # 높은 우선순위는 강제 처리
                    logger.critical(
                        f"높은 우선순위 이벤트 강제 처리: {event.event_type.value}"
                    )

            await queue.put(event)

            # 긴급 이벤트는 로그에 즉시 기록
            if event.priority >= EventPriority.CRITICAL:
                logger.critical(
                    f"긴급 이벤트 발행: {event.event_type.value} (출처: {event.source})"
                )

        except (asyncio.QueueFull, AttributeError, TypeError) as e:
            logger.error(f"이벤트 발행 중 오류: {e}")

    async def _process_priority_queue(self, priority: EventPriority):
        """우선순위별 이벤트 큐 처리"""
        queue = self._priority_queues[priority]

        while self._running:
            try:
                # 이벤트 대기 (타임아웃으로 주기적으로 깨어남)
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue  # 타임아웃되면 계속 대기

                # 이벤트 처리
                await self._handle_event(event)

                # 큐에서 작업 완료 표시
                queue.task_done()

            except (asyncio.CancelledError, asyncio.TimeoutError) as e:
                logger.info(f"{priority.name} 우선순위 큐 처리 취소됨: {e}")
                break
            except Exception as e:
                logger.error(f"{priority.name} 큐 처리 중 오류: {e}", exc_info=True)

    async def _handle_event(self, event: Event):
        """개별 이벤트 처리"""
        start_time = time.time()

        try:
            # 해당 이벤트를 처리할 수 있는 핸들러들 수집
            target_handlers = []

            # 특정 이벤트 타입 핸들러들
            if event.event_type in self._handlers:
                target_handlers.extend(self._handlers[event.event_type])

            # 글로벌 핸들러들
            target_handlers.extend(self._global_handlers)

            # 실제 핸들러 객체들 추출 (약한 참조에서)
            active_handlers = []
            for handler_ref in target_handlers:
                handler = handler_ref()
                if handler is not None:
                    if handler.can_handle(event.event_type):
                        active_handlers.append(handler)

            if not active_handlers:
                logger.debug(f"이벤트 {event.event_type.value}를 처리할 핸들러가 없음")
                return

            # 모든 핸들러에게 병렬로 이벤트 전달
            handler_tasks = []
            for handler in active_handlers:
                task = asyncio.create_task(self._execute_handler_safely(handler, event))
                handler_tasks.append(task)

            # 모든 핸들러 처리 완료 대기
            results = await asyncio.gather(*handler_tasks, return_exceptions=True)

            # 결과 분석
            success_count = sum(1 for result in results if result is True)
            failure_count = len(results) - success_count

            # 처리 완료 표시
            event.handled = True
            event.handlers_notified = {
                handler.get_handler_name() for handler in active_handlers
            }

            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._processing_times.append(processing_time)

            # 처리 결과 로깅
            if failure_count > 0:
                logger.warning(
                    f"이벤트 {event.event_type.value} 처리 완료: {success_count}성공/{failure_count}실패"
                )
            else:
                logger.debug(
                    f"이벤트 {event.event_type.value} 처리 완료: {success_count}개 핸들러"
                )

        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"이벤트 처리 중 오류: {e}", exc_info=True)
            raise

    async def _execute_handler_safely(
        self, handler: IEventHandler, event: Event
    ) -> bool:
        """핸들러를 안전하게 실행 (개별 핸들러 실패가 전체에 영향 안 줌)"""
        handler_name = handler.get_handler_name()

        try:
            # 타임아웃 적용 (핸들러가 너무 오래 걸리면 취소)
            timeout = 5.0 if event.priority >= EventPriority.CRITICAL else 10.0

            result = await asyncio.wait_for(
                handler.handle_event(event), timeout=timeout
            )

            # 성능 통계 업데이트
            self._event_statistics["handler_performance"][handler_name]["success"] += 1

            return bool(result)

        except (asyncio.TimeoutError, AttributeError, TypeError, ValueError) as e:
            logger.error(f"핸들러 {handler_name} 타임아웃/오류 ({timeout}초): {e}")
            self._event_statistics["handler_performance"][handler_name]["failure"] += 1
            return False

        except Exception as e:
            logger.error(f"핸들러 {handler_name} 실행 중 오류: {e}", exc_info=True)
            self._event_statistics["handler_performance"][handler_name]["failure"] += 1
            return False

    async def _periodic_cleanup(self):
        """주기적 정리 작업"""
        while self._running:
            try:
                await asyncio.sleep(30.0)  # 30초마다 실행

                # 끊어진 약한 참조들 정리
                self._cleanup_dead_references()

                # 통계 리포트 (5분마다)
                if int(time.time()) % 300 == 0:
                    self._log_statistics_report()

            except asyncio.CancelledError:
                break
            except (asyncio.CancelledError, AttributeError, TypeError) as e:
                logger.error(f"정리 작업 중 오류: {e}", exc_info=True)

    def _cleanup_dead_references(self):
        """끊어진 약한 참조들 정리"""
        # 글로벌 핸들러 정리
        self._global_handlers = [
            ref for ref in self._global_handlers if ref() is not None
        ]

        # 이벤트별 핸들러 정리
        for event_type in list(self._handlers.keys()):
            self._handlers[event_type] = [
                ref for ref in self._handlers[event_type] if ref() is not None
            ]
            if not self._handlers[event_type]:
                del self._handlers[event_type]

    def _update_statistics(self, event: Event):
        """통계 정보 업데이트"""
        self._event_statistics["total_events"] += 1
        self._event_statistics["events_by_type"][event.event_type.value] += 1
        self._event_statistics["events_by_priority"][event.priority.value] += 1

    def _log_statistics_report(self):
        """통계 리포트 로깅"""
        stats = self._event_statistics

        # 평균 처리 시간 계산
        if self._processing_times:
            avg_time = sum(self._processing_times) / len(self._processing_times)
            stats["average_processing_time"] = avg_time

        logger.info(
            f"EventBus 통계 - 총 이벤트: {stats['total_events']}, "
            f"평균 처리시간: {stats['average_processing_time'] * 1000:.1f}ms"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환 (외부 모니터링용)"""
        return {
            "total_events": self._event_statistics["total_events"],
            "events_by_type": dict(self._event_statistics["events_by_type"]),
            "events_by_priority": dict(self._event_statistics["events_by_priority"]),
            "handler_performance": dict(self._event_statistics["handler_performance"]),
            "average_processing_time_ms": self._event_statistics[
                "average_processing_time"
            ]
            * 1000,
            "queue_sizes": {
                priority.name: queue.qsize()
                for priority, queue in self._priority_queues.items()
            },
            "active_handlers": len(self._global_handlers)
            + sum(len(handlers) for handlers in self._handlers.values()),
            "event_history_size": len(self._event_history),
            "is_running": self._running,
        }

    def get_recent_events(self, limit: int = 50) -> List[Event]:
        """최근 이벤트 목록 반환"""
        return list(self._event_history)[-limit:]


# === 전역 이벤트 시스템 인터페이스 ===

# 전역 이벤트 버스 인스턴스
_global_event_bus: Optional[EventBus] = None


async def initialize_event_system(max_queue_size: int = 10000) -> EventBus:
    """
    이벤트 시스템 초기화

    비유: 도시의 종합상황실 개소
    - 모든 통신 장비 점검 및 가동
    - 대응팀들과의 연결 확인
    - 비상 대응 절차 준비
    """
    global _global_event_bus

    if _global_event_bus is not None:
        logger.warning("이벤트 시스템이 이미 초기화되었습니다")
        return _global_event_bus

    _global_event_bus = EventBus(max_queue_size)
    await _global_event_bus.start()

    logger.info("글로벌 이벤트 시스템 초기화 완료")
    return _global_event_bus


async def shutdown_event_system():
    """이벤트 시스템 종료"""
    global _global_event_bus

    if _global_event_bus is not None:
        await _global_event_bus.stop()
        _global_event_bus = None
        logger.info("글로벌 이벤트 시스템 종료 완료")


def get_event_bus() -> EventBus:
    """현재 이벤트 버스 반환"""
    if _global_event_bus is None:
        raise RuntimeError(
            "이벤트 시스템이 초기화되지 않았습니다. initialize_event_system()을 먼저 호출하세요."
        )
    return _global_event_bus


# === 편의 함수들 ===


async def publish_safety_event(
    event_type: EventType,
    data: Dict[str, Any],
    source: str = "unknown",
    priority: EventPriority = EventPriority.HIGH,
):
    """안전 이벤트 발행 편의 함수"""
    event = Event(
        event_type=event_type,
        priority=priority,
        timestamp=datetime.now(),
        data=data,
        source=source,
    )

    event_bus = get_event_bus()
    await event_bus.publish(event)


async def publish_analysis_event(
    event_type: EventType, data: Dict[str, Any], source: str = "unknown"
):
    """분석 이벤트 발행 편의 함수"""
    event = Event(
        event_type=event_type,
        priority=EventPriority.MEDIUM,
        timestamp=datetime.now(),
        data=data,
        source=source,
    )

    event_bus = get_event_bus()
    await event_bus.publish(event)


async def publish_system_event(
    event_type: EventType,
    data: Dict[str, Any],
    source: str = "unknown",
    priority: EventPriority = EventPriority.MEDIUM,
):
    """시스템 이벤트 발행 편의 함수"""
    event = Event(
        event_type=event_type,
        priority=priority,
        timestamp=datetime.now(),
        data=data,
        source=source,
    )

    event_bus = get_event_bus()
    await event_bus.publish(event)


def register_global_handler(handler: IEventHandler):
    """글로벌 핸들러 등록 편의 함수"""
    event_bus = get_event_bus()
    event_bus.subscribe(handler)


def register_specific_handler(handler: IEventHandler, event_types: List[EventType]):
    """특정 이벤트 타입 핸들러 등록 편의 함수"""
    event_bus = get_event_bus()
    event_bus.subscribe(handler, event_types)


# === 이벤트 시스템 상태 모니터링 ===


class EventSystemMonitor:
    """이벤트 시스템 상태 모니터링"""

    def __init__(self):
        self.monitoring_active = False

    def start_monitoring(self):
        """모니터링 시작"""
        self.monitoring_active = True
        logger.info("이벤트 시스템 모니터링 시작")

    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        logger.info("이벤트 시스템 모니터링 중지")

    def get_health_report(self) -> Dict[str, Any]:
        """시스템 건강도 리포트"""
        if _global_event_bus is None:
            return {"status": "not_initialized"}

        stats = _global_event_bus.get_statistics()

        # 건강도 평가
        health_indicators = {
            "queue_congestion": max(stats["queue_sizes"].values()) < 1000,
            "processing_speed": stats["average_processing_time_ms"] < 100,
            "handler_reliability": True,  # 핸들러 성능 기반 계산 가능
            "error_rate": True,  # 오류율 기반 계산 가능
        }

        overall_health = sum(health_indicators.values()) / len(health_indicators)

        return {
            "status": "healthy"
            if overall_health > 0.8
            else "degraded"
            if overall_health > 0.5
            else "critical",
            "overall_health_score": overall_health,
            "indicators": health_indicators,
            "statistics": stats,
            "recommendations": self._generate_recommendations(stats, health_indicators),
        }

    def _generate_recommendations(self, stats: Dict, indicators: Dict) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []

        if not indicators["queue_congestion"]:
            recommendations.append(
                "이벤트 큐 정체가 발생하고 있습니다. 핸들러 성능을 최적화하거나 큐 크기를 증가시키세요."
            )

        if not indicators["processing_speed"]:
            recommendations.append(
                "이벤트 처리 속도가 느립니다. 핸들러 로직을 최적화하거나 병렬 처리를 증가시키세요."
            )

        if stats["total_events"] > 10000 and stats["average_processing_time_ms"] > 50:
            recommendations.append(
                "높은 이벤트 볼륨과 처리 시간이 감지되었습니다. 시스템 리소스를 점검하세요."
            )

        return recommendations


# 전역 모니터 인스턴스
_global_monitor = EventSystemMonitor()


def get_event_system_monitor() -> EventSystemMonitor:
    """이벤트 시스템 모니터 반환"""
    return _global_monitor
