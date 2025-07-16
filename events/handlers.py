"""
Event Handlers - 이벤트 기반 시스템의 반응 메커니즘

이 모듈은 DMS 시스템에서 발생하는 다양한 이벤트들을 감지하고
적절한 대응을 수행하는 핸들러들을 구현합니다.

비유: 자동차의 경고등 시스템
- 엔진 과열 감지 → 경고등 점등 + 팬 가동
- 연료 부족 감지 → 경고음 + 주유소 안내
- 브레이크 문제 감지 → 경고등 + 응급 브레이크 준비

마찬가지로 DMS에서:
- 졸음 감지 → 경고음 + 진동 + 카페인 휴식 제안
- 주의산만 감지 → 시각 경고 + 자동 알림 차단
- 시스템 과부하 감지 → 분석 모드 축소 + 성능 최적화
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from config.settings import get_config
from core.constants import AnalysisConstants

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """이벤트 우선순위 - 응급실의 트리아지와 같은 개념"""

    LOW = 1  # 정보성 이벤트 (예: 일반적인 분석 완료)
    MEDIUM = 2  # 주의 필요 (예: 가벼운 졸음 징후)
    HIGH = 3  # 경고 필요 (예: 주의산만 감지)
    CRITICAL = 4  # 즉시 대응 (예: 심각한 졸음 상태)
    EMERGENCY = 5  # 응급상황 (예: 시스템 전체 실패)


class EventType(Enum):
    """이벤트 타입 정의 - 자동차 계기판의 다양한 경고등들"""

    # === 안전 관련 이벤트 ===
    DROWSINESS_DETECTED = "drowsiness_detected"
    CRITICAL_DROWSINESS = "critical_drowsiness"
    DISTRACTION_DETECTED = "distraction_detected"
    CRITICAL_DISTRACTION = "critical_distraction"
    DANGEROUS_BEHAVIOR = "dangerous_behavior"

    # === 시스템 관련 이벤트 ===
    FRAME_ANALYSIS_COMPLETE = "frame_analysis_complete"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SYSTEM_OVERLOAD = "system_overload"
    MODULE_FAILURE = "module_failure"

    # === 분석 품질 이벤트 ===
    LOW_CONFIDENCE_DETECTION = "low_confidence_detection"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    CALIBRATION_REQUIRED = "calibration_required"


@dataclass
class SafetyEvent:
    """
    안전 이벤트 데이터 구조

    마치 의료진이 환자 상태를 기록하는 차트와 같습니다:
    - 언제 발생했나? (timestamp)
    - 무엇이 문제인가? (event_type)
    - 얼마나 심각한가? (priority)
    - 어떤 증거가 있나? (data)
    - 어디서 발견되었나? (source)
    """

    event_type: EventType
    priority: EventPriority
    timestamp: datetime
    data: Dict[str, Any]
    source: str
    confidence: float = 1.0
    resolved: bool = False


class IEventHandler(ABC):
    """이벤트 핸들러 인터페이스 - 모든 핸들러가 따라야 할 공통 규칙"""

    @abstractmethod
    async def handle_event(self, event: SafetyEvent) -> bool:
        """이벤트를 처리하고 성공 여부를 반환"""
        pass

    @abstractmethod
    def can_handle(self, event_type: EventType) -> bool:
        """이 핸들러가 해당 이벤트 타입을 처리할 수 있는지 확인"""
        pass

    @abstractmethod
    def get_handler_name(self) -> str:
        """핸들러 식별을 위한 이름 반환"""
        pass


class SafetyEventHandler(IEventHandler):
    """
    안전 이벤트 전담 핸들러

    비유: 자동차의 능동 안전 시스템
    - 차선 이탈 → 핸들 진동 + 경고음
    - 전방 장애물 → 자동 브레이크 준비
    - 운전자 졸음 → 경고 + 휴게소 안내

    이 핸들러는 운전자의 안전에 직접적으로 관련된 모든 이벤트를 담당합니다.
    """

    def __init__(self, alert_system=None):
        """
        Args:
            alert_system: 경고 시스템 (예: 음성 알림, 진동, 시각적 경고)
        """
        self.config = get_config()
        self.alert_system = alert_system

        # 이벤트 이력 추적 (최근 300개 이벤트 보관)
        self.event_history = deque(maxlen=300)

        # 이벤트별 발생 빈도 추적 (패턴 분석용)
        self.event_frequency = defaultdict(int)

        # 연속 발생 이벤트 추적 (false positive 방지)
        self.consecutive_events = defaultdict(int)

        # 최근 경고 시간 (중복 경고 방지)
        self.last_alert_times = defaultdict(datetime)

        # 경고 쿨다운 시간 설정 (초 단위)
        self.alert_cooldowns = {
            EventType.DROWSINESS_DETECTED: 10,  # 졸음: 10초 간격
            EventType.CRITICAL_DROWSINESS: 5,  # 심각한 졸음: 5초 간격
            EventType.DISTRACTION_DETECTED: 15,  # 주의산만: 15초 간격
            EventType.CRITICAL_DISTRACTION: 8,  # 심각한 주의산만: 8초 간격
            EventType.DANGEROUS_BEHAVIOR: 3,  # 위험 행동: 3초 간격
        }

        logger.info("SafetyEventHandler 초기화 완료 - 안전 모니터링 시작")

    def can_handle(self, event_type: EventType) -> bool:
        """안전 관련 이벤트만 처리"""
        safety_events = {
            EventType.DROWSINESS_DETECTED,
            EventType.CRITICAL_DROWSINESS,
            EventType.DISTRACTION_DETECTED,
            EventType.CRITICAL_DISTRACTION,
            EventType.DANGEROUS_BEHAVIOR,
        }
        return event_type in safety_events

    def get_handler_name(self) -> str:
        return "SafetyEventHandler"

    async def handle_event(self, event: SafetyEvent) -> bool:
        """
        안전 이벤트 처리 메인 로직

        처리 단계:
        1. 이벤트 유효성 검증
        2. 중복/연속 이벤트 필터링
        3. 심각도별 대응 전략 결정
        4. 적절한 경고 시스템 활성화
        5. 이력 기록 및 패턴 분석
        """
        try:
            if not self.can_handle(event.event_type):
                return False

            # 1. 이벤트 기록 (패턴 분석을 위해)
            self.event_history.append(event)
            self.event_frequency[event.event_type] += 1

            # 2. 중복 경고 방지 검사
            if self._is_duplicate_alert(event):
                logger.debug(f"중복 경고 필터링: {event.event_type.value}")
                return True  # 처리했지만 경고는 생략

            # 3. 연속 발생 검증 (false positive 방지)
            if not self._validate_consecutive_occurrence(event):
                logger.debug(f"연속 발생 검증 실패: {event.event_type.value}")
                return True

            # 4. 심각도별 대응 실행
            success = await self._execute_safety_response(event)

            # 5. 경고 시간 기록
            self.last_alert_times[event.event_type] = event.timestamp

            # 6. 패턴 분석 (향후 예방을 위해)
            await self._analyze_safety_patterns()

            return success

        except Exception as e:
            logger.error(f"안전 이벤트 처리 중 오류: {e}")
            return False

    def _is_duplicate_alert(self, event: SafetyEvent) -> bool:
        """중복 경고 검사 - 같은 종류의 경고가 너무 자주 발생하는 것을 방지"""
        last_alert = self.last_alert_times.get(event.event_type)
        if last_alert is None:
            return False

        cooldown_seconds = self.alert_cooldowns.get(event.event_type, 10)
        time_since_last = (event.timestamp - last_alert).total_seconds()

        return time_since_last < cooldown_seconds

    def _validate_consecutive_occurrence(self, event: SafetyEvent) -> bool:
        """
        연속 발생 검증 - false positive 방지

        비유: 자동차 경고음이 잠깐 울렸다고 바로 정비소에 가지 않는 것처럼,
        일정 횟수 이상 연속으로 감지되어야 실제 경고로 인정합니다.
        """
        # 심각한 이벤트는 즉시 처리
        if (
            event.priority == EventPriority.CRITICAL
            or event.priority == EventPriority.EMERGENCY
        ):
            return True

        # 일반 이벤트는 연속 2-3회 발생시에만 처리
        self.consecutive_events[event.event_type] += 1

        required_consecutive = 2  # 기본 2회
        if event.event_type == EventType.DROWSINESS_DETECTED:
            required_consecutive = 3  # 졸음은 3회 연속 (더 신중하게)

        return self.consecutive_events[event.event_type] >= required_consecutive

    async def _execute_safety_response(self, event: SafetyEvent) -> bool:
        """
        심각도별 안전 대응 실행

        비유: 병원의 응급 대응 프로토콜
        - GREEN (낮음): 모니터링 지속
        - YELLOW (중간): 주의 경고
        - ORANGE (높음): 즉시 경고 + 권고사항
        - RED (매우 높음): 긴급 대응 + 자동 조치
        """
        response_success = True

        try:
            if event.priority == EventPriority.EMERGENCY:
                response_success &= await self._handle_emergency_response(event)

            elif event.priority == EventPriority.CRITICAL:
                response_success &= await self._handle_critical_response(event)

            elif event.priority == EventPriority.HIGH:
                response_success &= await self._handle_high_priority_response(event)

            elif event.priority == EventPriority.MEDIUM:
                response_success &= await self._handle_medium_priority_response(event)

            else:  # LOW priority
                response_success &= await self._handle_low_priority_response(event)

            logger.info(
                f"안전 대응 완료: {event.event_type.value} (성공: {response_success})"
            )
            return response_success

        except Exception as e:
            logger.error(f"안전 대응 실행 중 오류: {e}")
            return False

    async def _handle_emergency_response(self, event: SafetyEvent) -> bool:
        """응급상황 대응 - 가장 강력한 경고"""
        logger.critical(f"응급상황 감지: {event.event_type.value}")

        # 모든 가능한 경고 수단 동원
        if self.alert_system:
            # 강력한 음성 경고
            await self.alert_system.play_urgent_audio_alert()
            # 화면 전체 경고
            await self.alert_system.show_critical_visual_alert()
            # 진동 패드 최대 강도
            await self.alert_system.activate_tactile_alert(intensity=100)

        return True

    async def _handle_critical_response(self, event: SafetyEvent) -> bool:
        """심각한 상황 대응"""
        logger.warning(f"심각한 안전 이벤트: {event.event_type.value}")

        if self.alert_system:
            if event.event_type == EventType.CRITICAL_DROWSINESS:
                await self.alert_system.play_drowsiness_alert()
                await self.alert_system.suggest_rest_break()

            elif event.event_type == EventType.CRITICAL_DISTRACTION:
                await self.alert_system.play_attention_alert()
                await self.alert_system.block_distracting_apps()

        return True

    async def _handle_high_priority_response(self, event: SafetyEvent) -> bool:
        """높은 우선순위 대응"""
        logger.warning(f"주의 필요: {event.event_type.value}")

        if self.alert_system:
            await self.alert_system.show_visual_warning()
            await self.alert_system.play_gentle_chime()

        return True

    async def _handle_medium_priority_response(self, event: SafetyEvent) -> bool:
        """중간 우선순위 대응"""
        logger.info(f"모니터링: {event.event_type.value}")

        if self.alert_system:
            await self.alert_system.show_status_indicator()

        return True

    async def _handle_low_priority_response(self, event: SafetyEvent) -> bool:
        """낮은 우선순위 대응"""
        logger.debug(f"정보 기록: {event.event_type.value}")
        # 로깅만 수행, 별도 경고 없음
        return True

    async def _analyze_safety_patterns(self):
        """
        안전 패턴 분석 - 빅데이터 분석의 축소판

        예를 들어:
        - 특정 시간대에 졸음 이벤트가 많이 발생 → 해당 시간 경계 강화
        - 특정 운전 패턴 후 주의산만 증가 → 예방적 알림 제공
        - 이벤트 빈도가 갑자기 증가 → 시스템 캘리브레이션 필요
        """
        try:
            # 최근 이벤트 패턴 분석 (최근 50개 이벤트)
            recent_events = (
                list(self.event_history)[-50:]
                if len(self.event_history) >= 50
                else list(self.event_history)
            )

            if len(recent_events) < 10:
                return  # 분석하기에 데이터가 부족

            # 이벤트 타입별 빈도 계산
            type_frequency = defaultdict(int)
            for event in recent_events:
                type_frequency[event.event_type] += 1

            # 비정상적으로 높은 빈도 감지
            for event_type, frequency in type_frequency.items():
                if frequency > 10:  # 최근 50개 중 10개 이상
                    logger.warning(
                        f"높은 빈도 감지: {event_type.value} ({frequency}회)"
                    )

                    # 자동 조치 제안
                    if event_type == EventType.DROWSINESS_DETECTED:
                        if self.alert_system:
                            await self.alert_system.suggest_extended_break()

                    elif event_type == EventType.DISTRACTION_DETECTED:
                        if self.alert_system:
                            await self.alert_system.suggest_focus_mode()

        except Exception as e:
            logger.error(f"안전 패턴 분석 중 오류: {e}")

    def get_safety_statistics(self) -> Dict[str, Any]:
        """안전 통계 정보 반환 - 대시보드용"""
        total_events = len(self.event_history)

        if total_events == 0:
            return {"total_events": 0, "status": "insufficient_data"}

        # 이벤트 타입별 통계
        type_stats = defaultdict(int)
        priority_stats = defaultdict(int)

        for event in self.event_history:
            type_stats[event.event_type.value] += 1
            priority_stats[event.priority.value] += 1

        # 최근 1시간 이벤트 수
        now = datetime.now()
        recent_events = [
            event
            for event in self.event_history
            if (now - event.timestamp).total_seconds() < 3600
        ]

        return {
            "total_events": total_events,
            "recent_1h_events": len(recent_events),
            "event_types": dict(type_stats),
            "priority_distribution": dict(priority_stats),
            "most_frequent_event": max(type_stats.items(), key=lambda x: x[1])[0]
            if type_stats
            else None,
            "average_confidence": np.mean(
                [event.confidence for event in self.event_history]
            )
            if self.event_history
            else 0.0,
        }


class AnalyticsEventHandler(IEventHandler):
    """
    분석 및 성능 이벤트 전담 핸들러

    비유: 자동차의 계기판과 진단 시스템
    - 연비 모니터링 → 운전 패턴 분석 + 효율 개선 제안
    - 엔진 성능 추적 → 정비 시기 예측 + 부품 교체 알림
    - 주행 거리 기록 → 정기 점검 스케줄링

    이 핸들러는 시스템의 성능, 데이터 품질, 분석 결과의 신뢰성 등을
    지속적으로 모니터링하고 최적화 제안을 제공합니다.
    """

    def __init__(self, analytics_engine=None):
        """
        Args:
            analytics_engine: 분석 엔진 (예: 메트릭 수집, 성능 분석)
        """
        self.config = get_config()
        self.analytics_engine = analytics_engine

        # 성능 메트릭 추적
        self.performance_history = deque(maxlen=1000)  # 최근 1000개 프레임
        self.quality_metrics = deque(maxlen=500)  # 최근 500개 품질 측정

        # 시스템 건강도 추적
        self.module_health = defaultdict(
            lambda: {"success_count": 0, "failure_count": 0}
        )

        # 데이터 품질 추적
        self.data_quality_history = deque(maxlen=200)

        # 성능 벤치마크 (기준값들)
        self.performance_benchmarks = {
            "target_fps": self.config.performance.target_fps,
            "max_processing_time": self.config.performance.max_processing_time_ms
            / 1000,  # 초 단위
            "min_confidence": 0.7,
            "max_failure_rate": 0.05,  # 5%
        }

        logger.info("AnalyticsEventHandler 초기화 완료 - 성능 모니터링 시작")

    def can_handle(self, event_type: EventType) -> bool:
        """시스템 및 분석 관련 이벤트 처리"""
        analytics_events = {
            EventType.FRAME_ANALYSIS_COMPLETE,
            EventType.PERFORMANCE_DEGRADATION,
            EventType.SYSTEM_OVERLOAD,
            EventType.MODULE_FAILURE,
            EventType.LOW_CONFIDENCE_DETECTION,
            EventType.DATA_QUALITY_ISSUE,
            EventType.CALIBRATION_REQUIRED,
        }
        return event_type in analytics_events

    def get_handler_name(self) -> str:
        return "AnalyticsEventHandler"

    async def handle_event(self, event: SafetyEvent) -> bool:
        """분석 이벤트 처리 메인 로직"""
        try:
            if not self.can_handle(event.event_type):
                return False

            # 이벤트 타입별 전문 처리
            if event.event_type == EventType.FRAME_ANALYSIS_COMPLETE:
                return await self._handle_frame_analysis_complete(event)

            elif event.event_type == EventType.PERFORMANCE_DEGRADATION:
                return await self._handle_performance_degradation(event)

            elif event.event_type == EventType.SYSTEM_OVERLOAD:
                return await self._handle_system_overload(event)

            elif event.event_type == EventType.MODULE_FAILURE:
                return await self._handle_module_failure(event)

            elif event.event_type == EventType.LOW_CONFIDENCE_DETECTION:
                return await self._handle_low_confidence(event)

            elif event.event_type == EventType.DATA_QUALITY_ISSUE:
                return await self._handle_data_quality_issue(event)

            elif event.event_type == EventType.CALIBRATION_REQUIRED:
                return await self._handle_calibration_required(event)

            return True

        except Exception as e:
            logger.error(f"분석 이벤트 처리 중 오류: {e}")
            return False

    async def _handle_frame_analysis_complete(self, event: SafetyEvent) -> bool:
        """프레임 분석 완료 이벤트 처리 - 성능 메트릭 수집"""

        # 성능 데이터 추출
        processing_time = event.data.get("processing_time", 0)
        confidence = event.data.get("confidence", 0)
        system_health = event.data.get("system_health", "unknown")

        # 성능 이력에 추가
        performance_record = {
            "timestamp": event.timestamp,
            "processing_time": processing_time,
            "confidence": confidence,
            "system_health": system_health,
        }
        self.performance_history.append(performance_record)

        # 실시간 성능 분석
        await self._analyze_real_time_performance()

        # 분석 엔진에 메트릭 전송
        if self.analytics_engine:
            await self.analytics_engine.update_performance_metrics(performance_record)

        return True

    async def _handle_performance_degradation(self, event: SafetyEvent) -> bool:
        """성능 저하 이벤트 처리"""

        degradation_type = event.data.get("degradation_type", "unknown")
        severity = event.data.get("severity", "medium")

        logger.warning(f"성능 저하 감지: {degradation_type} (심각도: {severity})")

        # 심각도별 대응
        if severity == "critical":
            # 긴급 최적화 모드 활성화
            if self.analytics_engine:
                await self.analytics_engine.enable_emergency_optimization()

        elif severity == "high":
            # 적응형 품질 조정
            if self.analytics_engine:
                await self.analytics_engine.adjust_quality_settings()

        # 성능 저하 패턴 분석
        await self._analyze_degradation_patterns()

        return True

    async def _handle_system_overload(self, event: SafetyEvent) -> bool:
        """시스템 과부하 이벤트 처리"""

        cpu_usage = event.data.get("cpu_usage", 0)
        memory_usage = event.data.get("memory_usage", 0)
        processing_queue_size = event.data.get("queue_size", 0)

        logger.warning(
            f"시스템 과부하: CPU {cpu_usage}%, Memory {memory_usage}%, Queue {processing_queue_size}"
        )

        # 자동 부하 경감 조치
        if self.analytics_engine:
            # 처리 품질 일시 조정
            await self.analytics_engine.reduce_processing_load()

            # 불필요한 모듈 일시 비활성화
            await self.analytics_engine.disable_non_critical_modules()

        return True

    async def _handle_module_failure(self, event: SafetyEvent) -> bool:
        """모듈 실패 이벤트 처리"""

        module_name = event.data.get("module_name", "unknown")
        error_type = event.data.get("error_type", "unknown")

        # 모듈 건강도 업데이트
        self.module_health[module_name]["failure_count"] += 1

        logger.error(f"모듈 실패: {module_name} ({error_type})")

        # 모듈별 복구 시도
        if self.analytics_engine:
            recovery_success = await self.analytics_engine.attempt_module_recovery(
                module_name
            )

            if recovery_success:
                logger.info(f"모듈 복구 성공: {module_name}")
                self.module_health[module_name]["success_count"] += 1
            else:
                logger.error(f"모듈 복구 실패: {module_name}")
                # 대체 모듈 활성화 시도
                await self.analytics_engine.activate_backup_module(module_name)

        return True

    async def _handle_low_confidence(self, event: SafetyEvent) -> bool:
        """낮은 신뢰도 감지 이벤트 처리"""

        confidence_score = event.data.get("confidence", 0)
        detection_type = event.data.get("detection_type", "unknown")

        # 품질 메트릭에 기록
        quality_record = {
            "timestamp": event.timestamp,
            "confidence": confidence_score,
            "detection_type": detection_type,
            "quality_issue": True,
        }
        self.quality_metrics.append(quality_record)

        # 지속적인 저신뢰도 패턴 감지
        recent_low_confidence = [
            record
            for record in list(self.quality_metrics)[-20:]  # 최근 20개
            if record.get("quality_issue", False)
        ]

        if len(recent_low_confidence) > 10:  # 20개 중 10개 이상이 문제
            logger.warning("지속적인 낮은 신뢰도 감지 - 캘리브레이션 권장")

            if self.analytics_engine:
                await self.analytics_engine.trigger_auto_calibration()

        return True

    async def _handle_data_quality_issue(self, event: SafetyEvent) -> bool:
        """데이터 품질 문제 이벤트 처리"""

        issue_type = event.data.get("issue_type", "unknown")
        affected_modules = event.data.get("affected_modules", [])

        logger.warning(
            f"데이터 품질 문제: {issue_type} (영향받는 모듈: {affected_modules})"
        )

        # 데이터 품질 이력에 기록
        quality_issue = {
            "timestamp": event.timestamp,
            "issue_type": issue_type,
            "affected_modules": affected_modules,
        }
        self.data_quality_history.append(quality_issue)

        # 영향받는 모듈에 대한 품질 보정
        if self.analytics_engine:
            for module in affected_modules:
                await self.analytics_engine.apply_quality_correction(module, issue_type)

        return True

    async def _handle_calibration_required(self, event: SafetyEvent) -> bool:
        """캘리브레이션 필요 이벤트 처리"""

        calibration_type = event.data.get("calibration_type", "general")
        priority = event.data.get("priority", "medium")

        logger.info(f"캘리브레이션 요청: {calibration_type} (우선순위: {priority})")

        if self.analytics_engine:
            if priority == "high":
                # 즉시 캘리브레이션 실행
                await self.analytics_engine.start_immediate_calibration(
                    calibration_type
                )
            else:
                # 대기열에 추가
                await self.analytics_engine.schedule_calibration(calibration_type)

        return True

    async def _analyze_real_time_performance(self):
        """실시간 성능 분석 - 최근 데이터를 기반으로 즉시 판단"""

        if len(self.performance_history) < 10:
            return  # 분석하기에 데이터 부족

        # 최근 30개 프레임의 평균 성능
        recent_records = list(self.performance_history)[-30:]
        avg_processing_time = np.mean([r["processing_time"] for r in recent_records])
        avg_confidence = np.mean([r["confidence"] for r in recent_records])

        # 성능 기준과 비교
        target_time = self.performance_benchmarks["max_processing_time"]
        min_confidence = self.performance_benchmarks["min_confidence"]

        # 성능 저하 감지
        if avg_processing_time > target_time * 1.5:  # 기준의 150% 초과
            await self._trigger_performance_alert(
                "processing_time_exceeded",
                {
                    "current": avg_processing_time,
                    "target": target_time,
                    "severity": "high"
                    if avg_processing_time > target_time * 2
                    else "medium",
                },
            )

        # 신뢰도 저하 감지
        if avg_confidence < min_confidence:
            await self._trigger_performance_alert(
                "confidence_below_threshold",
                {
                    "current": avg_confidence,
                    "target": min_confidence,
                    "severity": "medium",
                },
            )

    async def _analyze_degradation_patterns(self):
        """성능 저하 패턴 분석 - 머신러닝의 이상 탐지와 유사"""

        if len(self.performance_history) < 100:
            return

        # 처리 시간 추세 분석
        times = [
            record["processing_time"]
            for record in list(self.performance_history)[-100:]
        ]

        # 간단한 선형 추세 계산 (실제로는 더 복잡한 알고리즘 사용 가능)
        x = np.arange(len(times))
        z = np.polyfit(x, times, 1)  # 1차 다항식 피팅
        slope = z[0]

        # 증가 추세 감지 (성능 저하)
        if slope > 0.001:  # 초당 1ms 이상 증가
            logger.warning(f"성능 저하 추세 감지: 기울기 {slope:.4f}")

            if self.analytics_engine:
                await self.analytics_engine.schedule_optimization_analysis()

    async def _trigger_performance_alert(self, alert_type: str, data: Dict[str, Any]):
        """성능 경고 발생"""

        # 새로운 이벤트 생성하여 다시 이벤트 시스템에 전달
        performance_event = SafetyEvent(
            event_type=EventType.PERFORMANCE_DEGRADATION,
            priority=EventPriority.HIGH
            if data.get("severity") == "high"
            else EventPriority.MEDIUM,
            timestamp=datetime.now(),
            data={"degradation_type": alert_type, **data},
            source="AnalyticsEventHandler",
        )

        # 자기 자신에게 이벤트 전달 (재귀적 처리)
        await self.handle_event(performance_event)

    def get_analytics_statistics(self) -> Dict[str, Any]:
        """분석 통계 정보 반환"""

        if not self.performance_history:
            return {"status": "no_data"}

        recent_performance = list(self.performance_history)[-50:]  # 최근 50개

        # 기본 통계
        avg_processing_time = np.mean(
            [r["processing_time"] for r in recent_performance]
        )
        avg_confidence = np.mean([r["confidence"] for r in recent_performance])

        # 모듈 건강도 계산
        module_health_scores = {}
        for module, health in self.module_health.items():
            total = health["success_count"] + health["failure_count"]
            if total > 0:
                module_health_scores[module] = health["success_count"] / total

        # 데이터 품질 점수
        recent_quality = (
            list(self.quality_metrics)[-20:] if self.quality_metrics else []
        )
        quality_score = (
            np.mean([r["confidence"] for r in recent_quality])
            if recent_quality
            else 0.0
        )

        return {
            "performance": {
                "avg_processing_time_ms": avg_processing_time * 1000,
                "avg_confidence": avg_confidence,
                "target_fps": self.performance_benchmarks["target_fps"],
                "current_fps": 1.0 / avg_processing_time
                if avg_processing_time > 0
                else 0,
            },
            "module_health": module_health_scores,
            "data_quality_score": quality_score,
            "total_frames_processed": len(self.performance_history),
            "system_status": "healthy"
            if avg_processing_time < self.performance_benchmarks["max_processing_time"]
            else "degraded",
        }


# === 이벤트 시스템 글로벌 인터페이스 ===

# 전역 핸들러 레지스트리
_global_handlers: List[IEventHandler] = []
_event_queue = asyncio.Queue()
_event_processor_task = None
_shutdown_event = asyncio.Event()  # 종료 신호 이벤트 추가


async def register_handler(handler: IEventHandler):
    """핸들러를 전역 레지스트리에 등록"""
    global _global_handlers
    _global_handlers.append(handler)
    logger.info(f"이벤트 핸들러 등록: {handler.get_handler_name()}")


async def publish_event(event: SafetyEvent):
    """이벤트를 발행하여 모든 관련 핸들러에 전달"""
    global _event_queue
    # 종료 신호 확인
    if _shutdown_event.is_set():
        logger.warning("시스템 종료 중이므로 이벤트 발행 차단")
        return
    await _event_queue.put(event)


async def _process_events():
    """백그라운드에서 이벤트 큐를 처리하는 워커 - 종료 신호 지원"""
    global _event_queue, _global_handlers, _shutdown_event

    while not _shutdown_event.is_set():
        try:
            # 종료 신호와 함께 타임아웃 설정으로 이벤트 대기
            try:
                event = await asyncio.wait_for(_event_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # 타임아웃 발생 시 종료 신호 다시 확인
                continue

            # 종료 신호가 설정되었다면 처리 중단
            if _shutdown_event.is_set():
                logger.info("이벤트 처리 중단 - 종료 신호 감지")
                break

            # 적절한 핸들러들에게 전달
            for handler in _global_handlers:
                if handler.can_handle(event.event_type):
                    try:
                        success = await handler.handle_event(event)
                        if not success:
                            logger.warning(
                                f"핸들러 처리 실패: {handler.get_handler_name()} for {event.event_type.value}"
                            )
                    except Exception as e:
                        logger.error(
                            f"핸들러 실행 중 오류: {handler.get_handler_name()}: {e}"
                        )

            _event_queue.task_done()

        except asyncio.CancelledError:
            logger.info("이벤트 처리 작업 취소됨")
            break
        except Exception as e:
            logger.error(f"이벤트 처리 중 오류: {e}")
            # 심각한 오류 발생 시 짧은 대기 후 재시도
            await asyncio.sleep(0.1)

    logger.info("이벤트 처리 루프 종료")


async def start_event_processing():
    """이벤트 처리 시스템 시작"""
    global _event_processor_task, _shutdown_event
    if _event_processor_task is None:
        _shutdown_event.clear()  # 종료 신호 초기화
        _event_processor_task = asyncio.create_task(_process_events())
        logger.info("이벤트 처리 시스템 시작됨")


async def stop_event_processing():
    """이벤트 처리 시스템 종료 - 개선된 종료 처리"""
    global _event_processor_task, _shutdown_event

    if _event_processor_task:
        # 종료 신호 설정
        _shutdown_event.set()
        logger.info("이벤트 처리 시스템 종료 신호 전송")

        # 작업 완료 대기 (최대 5초)
        try:
            await asyncio.wait_for(_event_processor_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("이벤트 처리 시스템 종료 타임아웃 - 강제 취소")
            _event_processor_task.cancel()
            try:
                await _event_processor_task
            except asyncio.CancelledError:
                pass

        _event_processor_task = None
        logger.info("이벤트 처리 시스템 종료 완료")

    # 대기 중인 이벤트 정리
    try:
        while not _event_queue.empty():
            _event_queue.get_nowait()
            _event_queue.task_done()
    except asyncio.QueueEmpty:
        pass


# === 편의 함수들 ===


async def publish_safety_event(
    event_type: EventType,
    data: Dict[str, Any],
    source: str = "unknown",
    priority: EventPriority = EventPriority.MEDIUM,
):
    """안전 이벤트 발행 편의 함수"""
    event = SafetyEvent(
        event_type=event_type,
        priority=priority,
        timestamp=datetime.now(),
        data=data,
        source=source,
    )
    await publish_event(event)


async def publish_analytics_event(
    event_type: EventType, data: Dict[str, Any], source: str = "unknown"
):
    """분석 이벤트 발행 편의 함수"""
    event = SafetyEvent(
        event_type=event_type,
        priority=EventPriority.LOW,
        timestamp=datetime.now(),
        data=data,
        source=source,
    )
    await publish_event(event)


# numpy import 추가
import numpy as np
