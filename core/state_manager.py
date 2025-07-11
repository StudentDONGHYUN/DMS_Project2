import time
from collections import deque
import logging
from typing import Dict, Any, Optional
from core.definitions import DriverState, AnalysisEvent

logger = logging.getLogger(__name__)


class StateManager:
    """기본 상태 관리자 (새로운 시스템용)"""
    
    def __init__(self):
        self.current_state = DriverState.SAFE
        self.state_start_time = time.time()
        self.state_history = deque(maxlen=100)
        
        # 메트릭 관리자와의 연결 (선택적)
        self.metrics_manager = None
        
    def update_state(self, driver_state_data: Dict[str, Any], timestamp: float) -> None:
        """운전자 상태 데이터를 기반으로 상태 업데이트"""
        fatigue_level = driver_state_data.get('fatigue_level', 0.0)
        distraction_level = driver_state_data.get('distraction_level', 0.0)
        
        # 상태 결정 로직
        new_state = self._determine_state_from_levels(fatigue_level, distraction_level)
        
        if new_state != self.current_state:
            self._transition_to_state(new_state, timestamp)
    
    def _determine_state_from_levels(self, fatigue: float, distraction: float) -> DriverState:
        """피로도와 주의산만 수준에 따른 상태 결정"""
        if fatigue > 0.8 or distraction > 0.8:
            return DriverState.MULTIPLE_RISK
        elif fatigue > 0.6:
            return DriverState.FATIGUE_HIGH
        elif fatigue > 0.3:
            return DriverState.FATIGUE_LOW
        elif distraction > 0.6:
            return DriverState.DISTRACTION_DANGER
        elif distraction > 0.3:
            return DriverState.DISTRACTION_NORMAL
        else:
            return DriverState.SAFE
    
    def _transition_to_state(self, new_state: DriverState, timestamp: float) -> None:
        """상태 전환 처리"""
        old_state = self.current_state
        
        self.state_history.append({
            "timestamp": timestamp,
            "from_state": old_state,
            "to_state": new_state,
            "duration": timestamp - self.state_start_time
        })
        
        logger.info(f"상태 전환: {old_state.value} -> {new_state.value}")
        
        self.current_state = new_state
        self.state_start_time = timestamp
    
    def handle_alert(self, alert_type: str, severity: str, value: float) -> None:
        """MetricsManager로부터의 경고 처리"""
        logger.warning(f"경고 수신: {alert_type} - {severity} (값: {value:.3f})")
        
        # 경고에 따른 상태 조정 로직 (필요시 구현)
        pass
    
    def update_trend_analysis(self, metric_name: str, trend_analysis) -> None:
        """MetricsManager로부터의 트렌드 분석 정보 수신"""
        logger.debug(f"트렌드 분석 수신: {metric_name} - {trend_analysis.trend_direction}")
    
    def set_metrics_manager(self, metrics_manager) -> None:
        """MetricsManager와 연결"""
        self.metrics_manager = metrics_manager
        logger.info("MetricsManager와 연결됨")
    
    def get_current_state(self) -> DriverState:
        """현재 상태 반환"""
        return self.current_state
    
    def get_state_duration(self) -> float:
        """현재 상태 지속 시간 반환"""
        return time.time() - self.state_start_time

class EnhancedStateManager:
    """향상된 상태 관리자"""

    def __init__(self):
        self.current_state = DriverState.SAFE
        self.state_start_time = time.time()
        self.state_history = deque(maxlen=100)

    def handle_event(self, event: AnalysisEvent):
        new_state = self._determine_enhanced_new_state(event)
        if new_state != self.current_state:
            self.state_history.append(
                {
                    "timestamp": time.time(),
                    "from_state": self.current_state,
                    "to_state": new_state,
                    "trigger_event": event,
                }
            )
            logger.info(
                f"상태 전환: {self.current_state.value} -> {new_state.value} (이벤트: {event.value})"
            )
            self.current_state = new_state
            self.state_start_time = time.time()

    def _determine_enhanced_new_state(self, event: AnalysisEvent) -> DriverState:
        current_duration = time.time() - self.state_start_time
        immediate_transitions = {
            AnalysisEvent.PHONE_USAGE_CONFIRMED: DriverState.PHONE_USAGE,
            AnalysisEvent.MICROSLEEP_PREDICTED: DriverState.MICROSLEEP,
            AnalysisEvent.EMOTION_STRESS_DETECTED: DriverState.EMOTIONAL_STRESS,
            AnalysisEvent.PREDICTIVE_RISK_HIGH: DriverState.PREDICTIVE_WARNING,
        }
        if event in immediate_transitions:
            return immediate_transitions[event]
        if event == AnalysisEvent.FATIGUE_ACCUMULATION:
            if self.current_state == DriverState.FATIGUE_LOW:
                return DriverState.FATIGUE_HIGH
            else:
                return DriverState.FATIGUE_LOW
        if event == AnalysisEvent.ATTENTION_DECLINE:
            if self.current_state == DriverState.DISTRACTION_NORMAL:
                return DriverState.DISTRACTION_DANGER
            else:
                return DriverState.DISTRACTION_NORMAL
        if event == AnalysisEvent.DISTRACTION_OBJECT_DETECTED:
            if self.current_state in [DriverState.FATIGUE_HIGH, DriverState.EMOTIONAL_STRESS]:
                return DriverState.MULTIPLE_RISK
            else:
                return DriverState.DISTRACTION_DANGER
        if event == AnalysisEvent.NORMAL_BEHAVIOR:
            if current_duration > 5.0:
                return DriverState.SAFE
        return self.current_state

    def get_current_state(self) -> DriverState:
        return self.current_state

    def get_state_duration(self) -> float:
        return time.time() - self.state_start_time

    def get_state_statistics(self) -> dict:
        if not self.state_history:
            return {}
        state_counts = {}
        for entry in self.state_history:
            state = entry["to_state"]
            state_counts[state] = state_counts.get(state, 0) + 1
        return {
            "state_counts": state_counts,
            "current_duration": self.get_state_duration(),
            "total_transitions": len(self.state_history),
        }
