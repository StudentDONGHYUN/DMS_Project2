"""
메트릭 관리 시스템
시스템의 모든 메트릭을 중앙에서 수집, 관리, 분석하는 핵심 컴포넌트

이 클래스는 마치 자동차의 종합 계기판과 같은 역할을 하며,
모든 분석 결과를 통합하여 운전자의 상태를 종합적으로 평가합니다.

주요 기능:
1. 멀티모달 메트릭 통합 관리 (얼굴, 자세, 손동작, 객체)
2. 실시간 상태 분석 및 트렌드 감지
3. 이벤트 기반 분석 엔진 역할
4. 상태 관리자와의 양방향 통신
5. 예측적 위험 평가 및 경고 시스템
"""

import time
import logging
from collections import deque, defaultdict
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np

# 시스템 핵심 모듈 임포트
from core.interfaces import IMetricsUpdater, IAdvancedMetricsUpdater
from core.definitions import AdvancedMetrics, DriverState, EmotionState
from config.settings import get_config

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """특정 시점의 메트릭 스냅샷"""
    timestamp: float
    drowsiness_score: float = 0.0
    distraction_score: float = 0.0
    emotion_state: Optional[EmotionState] = None
    gaze_focus: float = 0.0
    prediction_risk: float = 0.0
    confidence: float = 0.0
    
    # 고급 메트릭들
    saccade_frequency: float = 0.0
    pupil_diameter: float = 0.0
    heart_rate: float = 0.0
    cognitive_load: float = 0.0


@dataclass
class TrendAnalysis:
    """트렌드 분석 결과"""
    metric_name: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0.0 ~ 1.0
    prediction_window: float  # 예측 윈도우 (초)
    confidence: float


class MetricsManager(IAdvancedMetricsUpdater):
    """
    중앙화된 메트릭 관리 시스템
    
    이 클래스는 DMS 시스템의 모든 분석 결과를 통합 관리하며,
    실시간으로 운전자 상태를 평가하고 위험을 예측합니다.
    
    아키텍처 설계:
    - 싱글톤 패턴: 시스템 전반에서 하나의 메트릭 저장소 사용
    - 관찰자 패턴: 상태 변화시 관련 컴포넌트들에 알림
    - 시간 윈도우 기반: 과거 데이터를 이용한 트렌드 분석
    - 신뢰도 기반: 각 메트릭의 신뢰도를 고려한 가중 평균
    """
    
    def __init__(self, history_size: int = 900):  # 30초 @ 30fps
        """
        메트릭 관리자 초기화
        
        Args:
            history_size: 메트릭 히스토리 버퍼 크기
        """
        self.config = get_config()
        self.history_size = history_size
        
        # === 메트릭 저장소 ===
        self.metric_history = deque(maxlen=history_size)
        self.current_metrics = self._create_default_metrics()
        
        # === 개별 메트릭 버퍼들 ===
        self.drowsiness_buffer = deque(maxlen=150)  # 5초
        self.emotion_buffer = deque(maxlen=150)     # 5초  
        self.gaze_buffer = deque(maxlen=90)         # 3초
        self.distraction_buffer = deque(maxlen=90)  # 3초
        self.prediction_buffer = deque(maxlen=300)  # 10초
        
        # === 고급 메트릭 버퍼들 ===
        self.saccade_buffer = deque(maxlen=300)     # 10초
        self.pupil_buffer = deque(maxlen=300)       # 10초
        self.rppg_buffer = deque(maxlen=900)        # 30초 (심박수 분석용)
        self.cognitive_buffer = deque(maxlen=150)   # 5초
        
        # === 상태 관리 ===
        self.state_manager = None  # StateManager와 연결
        self.last_update_time = time.time()
        self.analysis_count = 0
        
        # === 트렌드 분석 ===
        self.trend_analyzers = {}
        self._initialize_trend_analyzers()
        
        # === 경고 시스템 ===
        self.alert_thresholds = self._load_alert_thresholds()
        self.last_alerts = {}
        
        logger.info(f"MetricsManager 초기화 완료 - 히스토리 크기: {history_size}")
    
    def _create_default_metrics(self) -> AdvancedMetrics:
        """기본 메트릭 객체 생성"""
        return AdvancedMetrics(
            # 기본 메트릭들은 core.definitions에 정의된 구조를 따름
            # 여기서는 기본값들로 초기화
        )
    
    def _initialize_trend_analyzers(self):
        """트렌드 분석기들 초기화"""
        metrics_to_analyze = [
            'drowsiness_score', 'distraction_score', 'gaze_focus',
            'heart_rate', 'cognitive_load', 'saccade_frequency'
        ]
        
        for metric in metrics_to_analyze:
            self.trend_analyzers[metric] = {
                'window_size': 150,  # 5초 윈도우
                'min_samples': 30,   # 최소 샘플 수
                'sensitivity': 0.1   # 변화 감지 민감도
            }
    
    def _load_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """경고 임계값 로딩"""
        return {
            'drowsiness': {
                'warning': 0.4,
                'danger': 0.7,
                'critical': 0.9
            },
            'distraction': {
                'warning': 0.3,
                'danger': 0.6,
                'critical': 0.8
            },
            'cognitive_load': {
                'warning': 0.5,
                'danger': 0.7,
                'critical': 0.9
            }
        }
    
    # === IMetricsUpdater 인터페이스 구현 ===
    
    def update_drowsiness_metrics(self, drowsiness_data: Dict[str, Any]) -> None:
        """졸음 관련 메트릭 업데이트"""
        timestamp = time.time()
        
        # 졸음 점수 추출 및 검증
        drowsiness_score = max(0.0, min(1.0, drowsiness_data.get('fatigue_score', 0.0)))
        confidence = drowsiness_data.get('confidence', 0.5)
        
        # 버퍼에 저장
        self.drowsiness_buffer.append({
            'timestamp': timestamp,
            'score': drowsiness_score,
            'confidence': confidence,
            'ear_value': drowsiness_data.get('ear_value', 0.0),
            'perclos': drowsiness_data.get('perclos', 0.0),
            'microsleep_detected': drowsiness_data.get('microsleep_detected', False)
        })
        
        # 현재 메트릭 업데이트
        self.current_metrics.drowsiness_score = drowsiness_score
        self.current_metrics.drowsiness_confidence = confidence
        
        # 경고 체크
        self._check_drowsiness_alerts(drowsiness_score)
        
        logger.debug(f"졸음 메트릭 업데이트 - 점수: {drowsiness_score:.3f}, 신뢰도: {confidence:.3f}")
    
    def update_emotion_metrics(self, emotion_data: Dict[str, Any]) -> None:
        """감정 관련 메트릭 업데이트"""
        timestamp = time.time()
        
        # 감정 상태 추출
        dominant_emotion = emotion_data.get('dominant_emotion', 'neutral')
        arousal_level = emotion_data.get('arousal_level', 0.5)
        stress_level = emotion_data.get('stress_level', 0.0)
        confidence = emotion_data.get('confidence', 0.5)
        
        # 버퍼에 저장
        self.emotion_buffer.append({
            'timestamp': timestamp,
            'emotion': dominant_emotion,
            'arousal': arousal_level,
            'stress': stress_level,
            'confidence': confidence
        })
        
        # 현재 메트릭 업데이트
        self.current_metrics.dominant_emotion = dominant_emotion
        self.current_metrics.arousal_level = arousal_level
        self.current_metrics.stress_level = stress_level
        
        logger.debug(f"감정 메트릭 업데이트 - 감정: {dominant_emotion}, 각성: {arousal_level:.3f}")
    
    def update_gaze_metrics(self, gaze_data: Dict[str, Any]) -> None:
        """시선 관련 메트릭 업데이트"""
        timestamp = time.time()
        
        # 시선 메트릭 추출
        gaze_zone = gaze_data.get('primary_zone', 'UNKNOWN')
        focus_score = gaze_data.get('attention_focus', 0.0)
        stability = gaze_data.get('gaze_stability', 0.0)
        off_road_duration = gaze_data.get('off_road_duration', 0.0)
        
        # 버퍼에 저장
        self.gaze_buffer.append({
            'timestamp': timestamp,
            'zone': gaze_zone,
            'focus': focus_score,
            'stability': stability,
            'off_road_duration': off_road_duration
        })
        
        # 현재 메트릭 업데이트
        self.current_metrics.primary_gaze_zone = gaze_zone
        self.current_metrics.attention_focus = focus_score
        self.current_metrics.gaze_stability = stability
        
        logger.debug(f"시선 메트릭 업데이트 - 구역: {gaze_zone}, 집중도: {focus_score:.3f}")
    
    def update_distraction_metrics(self, distraction_data: Dict[str, Any]) -> None:
        """주의산만 관련 메트릭 업데이트"""
        timestamp = time.time()
        
        # 주의산만 메트릭 추출
        distraction_score = max(0.0, min(1.0, distraction_data.get('distraction_risk_score', 0.0)))
        phone_detected = distraction_data.get('phone_detected', False)
        hand_stability = distraction_data.get('hand_stability_score', 1.0)
        
        # 버퍼에 저장
        self.distraction_buffer.append({
            'timestamp': timestamp,
            'score': distraction_score,
            'phone_detected': phone_detected,
            'hand_stability': hand_stability,
            'left_hand_safe': distraction_data.get('left_hand_in_safe_zone', True),
            'right_hand_safe': distraction_data.get('right_hand_in_safe_zone', True)
        })
        
        # 현재 메트릭 업데이트
        self.current_metrics.distraction_score = distraction_score
        self.current_metrics.phone_detected = phone_detected
        self.current_metrics.hand_stability = hand_stability
        
        # 경고 체크
        self._check_distraction_alerts(distraction_score)
        
        logger.debug(f"주의산만 메트릭 업데이트 - 점수: {distraction_score:.3f}, 휴대폰: {phone_detected}")
    
    def update_prediction_metrics(self, prediction_data: Dict[str, Any]) -> None:
        """예측 관련 메트릭 업데이트"""
        timestamp = time.time()
        
        # 예측 메트릭 추출
        risk_score = prediction_data.get('risk_score', 0.0)
        risk_factors = prediction_data.get('risk_factors', [])
        prediction_confidence = prediction_data.get('confidence', 0.5)
        time_horizon = prediction_data.get('time_horizon', 30.0)
        
        # 버퍼에 저장
        self.prediction_buffer.append({
            'timestamp': timestamp,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'confidence': prediction_confidence,
            'time_horizon': time_horizon
        })
        
        # 현재 메트릭 업데이트
        self.current_metrics.prediction_risk = risk_score
        self.current_metrics.prediction_confidence = prediction_confidence
        
        logger.debug(f"예측 메트릭 업데이트 - 위험도: {risk_score:.3f}, 요소: {len(risk_factors)}개")
    
    def get_current_metrics(self) -> AdvancedMetrics:
        """현재 메트릭 반환"""
        # 메트릭 스냅샷을 히스토리에 저장
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            drowsiness_score=self.current_metrics.drowsiness_score,
            distraction_score=self.current_metrics.distraction_score,
            gaze_focus=self.current_metrics.attention_focus,
            prediction_risk=self.current_metrics.prediction_risk,
            confidence=self._calculate_overall_confidence()
        )
        self.metric_history.append(snapshot)
        
        # 트렌드 분석 수행
        self._update_trend_analysis()
        
        return self.current_metrics
    
    # === IAdvancedMetricsUpdater 인터페이스 구현 ===
    
    def update_saccade_metrics(self, saccade_data: Dict[str, Any]) -> None:
        """사케이드 메트릭 업데이트"""
        timestamp = time.time()
        
        saccade_frequency = saccade_data.get('frequency', 0.0)
        saccade_amplitude = saccade_data.get('amplitude', 0.0)
        fixation_stability = saccade_data.get('fixation_stability', 1.0)
        
        self.saccade_buffer.append({
            'timestamp': timestamp,
            'frequency': saccade_frequency,
            'amplitude': saccade_amplitude,
            'stability': fixation_stability
        })
        
        # 현재 메트릭 업데이트
        self.current_metrics.saccade_frequency = saccade_frequency
        
        logger.debug(f"사케이드 메트릭 업데이트 - 빈도: {saccade_frequency:.3f}")
    
    def update_pupil_metrics(self, pupil_data: Dict[str, Any]) -> None:
        """동공 메트릭 업데이트"""
        timestamp = time.time()
        
        pupil_diameter = pupil_data.get('diameter', 0.0)
        pupil_response = pupil_data.get('light_response', 1.0)
        
        self.pupil_buffer.append({
            'timestamp': timestamp,
            'diameter': pupil_diameter,
            'response': pupil_response
        })
        
        # 현재 메트릭 업데이트  
        self.current_metrics.pupil_diameter = pupil_diameter
        
        logger.debug(f"동공 메트릭 업데이트 - 직경: {pupil_diameter:.3f}")
    
    def update_rppg_metrics(self, rppg_data: Dict[str, Any]) -> None:
        """rPPG 메트릭 업데이트"""
        timestamp = time.time()
        
        heart_rate = rppg_data.get('heart_rate', 0.0)
        hrv_score = rppg_data.get('hrv_score', 0.0)
        signal_quality = rppg_data.get('signal_quality', 0.0)
        
        self.rppg_buffer.append({
            'timestamp': timestamp,
            'heart_rate': heart_rate,
            'hrv': hrv_score,
            'quality': signal_quality
        })
        
        # 현재 메트릭 업데이트
        self.current_metrics.heart_rate = heart_rate
        self.current_metrics.hrv_score = hrv_score
        
        logger.debug(f"rPPG 메트릭 업데이트 - 심박수: {heart_rate:.1f}")
    
    def update_cognitive_load_metrics(self, cognitive_data: Dict[str, Any]) -> None:
        """인지 부하 메트릭 업데이트"""
        timestamp = time.time()
        
        cognitive_load = cognitive_data.get('cognitive_load', 0.0)
        mental_workload = cognitive_data.get('mental_workload', 0.0)
        
        self.cognitive_buffer.append({
            'timestamp': timestamp,
            'cognitive_load': cognitive_load,
            'mental_workload': mental_workload
        })
        
        # 현재 메트릭 업데이트
        self.current_metrics.cognitive_load = cognitive_load
        
        # 경고 체크
        self._check_cognitive_load_alerts(cognitive_load)
        
        logger.debug(f"인지 부하 메트릭 업데이트 - 부하: {cognitive_load:.3f}")
    
    def update_system_performance_metrics(self, performance_data: Dict[str, Any]) -> None:
        """시스템 성능 메트릭 업데이트"""
        self.analysis_count += 1
        
        processing_time = performance_data.get('processing_time_ms', 0.0)
        fps = performance_data.get('fps', 0.0)
        memory_usage = performance_data.get('memory_mb', 0.0)
        
        # 현재 메트릭 업데이트
        self.current_metrics.processing_time = processing_time
        self.current_metrics.system_fps = fps
        
        logger.debug(f"시스템 성능 업데이트 - 처리시간: {processing_time:.1f}ms, FPS: {fps:.1f}")
    
    # === 분석 및 유틸리티 메서드들 ===
    
    def _calculate_overall_confidence(self) -> float:
        """전체 신뢰도 점수 계산"""
        confidences = [
            self.current_metrics.drowsiness_confidence,
            self.current_metrics.prediction_confidence,
            # 다른 신뢰도 메트릭들도 추가
        ]
        
        valid_confidences = [c for c in confidences if c > 0]
        return np.mean(valid_confidences) if valid_confidences else 0.5
    
    def _update_trend_analysis(self):
        """트렌드 분석 업데이트"""
        if len(self.metric_history) < 30:  # 최소 1초 데이터 필요
            return
        
        for metric_name, config in self.trend_analyzers.items():
            trend = self._analyze_metric_trend(metric_name, config)
            if trend:
                # 트렌드 정보를 StateManager에 전달 (연결되어 있다면)
                if self.state_manager:
                    self.state_manager.update_trend_analysis(metric_name, trend)
    
    def _analyze_metric_trend(self, metric_name: str, config: Dict) -> Optional[TrendAnalysis]:
        """특정 메트릭의 트렌드 분석"""
        window_size = min(config['window_size'], len(self.metric_history))
        if window_size < config['min_samples']:
            return None
        
        # 최근 데이터에서 메트릭 값들 추출
        recent_snapshots = list(self.metric_history)[-window_size:]
        values = []
        
        for snapshot in recent_snapshots:
            if hasattr(snapshot, metric_name):
                values.append(getattr(snapshot, metric_name))
        
        if len(values) < config['min_samples']:
            return None
        
        # 선형 회귀를 통한 트렌드 분석
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        slope = z[0]
        
        # 트렌드 방향 및 강도 계산
        if abs(slope) < config['sensitivity']:
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        strength = min(1.0, abs(slope) / config['sensitivity'])
        
        return TrendAnalysis(
            metric_name=metric_name,
            trend_direction=direction,
            trend_strength=strength,
            prediction_window=window_size / 30.0,  # 30fps 가정
            confidence=0.8 if len(values) >= config['window_size'] else 0.6
        )
    
    def _check_drowsiness_alerts(self, score: float):
        """졸음 경고 체크"""
        thresholds = self.alert_thresholds['drowsiness']
        
        if score >= thresholds['critical']:
            self._trigger_alert('drowsiness', 'critical', score)
        elif score >= thresholds['danger']:
            self._trigger_alert('drowsiness', 'danger', score)
        elif score >= thresholds['warning']:
            self._trigger_alert('drowsiness', 'warning', score)
    
    def _check_distraction_alerts(self, score: float):
        """주의산만 경고 체크"""
        thresholds = self.alert_thresholds['distraction']
        
        if score >= thresholds['critical']:
            self._trigger_alert('distraction', 'critical', score)
        elif score >= thresholds['danger']:
            self._trigger_alert('distraction', 'danger', score)
        elif score >= thresholds['warning']:
            self._trigger_alert('distraction', 'warning', score)
    
    def _check_cognitive_load_alerts(self, load: float):
        """인지 부하 경고 체크"""
        thresholds = self.alert_thresholds['cognitive_load']
        
        if load >= thresholds['critical']:
            self._trigger_alert('cognitive_load', 'critical', load)
        elif load >= thresholds['danger']:
            self._trigger_alert('cognitive_load', 'danger', load)
        elif load >= thresholds['warning']:
            self._trigger_alert('cognitive_load', 'warning', load)
    
    def _trigger_alert(self, alert_type: str, severity: str, value: float):
        """경고 발생"""
        current_time = time.time()
        alert_key = f"{alert_type}_{severity}"
        
        # 중복 경고 방지 (5초 간격)
        if alert_key in self.last_alerts:
            if current_time - self.last_alerts[alert_key] < 5.0:
                return
        
        self.last_alerts[alert_key] = current_time
        
        logger.warning(f"경고 발생 - {alert_type}: {severity} (값: {value:.3f})")
        
        # StateManager에 경고 전달 (연결되어 있다면)
        if self.state_manager:
            self.state_manager.handle_alert(alert_type, severity, value)
    
    # === StateManager와의 연동 ===
    
    def set_state_manager(self, state_manager):
        """StateManager 연결"""
        self.state_manager = state_manager
        logger.info("StateManager와 연결됨")
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """분석 요약 정보 반환 (AnalyticsEventHandler용)"""
        if not self.metric_history:
            return {}
        
        return {
            'total_analyses': self.analysis_count,
            'avg_drowsiness': np.mean([s.drowsiness_score for s in self.metric_history]),
            'avg_distraction': np.mean([s.distraction_score for s in self.metric_history]),
            'avg_confidence': np.mean([s.confidence for s in self.metric_history]),
            'alert_count': len(self.last_alerts),
            'uptime_seconds': time.time() - self.last_update_time
        }
    
    def reset_metrics(self):
        """메트릭 리셋"""
        self.metric_history.clear()
        self.drowsiness_buffer.clear()
        self.emotion_buffer.clear()
        self.gaze_buffer.clear()
        self.distraction_buffer.clear()
        self.prediction_buffer.clear()
        
        self.current_metrics = self._create_default_metrics()
        self.analysis_count = 0
        self.last_alerts.clear()
        
        logger.info("메트릭 데이터 리셋 완료")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """디버그 정보 반환"""
        return {
            'history_size': len(self.metric_history),
            'buffer_sizes': {
                'drowsiness': len(self.drowsiness_buffer),
                'emotion': len(self.emotion_buffer),
                'gaze': len(self.gaze_buffer),
                'distraction': len(self.distraction_buffer),
                'prediction': len(self.prediction_buffer)
            },
            'analysis_count': self.analysis_count,
            'active_alerts': list(self.last_alerts.keys()),
            'trend_analyzers': list(self.trend_analyzers.keys())
        }
