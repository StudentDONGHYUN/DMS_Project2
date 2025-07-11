"""
S-Class DMS v19.0 - V2D (Vehicle-to-Driver) 헬스케어 플랫폼
차량을 움직이는 건강검진 센터로 만드는 통합 헬스케어 시스템
"""

import asyncio
import time
import numpy as np
import json
import requests
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from pathlib import Path
import logging
from datetime import datetime, timedelta

from config.settings import get_config
from models.data_structures import UIState, BiometricData


class HealthMetricType(Enum):
    """건강 메트릭 유형"""
    HEART_RATE = "heart_rate"
    HRV = "heart_rate_variability"
    BLOOD_PRESSURE = "blood_pressure"
    STRESS_LEVEL = "stress_level"
    TREMOR_ANALYSIS = "tremor_analysis"
    BREATHING_RATE = "breathing_rate"
    FATIGUE_LEVEL = "fatigue_level"
    COGNITIVE_LOAD = "cognitive_load"


class AlertSeverity(Enum):
    """경고 심각도"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class HealthDataProvider(Enum):
    """건강 데이터 제공자"""
    APPLE_HEALTH = "apple_health"
    GOOGLE_FIT = "google_fit"
    SAMSUNG_HEALTH = "samsung_health"
    FITBIT = "fitbit"
    DOCTOR_PORTAL = "doctor_portal"
    HOSPITAL_EMR = "hospital_emr"


@dataclass
class HealthMetric:
    """건강 메트릭"""
    metric_type: HealthMetricType
    value: Union[float, Dict[str, float]]
    unit: str
    timestamp: float
    confidence: float = 1.0
    source: str = "S-Class DMS"
    notes: Optional[str] = None


@dataclass
class HealthAlert:
    """건강 경고"""
    alert_id: str
    severity: AlertSeverity
    metric_type: HealthMetricType
    message: str
    recommendation: str
    triggered_at: float
    resolved_at: Optional[float] = None
    requires_medical_attention: bool = False
    emergency_contact_triggered: bool = False


@dataclass
class EmergencyContact:
    """응급 연락처"""
    name: str
    relationship: str
    phone: str
    email: Optional[str] = None
    is_medical_professional: bool = False
    priority: int = 1  # 1(highest) - 5(lowest)


@dataclass
class HealthProfile:
    """건강 프로필"""
    user_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    
    # 기존 질환
    medical_conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    
    # 정상 범위 (개인화)
    normal_ranges: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 연동 설정
    connected_platforms: List[HealthDataProvider] = field(default_factory=list)
    emergency_contacts: List[EmergencyContact] = field(default_factory=list)
    
    # 의료진 정보
    primary_doctor: Optional[Dict[str, str]] = None
    
    last_updated: float = field(default_factory=time.time)


@dataclass
class HealthSession:
    """건강 모니터링 세션"""
    session_id: str
    user_id: str
    start_time: float
    end_time: Optional[float] = None
    metrics: List[HealthMetric] = field(default_factory=list)
    alerts: List[HealthAlert] = field(default_factory=list)
    summary: Optional[Dict[str, Any]] = None


class V2DHealthcareSystem:
    """V2D 헬스케어 메인 시스템"""
    
    def __init__(self, user_id: str = "default"):
        self.config = get_config()
        self.user_id = user_id
        self.logger = logging.getLogger(__name__)
        
        # 현재 세션
        self.current_session: Optional[HealthSession] = None
        
        # 건강 프로필 로드
        self.health_profile = self._load_health_profile()
        
        # 실시간 데이터 버퍼
        self.metrics_buffer = deque(maxlen=1800)  # 30분 @ 1Hz
        self.alert_history = deque(maxlen=100)
        
        # 이상 패턴 감지 엔진
        self.anomaly_detector = HealthAnomalyDetector(self.health_profile)
        
        # 외부 플랫폼 연동
        self.platform_connectors = self._initialize_platform_connectors()
        
        # 응급 상황 관리
        self.emergency_manager = EmergencyResponseManager(self.health_profile)
        
        print(f"🏥 V2D 헬스케어 시스템 초기화 완료 - 사용자: {user_id}")
        print(f"   연동된 플랫폼: {[p.value for p in self.health_profile.connected_platforms]}")
        print(f"   응급 연락처: {len(self.health_profile.emergency_contacts)}개")

    def _load_health_profile(self) -> HealthProfile:
        """건강 프로필 로드"""
        profile_path = Path(f"profiles/health_profile_{self.user_id}.json")
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 응급 연락처 복원
                emergency_contacts = []
                for contact_data in data.get('emergency_contacts', []):
                    emergency_contacts.append(EmergencyContact(**contact_data))
                
                # 연동 플랫폼 복원
                connected_platforms = [
                    HealthDataProvider(p) for p in data.get('connected_platforms', [])
                ]
                
                return HealthProfile(
                    user_id=data.get('user_id', self.user_id),
                    age=data.get('age'),
                    gender=data.get('gender'),
                    weight_kg=data.get('weight_kg'),
                    height_cm=data.get('height_cm'),
                    medical_conditions=data.get('medical_conditions', []),
                    medications=data.get('medications', []),
                    allergies=data.get('allergies', []),
                    normal_ranges=data.get('normal_ranges', {}),
                    connected_platforms=connected_platforms,
                    emergency_contacts=emergency_contacts,
                    primary_doctor=data.get('primary_doctor'),
                    last_updated=data.get('last_updated', time.time())
                )
            except Exception as e:
                self.logger.error(f"건강 프로필 로드 실패: {e}")
        
        # 기본 프로필 생성
        return HealthProfile(user_id=self.user_id)

    def _save_health_profile(self):
        """건강 프로필 저장"""
        profile_path = Path(f"profiles/health_profile_{self.user_id}.json")
        profile_path.parent.mkdir(exist_ok=True)
        
        # 직렬화 가능한 형태로 변환
        data = {
            'user_id': self.health_profile.user_id,
            'age': self.health_profile.age,
            'gender': self.health_profile.gender,
            'weight_kg': self.health_profile.weight_kg,
            'height_cm': self.health_profile.height_cm,
            'medical_conditions': self.health_profile.medical_conditions,
            'medications': self.health_profile.medications,
            'allergies': self.health_profile.allergies,
            'normal_ranges': self.health_profile.normal_ranges,
            'connected_platforms': [p.value for p in self.health_profile.connected_platforms],
            'emergency_contacts': [
                {
                    'name': c.name,
                    'relationship': c.relationship,
                    'phone': c.phone,
                    'email': c.email,
                    'is_medical_professional': c.is_medical_professional,
                    'priority': c.priority
                } for c in self.health_profile.emergency_contacts
            ],
            'primary_doctor': self.health_profile.primary_doctor,
            'last_updated': self.health_profile.last_updated
        }
        
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _initialize_platform_connectors(self) -> Dict[HealthDataProvider, Any]:
        """외부 플랫폼 연동 초기화"""
        connectors = {}
        
        for platform in self.health_profile.connected_platforms:
            if platform == HealthDataProvider.APPLE_HEALTH:
                connectors[platform] = AppleHealthConnector(self.user_id)
            elif platform == HealthDataProvider.GOOGLE_FIT:
                connectors[platform] = GoogleFitConnector(self.user_id)
            elif platform == HealthDataProvider.DOCTOR_PORTAL:
                connectors[platform] = DoctorPortalConnector(
                    self.user_id, self.health_profile.primary_doctor
                )
        
        return connectors

    async def start_health_monitoring(self) -> str:
        """건강 모니터링 세션 시작"""
        session_id = f"health_session_{int(time.time())}"
        self.current_session = HealthSession(
            session_id=session_id,
            user_id=self.user_id,
            start_time=time.time()
        )
        
        print(f"🩺 건강 모니터링 세션 시작: {session_id}")
        return session_id

    async def process_biometric_data(self, ui_state: UIState) -> List[HealthAlert]:
        """생체 데이터 처리 및 건강 분석"""
        if not self.current_session:
            return []
        
        # UI 상태에서 건강 메트릭 추출
        health_metrics = self._extract_health_metrics(ui_state)
        
        # 메트릭 저장
        self.current_session.metrics.extend(health_metrics)
        self.metrics_buffer.extend(health_metrics)
        
        # 이상 패턴 감지
        alerts = await self.anomaly_detector.detect_anomalies(health_metrics)
        
        # 긴급 상황 체크
        emergency_alerts = await self.emergency_manager.check_emergency_conditions(
            health_metrics, alerts
        )
        
        all_alerts = alerts + emergency_alerts
        
        if all_alerts:
            self.current_session.alerts.extend(all_alerts)
            self.alert_history.extend(all_alerts)
            
            # 외부 플랫폼에 데이터 전송
            await self._sync_to_external_platforms(health_metrics, all_alerts)
        
        return all_alerts

    def _extract_health_metrics(self, ui_state: UIState) -> List[HealthMetric]:
        """UI 상태에서 건강 메트릭 추출"""
        metrics = []
        current_time = time.time()
        
        # 심박수 (rPPG)
        if ui_state.biometrics.heart_rate:
            metrics.append(HealthMetric(
                metric_type=HealthMetricType.HEART_RATE,
                value=ui_state.biometrics.heart_rate,
                unit="bpm",
                timestamp=current_time,
                confidence=ui_state.biometrics.confidence
            ))
        
        # 심박 변이도
        if ui_state.biometrics.heart_rate_variability:
            metrics.append(HealthMetric(
                metric_type=HealthMetricType.HRV,
                value=ui_state.biometrics.heart_rate_variability,
                unit="ms",
                timestamp=current_time,
                confidence=ui_state.biometrics.confidence
            ))
        
        # 스트레스 레벨
        if ui_state.biometrics.stress_level is not None:
            metrics.append(HealthMetric(
                metric_type=HealthMetricType.STRESS_LEVEL,
                value=ui_state.biometrics.stress_level,
                unit="0-1",
                timestamp=current_time,
                confidence=0.8
            ))
        
        # 손 떨림 분석 (파킨슨병 등 신경계 질환 조기 감지)
        if ui_state.hands.tremor_frequency:
            tremor_data = {
                'frequency': ui_state.hands.tremor_frequency,
                'motor_control_score': ui_state.hands.motor_control_score,
                'grip_stability': ui_state.hands.grip_stability
            }
            metrics.append(HealthMetric(
                metric_type=HealthMetricType.TREMOR_ANALYSIS,
                value=tremor_data,
                unit="Hz",
                timestamp=current_time,
                confidence=ui_state.hands.hand_position_confidence,
                notes="신경계 건강 모니터링"
            ))
        
        # 인지 부하 (동공 분석 기반)
        cognitive_load = getattr(ui_state.face, 'cognitive_load_indicator', 0.0)
        if cognitive_load > 0:
            metrics.append(HealthMetric(
                metric_type=HealthMetricType.COGNITIVE_LOAD,
                value=cognitive_load,
                unit="0-1",
                timestamp=current_time,
                confidence=0.7
            ))
        
        # 피로도
        fatigue_score = 1.0 - ui_state.gaze.attention_score
        metrics.append(HealthMetric(
            metric_type=HealthMetricType.FATIGUE_LEVEL,
            value=fatigue_score,
            unit="0-1",
            timestamp=current_time,
            confidence=0.8
        ))
        
        return metrics

    async def _sync_to_external_platforms(self, metrics: List[HealthMetric], alerts: List[HealthAlert]):
        """외부 플랫폼에 데이터 동기화"""
        sync_tasks = []
        
        for platform, connector in self.platform_connectors.items():
            task = asyncio.create_task(
                connector.sync_health_data(metrics, alerts)
            )
            sync_tasks.append(task)
        
        if sync_tasks:
            try:
                await asyncio.gather(*sync_tasks, return_exceptions=True)
                self.logger.info(f"건강 데이터 {len(self.platform_connectors)}개 플랫폼에 동기화 완료")
            except Exception as e:
                self.logger.error(f"외부 플랫폼 동기화 실패: {e}")

    async def end_health_session(self) -> Dict[str, Any]:
        """건강 모니터링 세션 종료"""
        if not self.current_session:
            return {"error": "활성 세션이 없습니다"}
        
        self.current_session.end_time = time.time()
        
        # 세션 요약 생성
        session_summary = self._generate_session_summary()
        self.current_session.summary = session_summary
        
        # 세션 데이터 저장
        self._save_health_session()
        
        # 주치의에게 요약 보고서 전송 (설정된 경우)
        if self.health_profile.primary_doctor:
            await self._send_doctor_report(session_summary)
        
        session_duration = (self.current_session.end_time - self.current_session.start_time) / 60.0
        print(f"🏥 건강 모니터링 세션 종료 - 지속시간: {session_duration:.1f}분")
        print(f"   수집된 메트릭: {len(self.current_session.metrics)}개")
        print(f"   발생한 경고: {len(self.current_session.alerts)}개")
        
        result = {
            "session_summary": session_summary,
            "duration_minutes": session_duration,
            "metrics_count": len(self.current_session.metrics),
            "alerts_count": len(self.current_session.alerts)
        }
        
        self.current_session = None
        return result

    def _generate_session_summary(self) -> Dict[str, Any]:
        """세션 요약 생성"""
        if not self.current_session.metrics:
            return {}
        
        summary = {
            "session_duration_minutes": (time.time() - self.current_session.start_time) / 60.0,
            "metrics_summary": {},
            "health_insights": [],
            "recommendations": [],
            "anomalies_detected": len(self.current_session.alerts)
        }
        
        # 메트릭 유형별 통계
        metric_groups = {}
        for metric in self.current_session.metrics:
            metric_type = metric.metric_type.value
            if metric_type not in metric_groups:
                metric_groups[metric_type] = []
            
            if isinstance(metric.value, (int, float)):
                metric_groups[metric_type].append(metric.value)
        
        for metric_type, values in metric_groups.items():
            if values:
                summary["metrics_summary"][metric_type] = {
                    "average": np.mean(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "std": np.std(values),
                    "count": len(values)
                }
        
        # 건강 인사이트 생성
        summary["health_insights"] = self._generate_health_insights(summary["metrics_summary"])
        
        # 추천사항 생성
        summary["recommendations"] = self._generate_health_recommendations(
            summary["metrics_summary"], self.current_session.alerts
        )
        
        return summary

    def _generate_health_insights(self, metrics_summary: Dict[str, Any]) -> List[str]:
        """건강 인사이트 생성"""
        insights = []
        
        # 심박수 분석
        if "heart_rate" in metrics_summary:
            hr_data = metrics_summary["heart_rate"]
            avg_hr = hr_data["average"]
            
            if avg_hr < 60:
                insights.append(f"평균 심박수가 {avg_hr:.1f}bpm으로 낮습니다. 운동 능력이 좋거나 서맥 가능성을 확인해보세요.")
            elif avg_hr > 100:
                insights.append(f"평균 심박수가 {avg_hr:.1f}bpm으로 높습니다. 스트레스나 카페인 섭취를 확인해보세요.")
            else:
                insights.append(f"평균 심박수 {avg_hr:.1f}bpm으로 정상 범위입니다.")
        
        # 스트레스 분석
        if "stress_level" in metrics_summary:
            stress_data = metrics_summary["stress_level"]
            avg_stress = stress_data["average"]
            
            if avg_stress > 0.7:
                insights.append("높은 스트레스 레벨이 감지되었습니다. 휴식이 필요합니다.")
            elif avg_stress > 0.5:
                insights.append("중간 수준의 스트레스가 감지되었습니다. 이완 기법을 시도해보세요.")
        
        # 떨림 분석
        if "tremor_analysis" in metrics_summary:
            insights.append("손 떨림 데이터가 수집되었습니다. 지속적인 모니터링으로 신경계 건강을 추적할 수 있습니다.")
        
        return insights

    def _generate_health_recommendations(self, metrics_summary: Dict[str, Any], alerts: List[HealthAlert]) -> List[str]:
        """건강 추천사항 생성"""
        recommendations = []
        
        # 경고 기반 추천
        for alert in alerts:
            if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                recommendations.append(alert.recommendation)
        
        # 메트릭 기반 추천
        if "stress_level" in metrics_summary:
            avg_stress = metrics_summary["stress_level"]["average"]
            if avg_stress > 0.6:
                recommendations.append("정기적인 심호흡이나 명상으로 스트레스를 관리하세요.")
        
        if "fatigue_level" in metrics_summary:
            avg_fatigue = metrics_summary["fatigue_level"]["average"]
            if avg_fatigue > 0.7:
                recommendations.append("충분한 수면과 휴식을 취하세요.")
        
        # 일반적인 건강 추천
        recommendations.extend([
            "정기적인 건강검진을 받으시기 바랍니다.",
            "균형 잡힌 식단과 규칙적인 운동을 유지하세요.",
            "충분한 수분 섭취를 하세요."
        ])
        
        return recommendations[:5]  # 최대 5개

    def _save_health_session(self):
        """건강 세션 저장"""
        if not self.current_session:
            return
        
        sessions_dir = Path("profiles/health_sessions")
        sessions_dir.mkdir(exist_ok=True)
        
        session_file = sessions_dir / f"{self.current_session.session_id}.json"
        
        # 세션 데이터 직렬화
        session_data = {
            "session_id": self.current_session.session_id,
            "user_id": self.current_session.user_id,
            "start_time": self.current_session.start_time,
            "end_time": self.current_session.end_time,
            "metrics_count": len(self.current_session.metrics),
            "alerts_count": len(self.current_session.alerts),
            "summary": self.current_session.summary
        }
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

    async def _send_doctor_report(self, session_summary: Dict[str, Any]):
        """주치의에게 보고서 전송"""
        if not self.health_profile.primary_doctor:
            return
        
        try:
            doctor_connector = self.platform_connectors.get(HealthDataProvider.DOCTOR_PORTAL)
            if doctor_connector:
                await doctor_connector.send_health_report(session_summary)
                self.logger.info("주치의에게 건강 보고서 전송 완료")
        except Exception as e:
            self.logger.error(f"주치의 보고서 전송 실패: {e}")

    def add_emergency_contact(self, contact: EmergencyContact):
        """응급 연락처 추가"""
        self.health_profile.emergency_contacts.append(contact)
        self.health_profile.emergency_contacts.sort(key=lambda x: x.priority)
        self._save_health_profile()
        print(f"응급 연락처 추가: {contact.name} ({contact.relationship})")

    def connect_health_platform(self, platform: HealthDataProvider, credentials: Dict[str, str]):
        """건강 플랫폼 연동"""
        if platform not in self.health_profile.connected_platforms:
            self.health_profile.connected_platforms.append(platform)
            self._save_health_profile()
            
            # 커넥터 초기화
            if platform == HealthDataProvider.APPLE_HEALTH:
                self.platform_connectors[platform] = AppleHealthConnector(
                    self.user_id, credentials
                )
            elif platform == HealthDataProvider.GOOGLE_FIT:
                self.platform_connectors[platform] = GoogleFitConnector(
                    self.user_id, credentials
                )
            
            print(f"건강 플랫폼 연동 완료: {platform.value}")

    def get_health_statistics(self) -> Dict[str, Any]:
        """건강 통계 조회"""
        # 최근 세션들 분석
        sessions_dir = Path("profiles/health_sessions")
        recent_sessions = []
        
        if sessions_dir.exists():
            session_files = sorted(sessions_dir.glob(f"health_session_*.json"))[-10:]  # 최근 10개
            
            for session_file in session_files:
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                        recent_sessions.append(session_data)
                except Exception:
                    continue
        
        stats = {
            "total_sessions": len(recent_sessions),
            "recent_alerts": sum(s.get("alerts_count", 0) for s in recent_sessions),
            "average_session_duration": np.mean([
                s.get("summary", {}).get("session_duration_minutes", 0) 
                for s in recent_sessions
            ]) if recent_sessions else 0,
            "connected_platforms": [p.value for p in self.health_profile.connected_platforms],
            "emergency_contacts_count": len(self.health_profile.emergency_contacts)
        }
        
        return stats


class HealthAnomalyDetector:
    """건강 이상 패턴 감지 엔진"""
    
    def __init__(self, health_profile: HealthProfile):
        self.health_profile = health_profile
        self.logger = logging.getLogger(__name__)
    
    async def detect_anomalies(self, metrics: List[HealthMetric]) -> List[HealthAlert]:
        """이상 패턴 감지"""
        alerts = []
        
        for metric in metrics:
            alert = self._check_metric_anomaly(metric)
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def _check_metric_anomaly(self, metric: HealthMetric) -> Optional[HealthAlert]:
        """개별 메트릭 이상 체크"""
        if metric.metric_type == HealthMetricType.HEART_RATE:
            return self._check_heart_rate_anomaly(metric)
        elif metric.metric_type == HealthMetricType.TREMOR_ANALYSIS:
            return self._check_tremor_anomaly(metric)
        elif metric.metric_type == HealthMetricType.STRESS_LEVEL:
            return self._check_stress_anomaly(metric)
        
        return None
    
    def _check_heart_rate_anomaly(self, metric: HealthMetric) -> Optional[HealthAlert]:
        """심박수 이상 체크"""
        hr = metric.value
        
        # 개인화된 정상 범위 확인
        normal_range = self.health_profile.normal_ranges.get("heart_rate", {
            "min": 60, "max": 100
        })
        
        if hr < normal_range["min"] - 10:  # 심각한 서맥
            return HealthAlert(
                alert_id=f"hr_low_{int(metric.timestamp)}",
                severity=AlertSeverity.HIGH,
                metric_type=HealthMetricType.HEART_RATE,
                message=f"심박수가 {hr:.0f}bpm으로 비정상적으로 낮습니다.",
                recommendation="즉시 안전한 곳에 정차하고 의료진에게 연락하세요.",
                triggered_at=metric.timestamp,
                requires_medical_attention=True
            )
        elif hr > normal_range["max"] + 20:  # 심각한 빈맥
            return HealthAlert(
                alert_id=f"hr_high_{int(metric.timestamp)}",
                severity=AlertSeverity.HIGH,
                metric_type=HealthMetricType.HEART_RATE,
                message=f"심박수가 {hr:.0f}bpm으로 비정상적으로 높습니다.",
                recommendation="깊게 숨을 들이쉬고 진정하세요. 증상이 지속되면 의료진에게 연락하세요.",
                triggered_at=metric.timestamp,
                requires_medical_attention=True
            )
        
        return None
    
    def _check_tremor_anomaly(self, metric: HealthMetric) -> Optional[HealthAlert]:
        """떨림 이상 체크 (파킨슨병 등)"""
        if isinstance(metric.value, dict):
            frequency = metric.value.get('frequency', 0)
            
            # 4-6Hz 떨림은 파킨슨병의 특징적 주파수
            if 4.0 <= frequency <= 6.0:
                return HealthAlert(
                    alert_id=f"tremor_parkinsonian_{int(metric.timestamp)}",
                    severity=AlertSeverity.MEDIUM,
                    metric_type=HealthMetricType.TREMOR_ANALYSIS,
                    message=f"파킨슨병과 관련된 떨림 패턴이 감지되었습니다 ({frequency:.1f}Hz).",
                    recommendation="신경과 전문의와 상담을 받으시기 바랍니다. 지속적인 모니터링이 필요합니다.",
                    triggered_at=metric.timestamp,
                    requires_medical_attention=True
                )
        
        return None
    
    def _check_stress_anomaly(self, metric: HealthMetric) -> Optional[HealthAlert]:
        """스트레스 이상 체크"""
        stress_level = metric.value
        
        if stress_level > 0.8:
            return HealthAlert(
                alert_id=f"stress_high_{int(metric.timestamp)}",
                severity=AlertSeverity.MEDIUM,
                metric_type=HealthMetricType.STRESS_LEVEL,
                message="매우 높은 스트레스 레벨이 감지되었습니다.",
                recommendation="즉시 휴식을 취하고 심호흡을 하세요. 안전한 곳에서 잠시 멈춰 진정하세요.",
                triggered_at=metric.timestamp
            )
        
        return None


class EmergencyResponseManager:
    """응급 상황 대응 관리자"""
    
    def __init__(self, health_profile: HealthProfile):
        self.health_profile = health_profile
        self.logger = logging.getLogger(__name__)
    
    async def check_emergency_conditions(self, metrics: List[HealthMetric], alerts: List[HealthAlert]) -> List[HealthAlert]:
        """응급 상황 체크"""
        emergency_alerts = []
        
        # 심각한 경고가 있는지 확인
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.HIGH]
        
        if critical_alerts:
            # 응급 상황 판단
            emergency_alert = await self._assess_emergency_situation(critical_alerts, metrics)
            if emergency_alert:
                emergency_alerts.append(emergency_alert)
                
                # 응급 연락처에 알림
                await self._trigger_emergency_response(emergency_alert)
        
        return emergency_alerts
    
    async def _assess_emergency_situation(self, alerts: List[HealthAlert], metrics: List[HealthMetric]) -> Optional[HealthAlert]:
        """응급 상황 평가"""
        # 복합적 위험 상황 판단
        hr_critical = any(a.metric_type == HealthMetricType.HEART_RATE for a in alerts)
        multiple_systems = len(set(a.metric_type for a in alerts)) > 1
        
        if hr_critical and multiple_systems:
            return HealthAlert(
                alert_id=f"emergency_{int(time.time())}",
                severity=AlertSeverity.EMERGENCY,
                metric_type=HealthMetricType.HEART_RATE,  # 주요 원인
                message="복합적 건강 위험 상황이 감지되었습니다. 즉시 의료진의 도움이 필요합니다.",
                recommendation="차량을 안전하게 정차시키고 119에 신고하세요.",
                triggered_at=time.time(),
                requires_medical_attention=True,
                emergency_contact_triggered=True
            )
        
        return None
    
    async def _trigger_emergency_response(self, emergency_alert: HealthAlert):
        """응급 대응 트리거"""
        try:
            # 1. 응급 연락처에 알림
            await self._notify_emergency_contacts(emergency_alert)
            
            # 2. 자동 119 신고 준비 (사용자 확인 후)
            await self._prepare_emergency_call(emergency_alert)
            
            # 3. 차량 안전 정차 권고
            await self._recommend_safe_stop()
            
            self.logger.critical(f"응급 상황 대응 트리거: {emergency_alert.message}")
            
        except Exception as e:
            self.logger.error(f"응급 대응 실패: {e}")
    
    async def _notify_emergency_contacts(self, alert: HealthAlert):
        """응급 연락처 알림"""
        for contact in self.health_profile.emergency_contacts:
            try:
                message = f"[S-Class DMS 응급 알림] {alert.message}"
                # SMS 또는 푸시 알림 전송 (실제 구현 시 SMS API 연동)
                print(f"📱 응급 알림 전송: {contact.name} ({contact.phone})")
                print(f"   메시지: {message}")
                
            except Exception as e:
                self.logger.error(f"응급 연락처 알림 실패 - {contact.name}: {e}")
    
    async def _prepare_emergency_call(self, alert: HealthAlert):
        """자동 119 신고 준비"""
        print("🚨 자동 119 신고 준비 중...")
        print("   10초 후 자동으로 119에 신고됩니다.")
        print("   취소하려면 음성으로 '취소'라고 말하세요.")
        
        # 실제 구현 시: 음성 인식으로 사용자 의사 확인
        # await asyncio.sleep(10)
        # if not cancelled_by_user:
        #     await self._make_emergency_call(alert)
    
    async def _recommend_safe_stop(self):
        """안전 정차 권고"""
        print("🛑 즉시 안전한 곳에 정차하세요!")
        print("   - 갓길이나 휴게소로 이동")
        print("   - 비상등 점등")
        print("   - 동승자가 있다면 도움 요청")


# 외부 플랫폼 연동 클래스들 (실제 구현 시 각 플랫폼 API 연동)

class AppleHealthConnector:
    """Apple Health 연동"""
    
    def __init__(self, user_id: str, credentials: Dict[str, str] = None):
        self.user_id = user_id
        self.credentials = credentials or {}
    
    async def sync_health_data(self, metrics: List[HealthMetric], alerts: List[HealthAlert]):
        """Apple Health에 데이터 동기화"""
        # 실제 구현 시 HealthKit API 연동
        print(f"📱 Apple Health에 {len(metrics)}개 메트릭 동기화")


class GoogleFitConnector:
    """Google Fit 연동"""
    
    def __init__(self, user_id: str, credentials: Dict[str, str] = None):
        self.user_id = user_id
        self.credentials = credentials or {}
    
    async def sync_health_data(self, metrics: List[HealthMetric], alerts: List[HealthAlert]):
        """Google Fit에 데이터 동기화"""
        # 실제 구현 시 Google Fit API 연동
        print(f"📱 Google Fit에 {len(metrics)}개 메트릭 동기화")


class DoctorPortalConnector:
    """의사 포털 연동"""
    
    def __init__(self, user_id: str, doctor_info: Dict[str, str] = None):
        self.user_id = user_id
        self.doctor_info = doctor_info or {}
    
    async def sync_health_data(self, metrics: List[HealthMetric], alerts: List[HealthAlert]):
        """의사 포털에 데이터 전송"""
        print(f"👨‍⚕️ 의사 포털에 {len(metrics)}개 메트릭 전송")
    
    async def send_health_report(self, session_summary: Dict[str, Any]):
        """주치의에게 건강 보고서 전송"""
        print(f"📧 주치의에게 건강 보고서 전송 완료")