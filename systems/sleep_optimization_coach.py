"""
S-Class DMS v19.0 - AI 수면 품질 최적화 코치
운전자의 수면 패턴과 피로도를 분석하여 최적의 수면 스케줄 제안 시스템
"""

import asyncio
import time
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from pathlib import Path
import datetime
import math

from config.settings import get_config
from models.data_structures import UIState, BiometricData


class SleepPhase(Enum):
    """수면 단계"""
    DEEP_SLEEP = "deep_sleep"         # 깊은 잠
    LIGHT_SLEEP = "light_sleep"       # 얕은 잠
    REM_SLEEP = "rem_sleep"           # 렘 수면
    AWAKE = "awake"                   # 깨어있음
    DROWSY = "drowsy"                 # 졸음


class CircadianPhase(Enum):
    """일주기 리듬 단계"""
    MORNING_PEAK = "morning_peak"     # 아침 각성 피크 (6-9시)
    MIDDAY_DIP = "midday_dip"        # 오후 피로 (13-15시)
    EVENING_PEAK = "evening_peak"     # 저녁 각성 피크 (18-21시)
    NIGHT_ONSET = "night_onset"      # 밤 졸음 시작 (21-23시)
    DEEP_NIGHT = "deep_night"        # 깊은 밤 (23-6시)


class FatigueLevel(Enum):
    """피로 수준"""
    EXCELLENT = "excellent"           # 매우 좋음 (0-20%)
    GOOD = "good"                    # 좋음 (20-40%)
    MODERATE = "moderate"            # 보통 (40-60%)
    HIGH = "high"                    # 높음 (60-80%)
    CRITICAL = "critical"            # 위험 (80-100%)


@dataclass
class SleepData:
    """수면 데이터"""
    date: str
    bedtime: float  # 취침 시간 (timestamp)
    wake_time: float  # 기상 시간 (timestamp)
    sleep_duration: float  # 수면 시간 (시간)
    sleep_quality: float  # 수면 품질 (0-1)
    deep_sleep_ratio: float  # 깊은 잠 비율
    rem_sleep_ratio: float  # 렘 수면 비율
    wake_up_feeling: str  # 기상 시 컨디션 ("refreshed", "tired", "groggy")
    sleep_efficiency: float  # 수면 효율성


@dataclass
class CircadianProfile:
    """개인 일주기 리듬 프로필"""
    user_id: str
    chronotype: str = "normal"  # "early", "normal", "late"
    optimal_bedtime: float = 23.0  # 최적 취침 시간 (시)
    optimal_wake_time: float = 7.0  # 최적 기상 시간 (시)
    natural_sleep_duration: float = 8.0  # 자연 수면 시간
    energy_peaks: List[float] = field(default_factory=lambda: [9.0, 19.0])  # 에너지 피크 시간
    energy_dips: List[float] = field(default_factory=lambda: [14.0, 2.0])   # 에너지 저하 시간
    light_sensitivity: float = 0.7  # 빛 민감도
    caffeine_tolerance: float = 0.5  # 카페인 내성
    last_updated: float = field(default_factory=time.time)


@dataclass
class SleepRecommendation:
    """수면 추천사항"""
    type: str  # "bedtime", "wake_time", "nap", "caffeine", "light_exposure"
    message: str
    optimal_time: Optional[float] = None
    duration: Optional[float] = None
    priority: int = 1  # 1(높음) - 5(낮음)
    reasoning: str = ""
    expected_benefit: str = ""


@dataclass
class PowerNapSuggestion:
    """파워 낮잠 제안"""
    suggested_time: float  # 제안 시간 (timestamp)
    optimal_duration: float  # 최적 지속 시간 (분)
    fatigue_reduction_expected: float  # 예상 피로 감소율
    risk_assessment: str  # "low", "medium", "high" (밤잠에 미치는 영향)


class SleepOptimizationCoach:
    """AI 수면 품질 최적화 코치 메인 시스템"""
    
    def __init__(self, user_id: str = "default"):
        self.config = get_config()
        self.user_id = user_id
        
        # 수면 프로필 로드
        self.circadian_profile = self._load_circadian_profile()
        
        # 수면 데이터 히스토리
        self.sleep_history = self._load_sleep_history()
        
        # 실시간 피로도 추적
        self.fatigue_history = deque(maxlen=1440)  # 24시간 (분 단위)
        self.alertness_history = deque(maxlen=300)  # 10분
        
        # 일주기 리듬 추적
        self.circadian_tracker = CircadianRhythmTracker(self.circadian_profile)
        
        # 수면 부채 계산기
        self.sleep_debt_calculator = SleepDebtCalculator()
        
        # 추천 엔진
        self.recommendation_engine = SleepRecommendationEngine(self.circadian_profile)
        
        # 현재 상태
        self.current_fatigue_level = FatigueLevel.GOOD
        self.last_recommendation_time = 0.0
        
        print(f"😴 수면 최적화 코치 초기화 완료 - 사용자: {user_id}")
        print(f"   크로노타입: {self.circadian_profile.chronotype}")
        print(f"   최적 취침 시간: {self.circadian_profile.optimal_bedtime:.1f}시")
        print(f"   수면 데이터: {len(self.sleep_history)}일")

    def _load_circadian_profile(self) -> CircadianProfile:
        """일주기 리듬 프로필 로드"""
        profile_path = Path(f"profiles/circadian_profile_{self.user_id}.json")
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return CircadianProfile(
                        user_id=data.get('user_id', self.user_id),
                        chronotype=data.get('chronotype', 'normal'),
                        optimal_bedtime=data.get('optimal_bedtime', 23.0),
                        optimal_wake_time=data.get('optimal_wake_time', 7.0),
                        natural_sleep_duration=data.get('natural_sleep_duration', 8.0),
                        energy_peaks=data.get('energy_peaks', [9.0, 19.0]),
                        energy_dips=data.get('energy_dips', [14.0, 2.0]),
                        light_sensitivity=data.get('light_sensitivity', 0.7),
                        caffeine_tolerance=data.get('caffeine_tolerance', 0.5),
                        last_updated=data.get('last_updated', time.time())
                    )
            except Exception as e:
                print(f"일주기 프로필 로드 실패: {e}")
        
        # 기본 프로필 생성
        return CircadianProfile(user_id=self.user_id)

    def _load_sleep_history(self) -> List[SleepData]:
        """수면 히스토리 로드"""
        history_path = Path(f"logs/sleep_history_{self.user_id}.json")
        
        if history_path.exists():
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return [
                        SleepData(
                            date=entry['date'],
                            bedtime=entry['bedtime'],
                            wake_time=entry['wake_time'],
                            sleep_duration=entry['sleep_duration'],
                            sleep_quality=entry['sleep_quality'],
                            deep_sleep_ratio=entry['deep_sleep_ratio'],
                            rem_sleep_ratio=entry['rem_sleep_ratio'],
                            wake_up_feeling=entry['wake_up_feeling'],
                            sleep_efficiency=entry['sleep_efficiency']
                        ) for entry in data
                    ]
            except Exception as e:
                print(f"수면 히스토리 로드 실패: {e}")
        
        return []

    async def analyze_current_fatigue(self, ui_state: UIState) -> FatigueLevel:
        """현재 피로도 분석"""
        
        # 다양한 지표로 피로도 계산
        fatigue_indicators = await self._collect_fatigue_indicators(ui_state)
        
        # 종합 피로도 점수 계산
        fatigue_score = await self._calculate_fatigue_score(fatigue_indicators)
        
        # 피로 수준 분류
        fatigue_level = self._classify_fatigue_level(fatigue_score)
        
        # 히스토리에 기록
        current_time = time.time()
        self.fatigue_history.append({
            'timestamp': current_time,
            'fatigue_score': fatigue_score,
            'fatigue_level': fatigue_level.value
        })
        
        self.current_fatigue_level = fatigue_level
        
        return fatigue_level

    async def _collect_fatigue_indicators(self, ui_state: UIState) -> Dict[str, float]:
        """피로도 지표 수집"""
        
        indicators = {}
        
        # 눈 기반 지표
        indicators['blink_frequency'] = getattr(ui_state.face, 'blink_frequency', 15.0)
        indicators['eyelid_closure'] = getattr(ui_state.face, 'eyelid_closure_rate', 0.0)
        indicators['eye_redness'] = getattr(ui_state.face, 'eye_redness_level', 0.0)
        
        # 주의력 지표
        indicators['attention_score'] = ui_state.gaze.attention_score
        indicators['reaction_time'] = getattr(ui_state.gaze, 'reaction_time_ms', 250.0)
        indicators['microsleep_events'] = getattr(ui_state.face, 'microsleep_count', 0)
        
        # 생체 지표
        indicators['heart_rate_variability'] = ui_state.biometrics.heart_rate_variability or 50.0
        indicators['stress_level'] = ui_state.biometrics.stress_level or 0.0
        
        # 자세 기반 지표
        indicators['head_nodding'] = getattr(ui_state.posture, 'head_nodding_frequency', 0.0)
        indicators['posture_stability'] = ui_state.posture.spinal_alignment_score
        
        # 손 떨림 (카페인/피로 구분)
        indicators['hand_tremor'] = ui_state.hands.tremor_frequency or 0.0
        
        # 시간 기반 지표
        current_hour = datetime.datetime.now().hour
        indicators['circadian_alertness'] = await self.circadian_tracker.get_predicted_alertness(current_hour)
        
        return indicators

    async def _calculate_fatigue_score(self, indicators: Dict[str, float]) -> float:
        """종합 피로도 점수 계산"""
        
        # 각 지표의 가중치
        weights = {
            'blink_frequency': 0.15,      # 깜박임 빈도 (높을수록 피로)
            'eyelid_closure': 0.20,       # 눈꺼풀 처짐 (높을수록 피로)
            'eye_redness': 0.10,          # 눈 충혈 (높을수록 피로)
            'attention_score': -0.20,     # 주의력 (높을수록 덜 피로)
            'reaction_time': 0.15,        # 반응 시간 (높을수록 피로)
            'microsleep_events': 0.25,    # 마이크로슬립 (높을수록 위험)
            'heart_rate_variability': -0.10,  # HRV (높을수록 덜 피로)
            'stress_level': 0.15,         # 스트레스 (높을수록 피로)
            'head_nodding': 0.20,         # 고개 끄덕임 (높을수록 피로)
            'posture_stability': -0.10,   # 자세 안정성 (높을수록 덜 피로)
            'hand_tremor': 0.05,          # 손 떨림
            'circadian_alertness': -0.15  # 일주기 각성도 (높을수록 덜 피로)
        }
        
        # 정규화된 점수 계산
        normalized_scores = {}
        
        # 각 지표를 0-1 범위로 정규화
        normalized_scores['blink_frequency'] = min(1.0, max(0.0, (indicators['blink_frequency'] - 10) / 30))
        normalized_scores['eyelid_closure'] = indicators['eyelid_closure']
        normalized_scores['eye_redness'] = indicators['eye_redness']
        normalized_scores['attention_score'] = indicators['attention_score']
        normalized_scores['reaction_time'] = min(1.0, max(0.0, (indicators['reaction_time'] - 200) / 300))
        normalized_scores['microsleep_events'] = min(1.0, indicators['microsleep_events'] / 5.0)
        normalized_scores['heart_rate_variability'] = min(1.0, indicators['heart_rate_variability'] / 100.0)
        normalized_scores['stress_level'] = indicators['stress_level']
        normalized_scores['head_nodding'] = min(1.0, indicators['head_nodding'] / 5.0)
        normalized_scores['posture_stability'] = indicators['posture_stability']
        normalized_scores['hand_tremor'] = min(1.0, indicators['hand_tremor'] / 3.0)
        normalized_scores['circadian_alertness'] = indicators['circadian_alertness']
        
        # 가중 평균 계산
        weighted_score = sum(
            normalized_scores[key] * weights[key] 
            for key in weights.keys() 
            if key in normalized_scores
        )
        
        # 0-1 범위로 클리핑
        return max(0.0, min(1.0, weighted_score + 0.5))  # 기준점 조정

    def _classify_fatigue_level(self, fatigue_score: float) -> FatigueLevel:
        """피로도 점수를 레벨로 분류"""
        
        if fatigue_score <= 0.2:
            return FatigueLevel.EXCELLENT
        elif fatigue_score <= 0.4:
            return FatigueLevel.GOOD
        elif fatigue_score <= 0.6:
            return FatigueLevel.MODERATE
        elif fatigue_score <= 0.8:
            return FatigueLevel.HIGH
        else:
            return FatigueLevel.CRITICAL

    async def generate_sleep_recommendations(self) -> List[SleepRecommendation]:
        """개인화된 수면 추천사항 생성"""
        
        recommendations = []
        current_time = time.time()
        current_hour = datetime.datetime.now().hour
        
        # 너무 자주 추천하지 않도록 제한 (30분 간격)
        if current_time - self.last_recommendation_time < 1800:
            return []
        
        # 현재 수면 부채 계산
        sleep_debt = await self.sleep_debt_calculator.calculate_current_debt(self.sleep_history)
        
        # 일주기 리듬 상태 확인
        circadian_phase = await self.circadian_tracker.get_current_phase(current_hour)
        
        # 추천사항 생성
        
        # 1. 취침 시간 추천
        if current_hour >= 20:  # 저녁 8시 이후
            bedtime_rec = await self._generate_bedtime_recommendation(sleep_debt, circadian_phase)
            if bedtime_rec:
                recommendations.append(bedtime_rec)
        
        # 2. 파워 낮잠 추천
        if 12 <= current_hour <= 16 and self.current_fatigue_level.value in ['high', 'critical']:
            nap_rec = await self._generate_nap_recommendation(current_hour, sleep_debt)
            if nap_rec:
                recommendations.append(nap_rec)
        
        # 3. 카페인 섭취 추천
        caffeine_rec = await self._generate_caffeine_recommendation(current_hour, self.current_fatigue_level)
        if caffeine_rec:
            recommendations.append(caffeine_rec)
        
        # 4. 광 노출 추천
        light_rec = await self._generate_light_exposure_recommendation(current_hour, circadian_phase)
        if light_rec:
            recommendations.append(light_rec)
        
        # 5. 기상 시간 추천 (아침)
        if 5 <= current_hour <= 9:
            wake_rec = await self._generate_wake_time_recommendation()
            if wake_rec:
                recommendations.append(wake_rec)
        
        self.last_recommendation_time = current_time
        return recommendations

    async def _generate_bedtime_recommendation(
        self, 
        sleep_debt: float, 
        circadian_phase: CircadianPhase
    ) -> Optional[SleepRecommendation]:
        """취침 시간 추천"""
        
        current_hour = datetime.datetime.now().hour
        optimal_bedtime = self.circadian_profile.optimal_bedtime
        
        # 수면 부채 고려한 취침 시간 조정
        adjusted_bedtime = optimal_bedtime
        if sleep_debt > 2.0:  # 2시간 이상 수면 부채
            adjusted_bedtime -= 0.5  # 30분 일찍 취침
        elif sleep_debt < -1.0:  # 1시간 이상 과다 수면
            adjusted_bedtime += 0.5  # 30분 늦게 취침
        
        # 현재 시간과 비교
        if current_hour >= adjusted_bedtime - 1 and current_hour <= adjusted_bedtime + 1:
            return SleepRecommendation(
                type="bedtime",
                message=f"최적 취침 시간이 {adjusted_bedtime:.1f}시입니다. 지금 잠자리에 드시는 것을 추천드려요.",
                optimal_time=adjusted_bedtime,
                priority=1 if abs(current_hour - adjusted_bedtime) < 0.5 else 2,
                reasoning=f"수면 부채: {sleep_debt:.1f}시간, 일주기 리듬 고려",
                expected_benefit="최적의 수면 질과 다음날 컨디션 향상"
            )
        
        return None

    async def _generate_nap_recommendation(
        self, 
        current_hour: int, 
        sleep_debt: float
    ) -> Optional[SleepRecommendation]:
        """파워 낮잠 추천"""
        
        # 낮잠 최적 시간 계산
        if 12 <= current_hour <= 15:  # 오후 12-3시
            
            # 낮잠 시간 결정 (10-30분)
            nap_duration = 20  # 기본 20분
            if sleep_debt > 3.0:
                nap_duration = 30  # 수면 부채가 많으면 30분
            elif self.current_fatigue_level == FatigueLevel.CRITICAL:
                nap_duration = 30
            
            # 밤잠에 미치는 영향 평가
            risk_level = "low"
            if current_hour > 15:
                risk_level = "medium"
            if nap_duration > 25:
                risk_level = "medium"
            
            return SleepRecommendation(
                type="nap",
                message=f"{nap_duration}분간의 파워 낮잠을 추천드려요. 피로 회복에 도움이 될 것입니다.",
                duration=nap_duration,
                priority=1 if self.current_fatigue_level == FatigueLevel.CRITICAL else 2,
                reasoning=f"현재 피로도: {self.current_fatigue_level.value}, 수면 부채: {sleep_debt:.1f}시간",
                expected_benefit=f"피로도 30-50% 감소, 밤잠 영향도: {risk_level}"
            )
        
        return None

    async def _generate_caffeine_recommendation(
        self, 
        current_hour: int, 
        fatigue_level: FatigueLevel
    ) -> Optional[SleepRecommendation]:
        """카페인 섭취 추천"""
        
        # 카페인 섭취 제한 시간 (취침 6시간 전)
        caffeine_cutoff = self.circadian_profile.optimal_bedtime - 6
        if caffeine_cutoff < 0:
            caffeine_cutoff += 24
        
        # 현재 시간이 제한 시간 이후라면 추천 안함
        if current_hour >= caffeine_cutoff and current_hour < self.circadian_profile.optimal_bedtime:
            return SleepRecommendation(
                type="caffeine",
                message="지금은 카페인 섭취를 피하시는 것이 좋겠어요. 밤잠에 영향을 줄 수 있습니다.",
                priority=2,
                reasoning=f"취침 {self.circadian_profile.optimal_bedtime - current_hour:.1f}시간 전",
                expected_benefit="밤잠 질 향상"
            )
        
        # 피로도가 높고 카페인 섭취 가능 시간이라면 추천
        if fatigue_level.value in ['high', 'critical'] and current_hour < caffeine_cutoff:
            
            # 개인 카페인 내성 고려
            caffeine_amount = "소량" if self.circadian_profile.caffeine_tolerance < 0.5 else "적당량"
            
            return SleepRecommendation(
                type="caffeine",
                message=f"{caffeine_amount}의 카페인 섭취로 각성도를 높이실 수 있어요. 하지만 {caffeine_cutoff:.0f}시 이후로는 피해주세요.",
                optimal_time=current_hour + 0.5,  # 30분 후
                priority=2,
                reasoning=f"현재 피로도: {fatigue_level.value}, 개인 내성: {self.circadian_profile.caffeine_tolerance}",
                expected_benefit="1-3시간 각성도 향상"
            )
        
        return None

    async def _generate_light_exposure_recommendation(
        self, 
        current_hour: int, 
        circadian_phase: CircadianPhase
    ) -> Optional[SleepRecommendation]:
        """광 노출 추천"""
        
        # 아침 광 노출 (일주기 리듬 조절)
        if 6 <= current_hour <= 10:
            return SleepRecommendation(
                type="light_exposure",
                message="밝은 빛에 노출되시면 일주기 리듬이 조절되어 밤잠이 좋아집니다.",
                duration=15,  # 15분
                priority=3,
                reasoning="아침 광 노출로 멜라토닌 분비 조절",
                expected_benefit="일주기 리듬 안정화, 밤잠 질 향상"
            )
        
        # 저녁 빛 차단 (멜라토닌 분비 촉진)
        elif current_hour >= 20:
            return SleepRecommendation(
                type="light_exposure",
                message="블루라이트를 차단하고 조명을 어둡게 하시면 자연스럽게 잠이 올 거예요.",
                priority=2,
                reasoning="멜라토닌 분비 촉진을 위한 빛 조절",
                expected_benefit="자연스러운 졸음 유도"
            )
        
        return None

    async def _generate_wake_time_recommendation(self) -> Optional[SleepRecommendation]:
        """기상 시간 추천"""
        
        if not self.sleep_history:
            return None
        
        # 최근 수면 패턴 분석
        recent_sleep = self.sleep_history[-7:] if len(self.sleep_history) >= 7 else self.sleep_history
        
        avg_bedtime = np.mean([sleep.bedtime for sleep in recent_sleep])
        avg_sleep_duration = np.mean([sleep.sleep_duration for sleep in recent_sleep])
        
        # 최적 기상 시간 계산
        optimal_wake_time = avg_bedtime + avg_sleep_duration
        
        current_time = time.time()
        current_hour = datetime.datetime.now().hour
        
        # 현재가 최적 기상 시간 근처라면 추천
        if abs(current_hour - optimal_wake_time) < 1.0:
            return SleepRecommendation(
                type="wake_time",
                message=f"지금이 최적 기상 시간입니다! 일어나시면 상쾌한 하루를 시작하실 수 있어요.",
                optimal_time=optimal_wake_time,
                priority=1,
                reasoning=f"평균 수면 패턴 기반 ({avg_sleep_duration:.1f}시간 수면)",
                expected_benefit="상쾌한 기상, 일주기 리듬 유지"
            )
        
        return None

    async def suggest_power_nap(self) -> Optional[PowerNapSuggestion]:
        """즉시 파워 낮잠 제안"""
        
        current_time = time.time()
        current_hour = datetime.datetime.now().hour
        
        # 낮잠 적절 시간 체크 (12-16시)
        if not (12 <= current_hour <= 16):
            return None
        
        # 현재 피로도가 높아야 함
        if self.current_fatigue_level.value not in ['high', 'critical']:
            return None
        
        # 최적 낮잠 시간 계산
        optimal_duration = 20  # 기본 20분
        
        if self.current_fatigue_level == FatigueLevel.CRITICAL:
            optimal_duration = 30
        
        # 수면 부채 고려
        sleep_debt = await self.sleep_debt_calculator.calculate_current_debt(self.sleep_history)
        if sleep_debt > 3.0:
            optimal_duration = min(30, optimal_duration + 10)
        
        # 피로 감소 예측
        fatigue_reduction = 0.4  # 기본 40%
        if optimal_duration >= 25:
            fatigue_reduction = 0.5
        if self.current_fatigue_level == FatigueLevel.CRITICAL:
            fatigue_reduction = 0.6
        
        # 밤잠 영향도 평가
        risk_assessment = "low"
        if current_hour > 15:
            risk_assessment = "medium"
        if optimal_duration > 25:
            risk_assessment = "medium"
        
        return PowerNapSuggestion(
            suggested_time=current_time,
            optimal_duration=optimal_duration,
            fatigue_reduction_expected=fatigue_reduction,
            risk_assessment=risk_assessment
        )

    async def track_sleep_session(self, bedtime: float, wake_time: float, quality_feedback: str):
        """수면 세션 추적 및 기록"""
        
        sleep_duration = (wake_time - bedtime) / 3600.0  # 시간 단위
        
        # 수면 품질 점수 계산 (피드백 기반)
        quality_scores = {
            "refreshed": 0.9,
            "good": 0.7,
            "okay": 0.5,
            "tired": 0.3,
            "groggy": 0.1
        }
        sleep_quality = quality_scores.get(quality_feedback, 0.5)
        
        # 수면 효율성 계산 (실제 수면 시간 / 침대에 있던 시간)
        # 실제로는 더 정교한 계산이 필요하지만 여기서는 단순화
        sleep_efficiency = min(1.0, sleep_duration / 9.0)  # 9시간 기준
        
        # 수면 단계 비율 추정 (실제로는 웨어러블 기기 데이터 활용)
        deep_sleep_ratio = 0.15 + (sleep_quality * 0.1)  # 15-25%
        rem_sleep_ratio = 0.20 + (sleep_quality * 0.05)   # 20-25%
        
        # 수면 데이터 생성
        sleep_data = SleepData(
            date=datetime.datetime.fromtimestamp(bedtime).strftime("%Y-%m-%d"),
            bedtime=bedtime,
            wake_time=wake_time,
            sleep_duration=sleep_duration,
            sleep_quality=sleep_quality,
            deep_sleep_ratio=deep_sleep_ratio,
            rem_sleep_ratio=rem_sleep_ratio,
            wake_up_feeling=quality_feedback,
            sleep_efficiency=sleep_efficiency
        )
        
        # 히스토리에 추가
        self.sleep_history.append(sleep_data)
        
        # 최근 30일만 유지
        if len(self.sleep_history) > 30:
            self.sleep_history = self.sleep_history[-30:]
        
        # 일주기 프로필 업데이트
        await self._update_circadian_profile(sleep_data)
        
        # 데이터 저장
        await self._save_sleep_data()
        
        print(f"😴 수면 세션 기록: {sleep_duration:.1f}시간, 품질: {sleep_quality:.2f}")

    async def _update_circadian_profile(self, sleep_data: SleepData):
        """일주기 프로필 업데이트"""
        
        if len(self.sleep_history) < 7:  # 최소 1주일 데이터 필요
            return
        
        recent_sleep = self.sleep_history[-7:]  # 최근 1주일
        
        # 평균 취침/기상 시간 계산
        avg_bedtime = np.mean([
            datetime.datetime.fromtimestamp(sleep.bedtime).hour + 
            datetime.datetime.fromtimestamp(sleep.bedtime).minute / 60.0 
            for sleep in recent_sleep
        ])
        
        avg_wake_time = np.mean([
            datetime.datetime.fromtimestamp(sleep.wake_time).hour + 
            datetime.datetime.fromtimestamp(sleep.wake_time).minute / 60.0 
            for sleep in recent_sleep
        ])
        
        avg_duration = np.mean([sleep.sleep_duration for sleep in recent_sleep])
        
        # 크로노타입 판정
        if avg_bedtime < 22.0 and avg_wake_time < 7.0:
            chronotype = "early"
        elif avg_bedtime > 24.0 and avg_wake_time > 8.0:
            chronotype = "late"
        else:
            chronotype = "normal"
        
        # 프로필 업데이트 (점진적 조정)
        self.circadian_profile.optimal_bedtime = (
            self.circadian_profile.optimal_bedtime * 0.7 + avg_bedtime * 0.3
        )
        self.circadian_profile.optimal_wake_time = (
            self.circadian_profile.optimal_wake_time * 0.7 + avg_wake_time * 0.3
        )
        self.circadian_profile.natural_sleep_duration = (
            self.circadian_profile.natural_sleep_duration * 0.7 + avg_duration * 0.3
        )
        self.circadian_profile.chronotype = chronotype

    async def _save_sleep_data(self):
        """수면 데이터 저장"""
        
        # 일주기 프로필 저장
        profile_path = Path(f"profiles/circadian_profile_{self.user_id}.json")
        profile_path.parent.mkdir(exist_ok=True)
        
        profile_data = {
            'user_id': self.circadian_profile.user_id,
            'chronotype': self.circadian_profile.chronotype,
            'optimal_bedtime': self.circadian_profile.optimal_bedtime,
            'optimal_wake_time': self.circadian_profile.optimal_wake_time,
            'natural_sleep_duration': self.circadian_profile.natural_sleep_duration,
            'energy_peaks': self.circadian_profile.energy_peaks,
            'energy_dips': self.circadian_profile.energy_dips,
            'light_sensitivity': self.circadian_profile.light_sensitivity,
            'caffeine_tolerance': self.circadian_profile.caffeine_tolerance,
            'last_updated': time.time()
        }
        
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, ensure_ascii=False, indent=2)
        
        # 수면 히스토리 저장
        history_path = Path(f"logs/sleep_history_{self.user_id}.json")
        history_path.parent.mkdir(exist_ok=True)
        
        history_data = [
            {
                'date': sleep.date,
                'bedtime': sleep.bedtime,
                'wake_time': sleep.wake_time,
                'sleep_duration': sleep.sleep_duration,
                'sleep_quality': sleep.sleep_quality,
                'deep_sleep_ratio': sleep.deep_sleep_ratio,
                'rem_sleep_ratio': sleep.rem_sleep_ratio,
                'wake_up_feeling': sleep.wake_up_feeling,
                'sleep_efficiency': sleep.sleep_efficiency
            } for sleep in self.sleep_history
        ]
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)

    def get_sleep_statistics(self) -> Dict[str, Any]:
        """수면 통계 정보 반환"""
        
        if not self.sleep_history:
            return {"status": "insufficient_data"}
        
        recent_sleep = self.sleep_history[-7:] if len(self.sleep_history) >= 7 else self.sleep_history
        
        # 기본 통계
        avg_duration = np.mean([sleep.sleep_duration for sleep in recent_sleep])
        avg_quality = np.mean([sleep.sleep_quality for sleep in recent_sleep])
        avg_efficiency = np.mean([sleep.sleep_efficiency for sleep in recent_sleep])
        
        # 수면 부채 계산
        current_debt = 0.0
        if len(recent_sleep) >= 3:
            target_duration = self.circadian_profile.natural_sleep_duration
            actual_durations = [sleep.sleep_duration for sleep in recent_sleep]
            current_debt = sum(target_duration - duration for duration in actual_durations)
        
        # 일관성 점수
        if len(recent_sleep) >= 3:
            bedtime_consistency = 1.0 - (np.std([
                datetime.datetime.fromtimestamp(sleep.bedtime).hour 
                for sleep in recent_sleep
            ]) / 12.0)  # 0-1 스케일
        else:
            bedtime_consistency = 0.5
        
        return {
            "chronotype": self.circadian_profile.chronotype,
            "average_sleep_duration": avg_duration,
            "average_sleep_quality": avg_quality,
            "average_sleep_efficiency": avg_efficiency,
            "sleep_debt_hours": current_debt,
            "bedtime_consistency": max(0.0, min(1.0, bedtime_consistency)),
            "optimal_bedtime": self.circadian_profile.optimal_bedtime,
            "optimal_wake_time": self.circadian_profile.optimal_wake_time,
            "current_fatigue_level": self.current_fatigue_level.value,
            "sleep_sessions_recorded": len(self.sleep_history)
        }


class CircadianRhythmTracker:
    """일주기 리듬 추적기"""
    
    def __init__(self, circadian_profile: CircadianProfile):
        self.profile = circadian_profile

    async def get_predicted_alertness(self, current_hour: float) -> float:
        """현재 시간의 예상 각성도 반환 (0-1)"""
        
        # 코사인 함수로 일주기 리듬 모델링
        # 각성도 피크: 오전 9시, 오후 7시
        # 각성도 저하: 오전 2시, 오후 2시
        
        # 24시간을 라디안으로 변환
        hour_rad = (current_hour / 24.0) * 2 * math.pi
        
        # 기본 일주기 리듬 (코사인 함수)
        base_alertness = 0.5 + 0.3 * math.cos(hour_rad - math.pi/3)  # 오전 6시 기준점
        
        # 개인 크로노타입 조정
        if self.profile.chronotype == "early":
            base_alertness += 0.1 * math.cos(hour_rad)  # 아침형 보정
        elif self.profile.chronotype == "late":
            base_alertness += 0.1 * math.cos(hour_rad - math.pi/2)  # 저녁형 보정
        
        # 오후 피로 (post-lunch dip) 추가
        if 13 <= current_hour <= 15:
            base_alertness -= 0.2
        
        return max(0.0, min(1.0, base_alertness))

    async def get_current_phase(self, current_hour: float) -> CircadianPhase:
        """현재 일주기 리듬 단계 반환"""
        
        if 6 <= current_hour <= 9:
            return CircadianPhase.MORNING_PEAK
        elif 13 <= current_hour <= 15:
            return CircadianPhase.MIDDAY_DIP
        elif 18 <= current_hour <= 21:
            return CircadianPhase.EVENING_PEAK
        elif 21 <= current_hour <= 23:
            return CircadianPhase.NIGHT_ONSET
        else:
            return CircadianPhase.DEEP_NIGHT


class SleepDebtCalculator:
    """수면 부채 계산기"""
    
    async def calculate_current_debt(self, sleep_history: List[SleepData]) -> float:
        """현재 수면 부채 계산 (시간 단위)"""
        
        if len(sleep_history) < 3:
            return 0.0
        
        # 최근 1주일간 수면 부채 누적
        recent_sleep = sleep_history[-7:] if len(sleep_history) >= 7 else sleep_history
        target_duration = 8.0  # 기본 목표 수면 시간
        
        total_debt = 0.0
        for sleep_data in recent_sleep:
            daily_debt = target_duration - sleep_data.sleep_duration
            total_debt += max(0, daily_debt)  # 음수 부채는 누적하지 않음
        
        return total_debt

    async def calculate_recovery_time(self, current_debt: float) -> float:
        """수면 부채 회복에 필요한 시간 계산"""
        
        # 하루에 1시간씩만 회복 가능하다고 가정
        return current_debt


class SleepRecommendationEngine:
    """수면 추천 엔진"""
    
    def __init__(self, circadian_profile: CircadianProfile):
        self.profile = circadian_profile

    async def generate_personalized_schedule(self) -> Dict[str, Any]:
        """개인화된 수면 스케줄 생성"""
        
        schedule = {
            "optimal_bedtime": self.profile.optimal_bedtime,
            "optimal_wake_time": self.profile.optimal_wake_time,
            "target_sleep_duration": self.profile.natural_sleep_duration,
            "recommended_nap_window": (13.0, 15.0),  # 오후 1-3시
            "caffeine_cutoff": self.profile.optimal_bedtime - 6,
            "light_exposure_morning": (7.0, 9.0),
            "light_restriction_evening": (21.0, 24.0)
        }
        
        return schedule