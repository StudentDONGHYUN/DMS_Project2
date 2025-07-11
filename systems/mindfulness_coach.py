"""
S-Class DMS v19.0 - 실시간 마음챙김 명상 코치
운전 중 스트레스와 감정 상태에 따른 맞춤형 명상 가이드 시스템
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
import random
import math

from config.settings import get_config
from models.data_structures import UIState, EmotionState


class MeditationType(Enum):
    """명상 유형"""
    BREATH_AWARENESS = "breath_awareness"      # 호흡 집중 명상
    BODY_SCAN = "body_scan"                   # 바디 스캔 명상
    LOVING_KINDNESS = "loving_kindness"       # 자애 명상
    STRESS_RELIEF = "stress_relief"           # 스트레스 해소 명상
    FOCUS_ENHANCEMENT = "focus_enhancement"    # 집중력 향상 명상
    MICRO_MEDITATION = "micro_meditation"      # 마이크로 명상 (신호대기용)


class MeditationIntensity(Enum):
    """명상 강도"""
    LIGHT = "light"           # 가벼운 명상 (1-2분)
    MODERATE = "moderate"     # 보통 명상 (3-5분)
    DEEP = "deep"            # 깊은 명상 (5-10분)
    MICRO = "micro"          # 마이크로 명상 (30초-1분)


class BreathingPattern(Enum):
    """호흡 패턴"""
    FOUR_SEVEN_EIGHT = "4-7-8"       # 4초 흡입, 7초 정지, 8초 호출
    FOUR_FOUR_FOUR = "4-4-4"         # 4초 흡입, 4초 정지, 4초 호출
    SIX_TWO_SIX = "6-2-6"           # 6초 흡입, 2초 정지, 6초 호출
    NATURAL = "natural"               # 자연스러운 호흡
    COHERENT = "coherent"             # 5초 흡입, 5초 호출 (심장박동 일치)


@dataclass
class MeditationSession:
    """명상 세션 데이터"""
    session_id: str
    meditation_type: MeditationType
    intensity: MeditationIntensity
    breathing_pattern: BreathingPattern
    start_time: float
    target_duration: float  # 목표 지속 시간 (초)
    actual_duration: Optional[float] = None
    effectiveness_score: float = 0.0
    interruption_count: int = 0
    user_feedback: Optional[str] = None
    physiological_improvement: Dict[str, float] = field(default_factory=dict)


@dataclass
class BreathingGuide:
    """호흡 가이드 데이터"""
    phase: str  # "inhale", "hold", "exhale"
    duration: float  # 해당 단계 지속 시간
    instruction: str  # 음성 가이드 텍스트
    visual_cue: Dict[str, Any]  # 시각적 가이드 (색상, 밝기 등)
    tactile_cue: Dict[str, Any]  # 촉각 가이드 (진동 패턴)


@dataclass
class MindfulnessProfile:
    """개인 마음챙김 프로필"""
    user_id: str
    preferred_meditation_types: List[MeditationType] = field(default_factory=list)
    preferred_breathing_pattern: BreathingPattern = BreathingPattern.NATURAL
    stress_triggers: List[str] = field(default_factory=list)
    effective_techniques: Dict[str, float] = field(default_factory=dict)
    meditation_history: List[Dict[str, Any]] = field(default_factory=list)
    total_meditation_time: float = 0.0  # 총 명상 시간 (분)
    consecutive_days: int = 0  # 연속 명상 일수
    mindfulness_level: int = 1  # 마음챙김 레벨 (1-10)
    last_updated: float = field(default_factory=time.time)


class MindfulnessCoach:
    """실시간 마음챙김 명상 코치 메인 시스템"""
    
    def __init__(self, user_id: str = "default"):
        self.config = get_config()
        self.user_id = user_id
        
        # 마음챙김 프로필 로드
        self.mindfulness_profile = self._load_mindfulness_profile()
        
        # 현재 세션
        self.current_session: Optional[MeditationSession] = None
        self.is_meditating = False
        
        # 실시간 데이터 버퍼
        self.stress_history = deque(maxlen=300)  # 10분 @ 30fps
        self.breathing_history = deque(maxlen=180)  # 6분 @ 30fps
        self.heart_rate_history = deque(maxlen=300)
        
        # 호흡 가이드 엔진
        self.breathing_guide_engine = BreathingGuideEngine()
        
        # 교통 상황 감지 (신호 대기 등)
        self.traffic_detector = TrafficSituationDetector()
        
        # 효과성 추적
        self.effectiveness_tracker = MeditationEffectivenessTracker()
        
        print(f"🧘 마음챙김 명상 코치 초기화 완료 - 사용자: {user_id}")
        print(f"   마음챙김 레벨: {self.mindfulness_profile.mindfulness_level}")
        print(f"   총 명상 경험: {self.mindfulness_profile.total_meditation_time:.1f}분")
        print(f"   연속 명상: {self.mindfulness_profile.consecutive_days}일")

    def _load_mindfulness_profile(self) -> MindfulnessProfile:
        """마음챙김 프로필 로드"""
        profile_path = Path(f"profiles/mindfulness_profile_{self.user_id}.json")
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return MindfulnessProfile(
                        user_id=data.get('user_id', self.user_id),
                        preferred_meditation_types=[
                            MeditationType(t) for t in data.get('preferred_meditation_types', [])
                        ],
                        preferred_breathing_pattern=BreathingPattern(
                            data.get('preferred_breathing_pattern', 'natural')
                        ),
                        stress_triggers=data.get('stress_triggers', []),
                        effective_techniques=data.get('effective_techniques', {}),
                        meditation_history=data.get('meditation_history', []),
                        total_meditation_time=data.get('total_meditation_time', 0.0),
                        consecutive_days=data.get('consecutive_days', 0),
                        mindfulness_level=data.get('mindfulness_level', 1),
                        last_updated=data.get('last_updated', time.time())
                    )
            except Exception as e:
                print(f"마음챙김 프로필 로드 실패: {e}")
        
        # 기본 프로필 생성
        return MindfulnessProfile(
            user_id=self.user_id,
            preferred_meditation_types=[MeditationType.BREATH_AWARENESS, MeditationType.STRESS_RELIEF],
            preferred_breathing_pattern=BreathingPattern.FOUR_FOUR_FOUR
        )

    def _save_mindfulness_profile(self):
        """마음챙김 프로필 저장"""
        profile_path = Path(f"profiles/mindfulness_profile_{self.user_id}.json")
        profile_path.parent.mkdir(exist_ok=True)
        
        data = {
            'user_id': self.mindfulness_profile.user_id,
            'preferred_meditation_types': [t.value for t in self.mindfulness_profile.preferred_meditation_types],
            'preferred_breathing_pattern': self.mindfulness_profile.preferred_breathing_pattern.value,
            'stress_triggers': self.mindfulness_profile.stress_triggers,
            'effective_techniques': self.mindfulness_profile.effective_techniques,
            'meditation_history': self.mindfulness_profile.meditation_history,
            'total_meditation_time': self.mindfulness_profile.total_meditation_time,
            'consecutive_days': self.mindfulness_profile.consecutive_days,
            'mindfulness_level': self.mindfulness_profile.mindfulness_level,
            'last_updated': self.mindfulness_profile.last_updated
        }
        
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def analyze_mindfulness_need(self, ui_state: UIState) -> Optional[MeditationType]:
        """마음챙김 필요성 분석 및 최적 명상 유형 추천"""
        
        # 스트레스 레벨 분석
        current_stress = ui_state.biometrics.stress_level or 0.0
        self.stress_history.append(current_stress)
        
        # 호흡 패턴 분석
        if hasattr(ui_state.biometrics, 'breathing_rate'):
            self.breathing_history.append(ui_state.biometrics.breathing_rate or 15.0)
        
        # 심박수 분석
        if ui_state.biometrics.heart_rate:
            self.heart_rate_history.append(ui_state.biometrics.heart_rate)
        
        # 교통 상황 감지
        traffic_situation = await self.traffic_detector.analyze_traffic_situation(ui_state)
        
        # 마음챙김 필요성 판단
        meditation_needed = await self._assess_meditation_need(current_stress, traffic_situation)
        
        if meditation_needed:
            # 최적 명상 유형 결정
            recommended_type = await self._recommend_meditation_type(
                current_stress, traffic_situation, ui_state
            )
            return recommended_type
        
        return None

    async def _assess_meditation_need(self, current_stress: float, traffic_situation: Dict[str, Any]) -> bool:
        """명상 필요성 평가"""
        
        # 높은 스트레스 레벨
        if current_stress > 0.7:
            return True
        
        # 지속적인 중간 수준 스트레스
        if len(self.stress_history) >= 60:  # 2분간
            recent_stress = list(self.stress_history)[-60:]
            if np.mean(recent_stress) > 0.5:
                return True
        
        # 신호 대기 등 명상하기 좋은 상황
        if traffic_situation.get('is_stopped', False) and traffic_situation.get('expected_wait_time', 0) > 30:
            return True
        
        # 호흡이 불규칙한 경우
        if len(self.breathing_history) >= 30:
            recent_breathing = list(self.breathing_history)[-30:]
            breathing_variability = np.std(recent_breathing)
            if breathing_variability > 3.0:  # 호흡 변동성이 높음
                return True
        
        # 심박수가 불안정한 경우
        if len(self.heart_rate_history) >= 30:
            recent_hr = list(self.heart_rate_history)[-30:]
            hr_variability = np.std(recent_hr)
            if hr_variability > 15.0:  # 심박 변동성이 높음
                return True
        
        return False

    async def _recommend_meditation_type(
        self, 
        current_stress: float, 
        traffic_situation: Dict[str, Any], 
        ui_state: UIState
    ) -> MeditationType:
        """최적 명상 유형 추천"""
        
        # 교통 상황 기반 추천
        if traffic_situation.get('is_stopped', False):
            wait_time = traffic_situation.get('expected_wait_time', 0)
            if wait_time > 120:  # 2분 이상 대기
                return MeditationType.BREATH_AWARENESS
            else:
                return MeditationType.MICRO_MEDITATION
        
        # 스트레스 레벨 기반 추천
        if current_stress > 0.8:
            return MeditationType.STRESS_RELIEF
        elif current_stress > 0.6:
            return MeditationType.BREATH_AWARENESS
        
        # 주의력 저하 감지 시
        if ui_state.gaze.attention_score < 0.6:
            return MeditationType.FOCUS_ENHANCEMENT
        
        # 피로 감지 시
        fatigue_score = 1.0 - ui_state.gaze.attention_score
        if fatigue_score > 0.7:
            return MeditationType.BODY_SCAN
        
        # 기본값: 호흡 집중 명상
        return MeditationType.BREATH_AWARENESS

    async def start_meditation_session(self, meditation_type: MeditationType) -> str:
        """명상 세션 시작"""
        
        if self.current_session:
            await self.end_meditation_session()
        
        # 명상 강도 결정
        intensity = await self._determine_meditation_intensity(meditation_type)
        
        # 호흡 패턴 선택
        breathing_pattern = await self._select_breathing_pattern(meditation_type, intensity)
        
        # 목표 지속 시간 설정
        target_duration = await self._calculate_target_duration(intensity, meditation_type)
        
        # 세션 생성
        session_id = f"meditation_{int(time.time())}"
        self.current_session = MeditationSession(
            session_id=session_id,
            meditation_type=meditation_type,
            intensity=intensity,
            breathing_pattern=breathing_pattern,
            start_time=time.time(),
            target_duration=target_duration
        )
        
        self.is_meditating = True
        
        # 호흡 가이드 시작
        await self.breathing_guide_engine.start_breathing_guide(
            breathing_pattern, target_duration
        )
        
        # 명상 시작 안내
        await self._announce_meditation_start(meditation_type, intensity)
        
        print(f"🧘 명상 세션 시작: {meditation_type.value} ({intensity.value})")
        print(f"   호흡 패턴: {breathing_pattern.value}")
        print(f"   목표 시간: {target_duration:.0f}초")
        
        return session_id

    async def _determine_meditation_intensity(self, meditation_type: MeditationType) -> MeditationIntensity:
        """명상 강도 결정"""
        
        if meditation_type == MeditationType.MICRO_MEDITATION:
            return MeditationIntensity.MICRO
        
        # 사용자 경험 레벨 고려
        if self.mindfulness_profile.mindfulness_level <= 3:
            return MeditationIntensity.LIGHT
        elif self.mindfulness_profile.mindfulness_level <= 6:
            return MeditationIntensity.MODERATE
        else:
            return MeditationIntensity.DEEP

    async def _select_breathing_pattern(
        self, 
        meditation_type: MeditationType, 
        intensity: MeditationIntensity
    ) -> BreathingPattern:
        """호흡 패턴 선택"""
        
        # 사용자 선호도 우선 고려
        if self.mindfulness_profile.preferred_breathing_pattern != BreathingPattern.NATURAL:
            return self.mindfulness_profile.preferred_breathing_pattern
        
        # 명상 유형별 최적 패턴
        if meditation_type == MeditationType.STRESS_RELIEF:
            return BreathingPattern.FOUR_SEVEN_EIGHT  # 스트레스 완화에 효과적
        elif meditation_type == MeditationType.FOCUS_ENHANCEMENT:
            return BreathingPattern.COHERENT  # 집중력 향상에 효과적
        elif meditation_type == MeditationType.MICRO_MEDITATION:
            return BreathingPattern.FOUR_FOUR_FOUR  # 간단하고 빠른 효과
        else:
            return BreathingPattern.SIX_TWO_SIX  # 일반적인 명상에 적합

    async def _calculate_target_duration(
        self, 
        intensity: MeditationIntensity, 
        meditation_type: MeditationType
    ) -> float:
        """목표 지속 시간 계산"""
        
        duration_map = {
            MeditationIntensity.MICRO: 45.0,      # 45초
            MeditationIntensity.LIGHT: 120.0,     # 2분
            MeditationIntensity.MODERATE: 300.0,  # 5분
            MeditationIntensity.DEEP: 600.0       # 10분
        }
        
        base_duration = duration_map.get(intensity, 120.0)
        
        # 마이크로 명상은 특별 처리
        if meditation_type == MeditationType.MICRO_MEDITATION:
            return 30.0  # 30초 고정
        
        return base_duration

    async def _announce_meditation_start(
        self, 
        meditation_type: MeditationType, 
        intensity: MeditationIntensity
    ):
        """명상 시작 안내"""
        
        announcements = {
            MeditationType.BREATH_AWARENESS: "호흡에 집중하는 명상을 시작합니다. 편안하게 호흡하며 현재 순간에 머물러 보세요.",
            MeditationType.STRESS_RELIEF: "스트레스 해소 명상을 시작합니다. 깊게 호흡하며 긴장을 내려놓으세요.",
            MeditationType.FOCUS_ENHANCEMENT: "집중력 향상 명상을 시작합니다. 호흡에 온전히 집중해보세요.",
            MeditationType.MICRO_MEDITATION: "잠깐의 마음챙김 시간입니다. 깊게 호흡하며 마음을 진정시켜보세요.",
            MeditationType.BODY_SCAN: "몸의 감각을 느끼는 명상을 시작합니다. 각 부위의 긴장을 확인하고 이완시켜보세요.",
            MeditationType.LOVING_KINDNESS: "자애 명상을 시작합니다. 자신과 타인에게 따뜻한 마음을 보내보세요."
        }
        
        message = announcements.get(meditation_type, "명상을 시작합니다.")
        
        # 실제 구현에서는 TTS 또는 음성 재생
        print(f"🔊 음성 가이드: {message}")

    async def monitor_meditation_progress(self, ui_state: UIState) -> Dict[str, Any]:
        """명상 진행 상황 모니터링"""
        
        if not self.current_session or not self.is_meditating:
            return {}
        
        # 현재 진행 상황
        elapsed_time = time.time() - self.current_session.start_time
        progress_percentage = min((elapsed_time / self.current_session.target_duration) * 100, 100.0)
        
        # 생리적 변화 추적
        physiological_changes = await self._track_physiological_changes(ui_state)
        
        # 주의산만 감지
        distraction_level = await self._detect_distraction(ui_state)
        
        # 세션 완료 체크
        if elapsed_time >= self.current_session.target_duration:
            await self.end_meditation_session()
            return {"session_completed": True}
        
        return {
            "session_active": True,
            "progress_percentage": progress_percentage,
            "elapsed_time": elapsed_time,
            "remaining_time": self.current_session.target_duration - elapsed_time,
            "physiological_changes": physiological_changes,
            "distraction_level": distraction_level,
            "breathing_guide": await self.breathing_guide_engine.get_current_guide()
        }

    async def _track_physiological_changes(self, ui_state: UIState) -> Dict[str, float]:
        """생리적 변화 추적"""
        
        if not self.current_session:
            return {}
        
        # 명상 시작 전 기준값과 비교
        session_start = self.current_session.start_time
        
        # 스트레스 레벨 변화
        current_stress = ui_state.biometrics.stress_level or 0.0
        baseline_stress = np.mean(list(self.stress_history)[:30]) if len(self.stress_history) >= 30 else current_stress
        stress_change = baseline_stress - current_stress
        
        # 심박수 변화
        current_hr = ui_state.biometrics.heart_rate or 70.0
        baseline_hr = np.mean(list(self.heart_rate_history)[:30]) if len(self.heart_rate_history) >= 30 else current_hr
        hr_change = baseline_hr - current_hr
        
        # 호흡 안정성
        breathing_stability = 0.0
        if len(self.breathing_history) >= 10:
            recent_breathing = list(self.breathing_history)[-10:]
            breathing_stability = 1.0 - (np.std(recent_breathing) / np.mean(recent_breathing))
        
        return {
            "stress_reduction": max(0.0, stress_change),
            "heart_rate_reduction": max(0.0, hr_change),
            "breathing_stability": max(0.0, min(1.0, breathing_stability))
        }

    async def _detect_distraction(self, ui_state: UIState) -> float:
        """주의산만 레벨 감지"""
        
        # 시선 분산도
        gaze_distraction = 1.0 - ui_state.gaze.attention_score
        
        # 자세 불안정성
        posture_instability = 1.0 - ui_state.posture.spinal_alignment_score
        
        # 손 움직임 (핸들 조작 등)
        hand_movement = ui_state.hands.tremor_frequency or 0.0
        
        # 종합 주의산만 점수
        overall_distraction = (gaze_distraction * 0.5 + 
                             posture_instability * 0.3 + 
                             min(hand_movement, 1.0) * 0.2)
        
        return min(1.0, overall_distraction)

    async def end_meditation_session(self) -> Dict[str, Any]:
        """명상 세션 종료"""
        
        if not self.current_session:
            return {"error": "활성 세션이 없습니다"}
        
        # 실제 지속 시간 기록
        self.current_session.actual_duration = time.time() - self.current_session.start_time
        
        # 효과성 평가
        effectiveness = await self.effectiveness_tracker.evaluate_session(self.current_session)
        self.current_session.effectiveness_score = effectiveness
        
        # 프로필 업데이트
        await self._update_mindfulness_profile()
        
        # 세션 저장
        await self._save_meditation_session()
        
        # 피드백 제공
        feedback = await self._generate_session_feedback()
        
        session_duration = self.current_session.actual_duration
        session_type = self.current_session.meditation_type.value
        
        print(f"🧘 명상 세션 완료: {session_type}")
        print(f"   지속 시간: {session_duration:.1f}초")
        print(f"   효과성 점수: {effectiveness:.2f}/1.0")
        
        result = {
            "session_completed": True,
            "session_duration": session_duration,
            "effectiveness_score": effectiveness,
            "meditation_type": session_type,
            "feedback": feedback
        }
        
        # 세션 정리
        self.current_session = None
        self.is_meditating = False
        await self.breathing_guide_engine.stop_breathing_guide()
        
        return result

    async def _update_mindfulness_profile(self):
        """마음챙김 프로필 업데이트"""
        
        if not self.current_session:
            return
        
        # 총 명상 시간 누적
        session_minutes = self.current_session.actual_duration / 60.0
        self.mindfulness_profile.total_meditation_time += session_minutes
        
        # 세션 기록 추가
        session_record = {
            "date": time.strftime("%Y-%m-%d", time.localtime()),
            "type": self.current_session.meditation_type.value,
            "duration": session_minutes,
            "effectiveness": self.current_session.effectiveness_score
        }
        self.mindfulness_profile.meditation_history.append(session_record)
        
        # 최근 20개만 유지
        if len(self.mindfulness_profile.meditation_history) > 20:
            self.mindfulness_profile.meditation_history = \
                self.mindfulness_profile.meditation_history[-20:]
        
        # 연속 명상 일수 계산
        await self._update_consecutive_days()
        
        # 마음챙김 레벨 업데이트
        await self._update_mindfulness_level()
        
        # 효과적인 기법 학습
        technique_key = f"{self.current_session.meditation_type.value}_{self.current_session.breathing_pattern.value}"
        if technique_key not in self.mindfulness_profile.effective_techniques:
            self.mindfulness_profile.effective_techniques[technique_key] = 0.0
        
        # 지수 평활법으로 효과성 업데이트
        current_effectiveness = self.mindfulness_profile.effective_techniques[technique_key]
        self.mindfulness_profile.effective_techniques[technique_key] = \
            0.7 * current_effectiveness + 0.3 * self.current_session.effectiveness_score

    async def _update_consecutive_days(self):
        """연속 명상 일수 업데이트"""
        today = time.strftime("%Y-%m-%d", time.localtime())
        
        if self.mindfulness_profile.meditation_history:
            last_session_date = self.mindfulness_profile.meditation_history[-1]["date"]
            
            if last_session_date == today:
                # 오늘 이미 명상했으면 연속 일수 유지
                pass
            else:
                # 어제 명상했으면 연속 일수 증가, 아니면 리셋
                yesterday = time.strftime(
                    "%Y-%m-%d", 
                    time.localtime(time.time() - 86400)
                )
                
                if last_session_date == yesterday:
                    self.mindfulness_profile.consecutive_days += 1
                else:
                    self.mindfulness_profile.consecutive_days = 1
        else:
            self.mindfulness_profile.consecutive_days = 1

    async def _update_mindfulness_level(self):
        """마음챙김 레벨 업데이트"""
        
        # 레벨업 기준
        total_hours = self.mindfulness_profile.total_meditation_time / 60.0
        consecutive_days = self.mindfulness_profile.consecutive_days
        avg_effectiveness = 0.0
        
        if self.mindfulness_profile.effective_techniques:
            avg_effectiveness = np.mean(list(self.mindfulness_profile.effective_techniques.values()))
        
        # 레벨 계산 (1-10)
        new_level = min(10, max(1, int(
            total_hours * 0.1 +  # 시간 기여도
            consecutive_days * 0.05 +  # 꾸준함 기여도
            avg_effectiveness * 3  # 효과성 기여도
        )))
        
        if new_level > self.mindfulness_profile.mindfulness_level:
            print(f"🎉 마음챙김 레벨업! {self.mindfulness_profile.mindfulness_level} → {new_level}")
            self.mindfulness_profile.mindfulness_level = new_level

    async def _save_meditation_session(self):
        """명상 세션 저장"""
        session_path = Path(f"logs/meditation_sessions_{self.user_id}.json")
        session_path.parent.mkdir(exist_ok=True)
        
        session_data = {
            "session_id": self.current_session.session_id,
            "timestamp": self.current_session.start_time,
            "type": self.current_session.meditation_type.value,
            "intensity": self.current_session.intensity.value,
            "breathing_pattern": self.current_session.breathing_pattern.value,
            "target_duration": self.current_session.target_duration,
            "actual_duration": self.current_session.actual_duration,
            "effectiveness_score": self.current_session.effectiveness_score,
            "interruption_count": self.current_session.interruption_count
        }
        
        # 기존 세션들 로드
        sessions = []
        if session_path.exists():
            try:
                with open(session_path, 'r', encoding='utf-8') as f:
                    sessions = json.load(f)
            except Exception:
                sessions = []
        
        sessions.append(session_data)
        
        # 최근 100개만 유지
        if len(sessions) > 100:
            sessions = sessions[-100:]
        
        with open(session_path, 'w', encoding='utf-8') as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)

    async def _generate_session_feedback(self) -> str:
        """세션 피드백 생성"""
        
        if not self.current_session:
            return "피드백을 생성할 수 없습니다."
        
        effectiveness = self.current_session.effectiveness_score
        duration = self.current_session.actual_duration / 60.0  # 분 단위
        
        if effectiveness >= 0.8:
            feedback = f"훌륭한 명상이었습니다! {duration:.1f}분 동안 깊은 집중 상태를 유지하셨네요."
        elif effectiveness >= 0.6:
            feedback = f"좋은 명상이었습니다. {duration:.1f}분 동안 마음챙김을 잘 실천하셨어요."
        elif effectiveness >= 0.4:
            feedback = f"괜찮은 명상이었습니다. {duration:.1f}분의 시간을 자신에게 투자하신 것만으로도 의미가 있어요."
        else:
            feedback = f"명상을 시도해주셔서 감사합니다. 처음에는 집중이 어려울 수 있어요. 꾸준히 연습하시면 더 나아질 거예요."
        
        # 개선 제안 추가
        if self.current_session.interruption_count > 3:
            feedback += " 다음번에는 더 조용한 환경에서 명상해보시는 것을 추천드려요."
        
        if duration < self.current_session.target_duration / 60.0 * 0.5:
            feedback += " 조금 더 오래 명상을 지속해보시면 더 큰 효과를 느끼실 수 있을 거예요."
        
        return feedback

    def get_mindfulness_statistics(self) -> Dict[str, Any]:
        """마음챙김 통계 정보 반환"""
        
        return {
            "mindfulness_level": self.mindfulness_profile.mindfulness_level,
            "total_meditation_time_hours": self.mindfulness_profile.total_meditation_time / 60.0,
            "consecutive_days": self.mindfulness_profile.consecutive_days,
            "session_count": len(self.mindfulness_profile.meditation_history),
            "average_effectiveness": np.mean([
                s["effectiveness"] for s in self.mindfulness_profile.meditation_history
            ]) if self.mindfulness_profile.meditation_history else 0.0,
            "preferred_meditation_type": self.mindfulness_profile.preferred_meditation_types[0].value if self.mindfulness_profile.preferred_meditation_types else "none",
            "most_effective_technique": max(
                self.mindfulness_profile.effective_techniques.items(),
                key=lambda x: x[1]
            )[0] if self.mindfulness_profile.effective_techniques else "none"
        }


class BreathingGuideEngine:
    """호흡 가이드 엔진"""
    
    def __init__(self):
        self.is_active = False
        self.current_pattern: Optional[BreathingPattern] = None
        self.guide_task: Optional[asyncio.Task] = None

    async def start_breathing_guide(self, pattern: BreathingPattern, duration: float):
        """호흡 가이드 시작"""
        self.current_pattern = pattern
        self.is_active = True
        
        self.guide_task = asyncio.create_task(
            self._run_breathing_guide(pattern, duration)
        )

    async def _run_breathing_guide(self, pattern: BreathingPattern, duration: float):
        """호흡 가이드 실행"""
        
        # 패턴별 타이밍 설정
        pattern_timings = {
            BreathingPattern.FOUR_SEVEN_EIGHT: (4, 7, 8),
            BreathingPattern.FOUR_FOUR_FOUR: (4, 4, 4),
            BreathingPattern.SIX_TWO_SIX: (6, 2, 6),
            BreathingPattern.COHERENT: (5, 0, 5),
            BreathingPattern.NATURAL: (4, 1, 6)
        }
        
        inhale_time, hold_time, exhale_time = pattern_timings.get(pattern, (4, 4, 4))
        cycle_time = inhale_time + hold_time + exhale_time
        
        start_time = time.time()
        
        while self.is_active and (time.time() - start_time) < duration:
            try:
                # 흡입 단계
                await self._guide_inhale(inhale_time)
                
                if not self.is_active:
                    break
                
                # 정지 단계 (있는 경우)
                if hold_time > 0:
                    await self._guide_hold(hold_time)
                
                if not self.is_active:
                    break
                
                # 호출 단계
                await self._guide_exhale(exhale_time)
                
            except asyncio.CancelledError:
                break

    async def _guide_inhale(self, duration: float):
        """흡입 가이드"""
        print(f"🌬️  흡입 시작 ({duration}초)")
        # 실제로는 LED 밝기 증가, 진동 패턴, 음성 가이드 등
        await asyncio.sleep(duration)

    async def _guide_hold(self, duration: float):
        """정지 가이드"""
        print(f"⏸️  호흡 정지 ({duration}초)")
        await asyncio.sleep(duration)

    async def _guide_exhale(self, duration: float):
        """호출 가이드"""
        print(f"🌪️  호출 시작 ({duration}초)")
        # 실제로는 LED 밝기 감소, 진동 패턴, 음성 가이드 등
        await asyncio.sleep(duration)

    async def get_current_guide(self) -> Dict[str, Any]:
        """현재 가이드 상태 반환"""
        return {
            "is_active": self.is_active,
            "current_pattern": self.current_pattern.value if self.current_pattern else None
        }

    async def stop_breathing_guide(self):
        """호흡 가이드 중지"""
        self.is_active = False
        if self.guide_task:
            self.guide_task.cancel()
            try:
                await self.guide_task
            except asyncio.CancelledError:
                pass


class TrafficSituationDetector:
    """교통 상황 감지기"""
    
    async def analyze_traffic_situation(self, ui_state: UIState) -> Dict[str, Any]:
        """교통 상황 분석"""
        
        # 실제로는 차량 센서, GPS, 카메라 등을 활용
        # 여기서는 간단한 시뮬레이션
        
        # 차량 속도 (임시로 0으로 가정)
        vehicle_speed = 0.0  # km/h
        
        # 정지 상태 판단
        is_stopped = vehicle_speed < 5.0
        
        # 대기 시간 예측 (신호등 패턴 분석 등)
        expected_wait_time = 0.0
        if is_stopped:
            expected_wait_time = random.uniform(30, 180)  # 30초-3분
        
        return {
            "is_stopped": is_stopped,
            "vehicle_speed": vehicle_speed,
            "expected_wait_time": expected_wait_time,
            "traffic_density": "low"  # low, medium, high
        }


class MeditationEffectivenessTracker:
    """명상 효과성 추적기"""
    
    async def evaluate_session(self, session: MeditationSession) -> float:
        """세션 효과성 평가"""
        
        # 기본 점수 (완료도 기준)
        completion_ratio = session.actual_duration / session.target_duration if session.target_duration > 0 else 0
        base_score = min(1.0, completion_ratio)
        
        # 중단 횟수 벌점
        interruption_penalty = min(0.5, session.interruption_count * 0.1)
        
        # 생리적 개선도 가점
        physiological_bonus = 0.0
        if session.physiological_improvement:
            avg_improvement = np.mean(list(session.physiological_improvement.values()))
            physiological_bonus = min(0.3, avg_improvement * 0.3)
        
        # 최종 점수 계산
        final_score = max(0.0, min(1.0, base_score - interruption_penalty + physiological_bonus))
        
        return final_score