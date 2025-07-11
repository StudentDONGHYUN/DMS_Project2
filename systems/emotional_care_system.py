"""
S-Class DMS v19.0 - 멀티모달 감성 케어 시스템
운전자의 감정을 인식하고 차량의 오감을 통해 감정을 케어하는 지능형 시스템
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
import colorsys

from config.settings import get_config
from models.data_structures import UIState, EmotionState


class ModalityType(Enum):
    """감각 모달리티 유형"""
    VISUAL = "visual"      # 시각 (조명, 색상)
    AUDITORY = "auditory"  # 청각 (음악, 사운드)
    TACTILE = "tactile"    # 촉각 (진동, 마사지)
    OLFACTORY = "olfactory"  # 후각 (향수)
    THERMAL = "thermal"    # 온감 (온도)


class EmotionIntensity(Enum):
    """감정 강도"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5


class CareMode(Enum):
    """케어 모드"""
    RELAXATION = "relaxation"      # 이완 모드
    ENERGIZING = "energizing"      # 활성화 모드
    FOCUS = "focus"                # 집중 모드
    COMFORT = "comfort"            # 위로 모드
    STRESS_RELIEF = "stress_relief"  # 스트레스 해소
    MOOD_BOOST = "mood_boost"      # 기분 향상


@dataclass
class EmotionData:
    """감정 데이터"""
    primary_emotion: EmotionState
    intensity: EmotionIntensity
    valence: float  # -1.0(부정) ~ 1.0(긍정)
    arousal: float  # 0.0(낮음) ~ 1.0(높음)
    stress_level: float  # 0.0 ~ 1.0
    confidence: float = 0.8
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModalityAction:
    """모달리티 액션"""
    modality: ModalityType
    action_type: str
    parameters: Dict[str, Any]
    duration: float  # 지속시간 (초)
    intensity: float  # 강도 (0.0 ~ 1.0)
    fade_in_time: float = 1.0
    fade_out_time: float = 1.0


@dataclass
class CareSession:
    """감성 케어 세션"""
    session_id: str
    start_time: float
    target_emotion: EmotionState
    care_mode: CareMode
    active_modalities: List[ModalityType]
    actions: List[ModalityAction] = field(default_factory=list)
    effectiveness_score: float = 0.0
    end_time: Optional[float] = None


@dataclass
class PersonalEmotionProfile:
    """개인화된 감정 프로필"""
    user_id: str
    
    # 감정 반응 패턴
    emotion_triggers: Dict[str, List[str]] = field(default_factory=dict)
    effective_interventions: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    
    # 개인 선호도
    preferred_music_genres: List[str] = field(default_factory=list)
    preferred_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    preferred_scents: List[str] = field(default_factory=list)
    massage_preference: str = "medium"  # "light", "medium", "strong"
    
    # 학습된 패턴
    successful_care_patterns: List[Dict[str, Any]] = field(default_factory=list)
    
    last_updated: float = field(default_factory=time.time)


class EmotionalCareSystem:
    """멀티모달 감성 케어 메인 시스템"""
    
    def __init__(self, user_id: str = "default"):
        self.config = get_config()
        self.user_id = user_id
        
        # 감정 인식 엔진
        self.emotion_analyzer = EmotionAnalyzer()
        
        # 개인화 프로필
        self.emotion_profile = self._load_emotion_profile()
        
        # 모달리티 컨트롤러들
        self.modality_controllers = self._initialize_modality_controllers()
        
        # 현재 세션
        self.current_session: Optional[CareSession] = None
        
        # 감정 히스토리
        self.emotion_history = deque(maxlen=300)  # 10분 @ 30fps
        self.care_history = deque(maxlen=100)
        
        # 학습 엔진
        self.adaptation_engine = AdaptationEngine(self.emotion_profile)
        
        # 상태
        self.is_care_active = False
        self.current_care_mode = None
        
        print(f"🎭 감성 케어 시스템 초기화 완료 - 사용자: {user_id}")
        print(f"   지원 모달리티: {[m.value for m in ModalityType]}")
        print(f"   개인화 패턴: {len(self.emotion_profile.successful_care_patterns)}개")

    def _load_emotion_profile(self) -> PersonalEmotionProfile:
        """감정 프로필 로드"""
        profile_path = Path(f"profiles/emotion_profile_{self.user_id}.json")
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return PersonalEmotionProfile(
                        user_id=data.get('user_id', self.user_id),
                        emotion_triggers=data.get('emotion_triggers', {}),
                        effective_interventions=data.get('effective_interventions', {}),
                        preferred_music_genres=data.get('preferred_music_genres', []),
                        preferred_colors=data.get('preferred_colors', []),
                        preferred_scents=data.get('preferred_scents', []),
                        massage_preference=data.get('massage_preference', 'medium'),
                        successful_care_patterns=data.get('successful_care_patterns', []),
                        last_updated=data.get('last_updated', time.time())
                    )
            except Exception as e:
                print(f"감정 프로필 로드 실패: {e}")
        
        # 기본 프로필 생성
        return PersonalEmotionProfile(
            user_id=self.user_id,
            preferred_music_genres=["classical", "ambient", "nature"],
            preferred_colors=[(70, 130, 180), (100, 149, 237), (173, 216, 230)],
            preferred_scents=["lavender", "vanilla", "eucalyptus"]
        )

    def _save_emotion_profile(self):
        """감정 프로필 저장"""
        profile_path = Path(f"profiles/emotion_profile_{self.user_id}.json")
        profile_path.parent.mkdir(exist_ok=True)
        
        data = {
            'user_id': self.emotion_profile.user_id,
            'emotion_triggers': self.emotion_profile.emotion_triggers,
            'effective_interventions': self.emotion_profile.effective_interventions,
            'preferred_music_genres': self.emotion_profile.preferred_music_genres,
            'preferred_colors': self.emotion_profile.preferred_colors,
            'preferred_scents': self.emotion_profile.preferred_scents,
            'massage_preference': self.emotion_profile.massage_preference,
            'successful_care_patterns': self.emotion_profile.successful_care_patterns,
            'last_updated': self.emotion_profile.last_updated
        }
        
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _initialize_modality_controllers(self) -> Dict[ModalityType, Any]:
        """모달리티 컨트롤러 초기화"""
        controllers = {
            ModalityType.VISUAL: VisualController(),
            ModalityType.AUDITORY: AudioController(),
            ModalityType.TACTILE: TactileController(),
            ModalityType.OLFACTORY: OlfactoryController(),
            ModalityType.THERMAL: ThermalController()
        }
        return controllers

    async def process_emotion_data(self, ui_state: UIState) -> Optional[CareSession]:
        """감정 데이터 처리 및 케어 실행"""
        
        # 1. 감정 분석
        emotion_data = await self.emotion_analyzer.analyze_emotion(ui_state)
        self.emotion_history.append(emotion_data)
        
        # 2. 감정 상태 평가
        care_needed = await self._assess_care_necessity(emotion_data)
        
        if care_needed:
            # 3. 케어 모드 결정
            care_mode = await self._determine_care_mode(emotion_data)
            
            # 4. 개인화된 케어 계획 생성
            care_plan = await self._generate_care_plan(emotion_data, care_mode)
            
            # 5. 케어 세션 시작
            session = await self._start_care_session(care_mode, care_plan)
            
            return session
        
        return None

    async def _assess_care_necessity(self, emotion_data: EmotionData) -> bool:
        """케어 필요성 평가"""
        
        # 부정적 감정이 강한 경우
        if emotion_data.valence < -0.3 and emotion_data.intensity.value >= 3:
            return True
        
        # 스트레스 레벨이 높은 경우
        if emotion_data.stress_level > 0.6:
            return True
        
        # 각성 수준이 너무 높거나 낮은 경우
        if emotion_data.arousal > 0.8 or emotion_data.arousal < 0.2:
            return True
        
        # 위험한 감정 상태
        dangerous_emotions = [
            EmotionState.ANGER,
            EmotionState.STRESS_HIGH,
            EmotionState.FATIGUE_EXTREME
        ]
        if emotion_data.primary_emotion in dangerous_emotions:
            return True
        
        return False

    async def _determine_care_mode(self, emotion_data: EmotionData) -> CareMode:
        """케어 모드 결정"""
        
        # 감정 상태별 케어 모드 매핑
        if emotion_data.primary_emotion in [EmotionState.ANGER, EmotionState.STRESS_HIGH]:
            return CareMode.STRESS_RELIEF
        
        elif emotion_data.primary_emotion in [EmotionState.SADNESS, EmotionState.ANXIETY]:
            return CareMode.COMFORT
        
        elif emotion_data.primary_emotion == EmotionState.FATIGUE_EXTREME:
            if emotion_data.arousal < 0.3:
                return CareMode.ENERGIZING
            else:
                return CareMode.RELAXATION
        
        elif emotion_data.valence < -0.3:
            return CareMode.MOOD_BOOST
        
        elif emotion_data.arousal < 0.3 and emotion_data.valence > 0:
            return CareMode.FOCUS
        
        else:
            return CareMode.RELAXATION

    async def _generate_care_plan(self, emotion_data: EmotionData, care_mode: CareMode) -> List[ModalityAction]:
        """개인화된 케어 계획 생성"""
        care_plan = []
        
        # 케어 모드별 기본 액션
        if care_mode == CareMode.STRESS_RELIEF:
            care_plan.extend(await self._create_stress_relief_actions())
        
        elif care_mode == CareMode.RELAXATION:
            care_plan.extend(await self._create_relaxation_actions())
        
        elif care_mode == CareMode.ENERGIZING:
            care_plan.extend(await self._create_energizing_actions())
        
        elif care_mode == CareMode.COMFORT:
            care_plan.extend(await self._create_comfort_actions())
        
        elif care_mode == CareMode.MOOD_BOOST:
            care_plan.extend(await self._create_mood_boost_actions())
        
        elif care_mode == CareMode.FOCUS:
            care_plan.extend(await self._create_focus_actions())
        
        # 개인화 적용
        care_plan = await self._personalize_care_plan(care_plan, emotion_data)
        
        return care_plan

    async def _create_stress_relief_actions(self) -> List[ModalityAction]:
        """스트레스 해소 액션 생성"""
        actions = []
        
        # 시각: 차분한 파란색 조명
        actions.append(ModalityAction(
            modality=ModalityType.VISUAL,
            action_type="ambient_lighting",
            parameters={
                "color": (70, 130, 180),  # Steel Blue
                "brightness": 0.4,
                "fade_pattern": "slow_pulse"
            },
            duration=300.0,  # 5분
            intensity=0.6,
            fade_in_time=5.0
        ))
        
        # 청각: 자연 소리 + 명상 음악
        actions.append(ModalityAction(
            modality=ModalityType.AUDITORY,
            action_type="play_music",
            parameters={
                "playlist": "stress_relief",
                "volume": 0.3,
                "type": "nature_sounds"
            },
            duration=300.0,
            intensity=0.5
        ))
        
        # 촉각: 부드러운 진동 마사지
        actions.append(ModalityAction(
            modality=ModalityType.TACTILE,
            action_type="seat_massage",
            parameters={
                "pattern": "gentle_waves",
                "frequency": 1.2,
                "focus_areas": ["neck", "shoulders", "lower_back"]
            },
            duration=180.0,  # 3분
            intensity=0.4
        ))
        
        # 후각: 라벤더 향
        actions.append(ModalityAction(
            modality=ModalityType.OLFACTORY,
            action_type="release_scent",
            parameters={
                "scent": "lavender",
                "concentration": 0.3
            },
            duration=240.0,  # 4분
            intensity=0.5,
            fade_in_time=10.0
        ))
        
        return actions

    async def _create_relaxation_actions(self) -> List[ModalityAction]:
        """이완 액션 생성"""
        actions = []
        
        # 시각: 따뜻한 조명
        actions.append(ModalityAction(
            modality=ModalityType.VISUAL,
            action_type="ambient_lighting",
            parameters={
                "color": (255, 228, 196),  # Bisque
                "brightness": 0.35,
                "fade_pattern": "gentle_waves"
            },
            duration=600.0,  # 10분
            intensity=0.4
        ))
        
        # 청각: 클래식 음악
        actions.append(ModalityAction(
            modality=ModalityType.AUDITORY,
            action_type="play_music",
            parameters={
                "playlist": "classical_relaxation",
                "volume": 0.25,
                "tempo": "slow"
            },
            duration=600.0,
            intensity=0.4
        ))
        
        # 온감: 약간 따뜻하게
        actions.append(ModalityAction(
            modality=ModalityType.THERMAL,
            action_type="adjust_temperature",
            parameters={
                "target_temp": 23.5,
                "zone": "driver_seat"
            },
            duration=600.0,
            intensity=0.3
        ))
        
        return actions

    async def _create_energizing_actions(self) -> List[ModalityAction]:
        """활성화 액션 생성"""
        actions = []
        
        # 시각: 밝고 활기찬 조명
        actions.append(ModalityAction(
            modality=ModalityType.VISUAL,
            action_type="ambient_lighting",
            parameters={
                "color": (255, 215, 0),  # Gold
                "brightness": 0.8,
                "fade_pattern": "energetic_pulse"
            },
            duration=180.0,  # 3분
            intensity=0.8
        ))
        
        # 청각: 업비트 음악
        actions.append(ModalityAction(
            modality=ModalityType.AUDITORY,
            action_type="play_music",
            parameters={
                "playlist": "energizing",
                "volume": 0.4,
                "tempo": "upbeat"
            },
            duration=240.0,
            intensity=0.7
        ))
        
        # 후각: 상쾌한 유칼립투스
        actions.append(ModalityAction(
            modality=ModalityType.OLFACTORY,
            action_type="release_scent",
            parameters={
                "scent": "eucalyptus",
                "concentration": 0.4
            },
            duration=120.0,
            intensity=0.6
        ))
        
        # 촉각: 활성화 진동
        actions.append(ModalityAction(
            modality=ModalityType.TACTILE,
            action_type="seat_vibration",
            parameters={
                "pattern": "energizing_taps",
                "frequency": 2.0,
                "duration_pulse": 0.5
            },
            duration=60.0,
            intensity=0.6
        ))
        
        return actions

    async def _create_comfort_actions(self) -> List[ModalityAction]:
        """위로 액션 생성"""
        actions = []
        
        # 시각: 부드럽고 따뜻한 색상
        actions.append(ModalityAction(
            modality=ModalityType.VISUAL,
            action_type="ambient_lighting",
            parameters={
                "color": (255, 182, 193),  # Light Pink
                "brightness": 0.45,
                "fade_pattern": "heartbeat"
            },
            duration=420.0,  # 7분
            intensity=0.5
        ))
        
        # 청각: 감정적인 음악
        actions.append(ModalityAction(
            modality=ModalityType.AUDITORY,
            action_type="play_music",
            parameters={
                "playlist": "emotional_healing",
                "volume": 0.35,
                "type": "instrumental"
            },
            duration=420.0,
            intensity=0.6
        ))
        
        # 촉각: 안아주는 느낌의 마사지
        actions.append(ModalityAction(
            modality=ModalityType.TACTILE,
            action_type="seat_massage",
            parameters={
                "pattern": "embracing_hug",
                "frequency": 0.8,
                "pressure": "medium"
            },
            duration=300.0,
            intensity=0.6
        ))
        
        # 후각: 바닐라 향
        actions.append(ModalityAction(
            modality=ModalityType.OLFACTORY,
            action_type="release_scent",
            parameters={
                "scent": "vanilla",
                "concentration": 0.35
            },
            duration=360.0,
            intensity=0.5
        ))
        
        return actions

    async def _create_mood_boost_actions(self) -> List[ModalityAction]:
        """기분 향상 액션 생성"""
        actions = []
        
        # 시각: 밝고 쾌활한 색상
        actions.append(ModalityAction(
            modality=ModalityType.VISUAL,
            action_type="ambient_lighting",
            parameters={
                "color": (255, 255, 0),  # Yellow
                "brightness": 0.7,
                "fade_pattern": "rainbow_cycle"
            },
            duration=300.0,
            intensity=0.7
        ))
        
        # 청각: 기분 좋은 음악
        actions.append(ModalityAction(
            modality=ModalityType.AUDITORY,
            action_type="play_music",
            parameters={
                "playlist": "mood_boost",
                "volume": 0.45,
                "type": "positive_vibes"
            },
            duration=300.0,
            intensity=0.7
        ))
        
        return actions

    async def _create_focus_actions(self) -> List[ModalityAction]:
        """집중 액션 생성"""
        actions = []
        
        # 시각: 중성적인 집중 조명
        actions.append(ModalityAction(
            modality=ModalityType.VISUAL,
            action_type="ambient_lighting",
            parameters={
                "color": (255, 255, 255),  # White
                "brightness": 0.6,
                "fade_pattern": "steady"
            },
            duration=900.0,  # 15분
            intensity=0.5
        ))
        
        # 청각: 집중 음악 (바이노럴 비트)
        actions.append(ModalityAction(
            modality=ModalityType.AUDITORY,
            action_type="play_music",
            parameters={
                "playlist": "focus_enhancement",
                "volume": 0.3,
                "type": "binaural_beats",
                "frequency": "40Hz"  # 감마파
            },
            duration=900.0,
            intensity=0.4
        ))
        
        return actions

    async def _personalize_care_plan(self, care_plan: List[ModalityAction], 
                                   emotion_data: EmotionData) -> List[ModalityAction]:
        """케어 계획 개인화"""
        personalized_plan = []
        
        for action in care_plan:
            # 개인 선호도 적용
            if action.modality == ModalityType.VISUAL:
                action = await self._personalize_visual_action(action)
            
            elif action.modality == ModalityType.AUDITORY:
                action = await self._personalize_audio_action(action)
            
            elif action.modality == ModalityType.OLFACTORY:
                action = await self._personalize_scent_action(action)
            
            elif action.modality == ModalityType.TACTILE:
                action = await self._personalize_tactile_action(action)
            
            # 강도 조정 (감정 강도에 따라)
            action.intensity *= (emotion_data.intensity.value / 5.0)
            
            personalized_plan.append(action)
        
        return personalized_plan

    async def _personalize_visual_action(self, action: ModalityAction) -> ModalityAction:
        """시각 액션 개인화"""
        if self.emotion_profile.preferred_colors:
            # 선호 색상으로 변경
            preferred_color = random.choice(self.emotion_profile.preferred_colors)
            action.parameters["color"] = preferred_color
        
        return action

    async def _personalize_audio_action(self, action: ModalityAction) -> ModalityAction:
        """청각 액션 개인화"""
        if self.emotion_profile.preferred_music_genres:
            # 선호 장르 반영
            preferred_genre = random.choice(self.emotion_profile.preferred_music_genres)
            action.parameters["genre"] = preferred_genre
        
        return action

    async def _personalize_scent_action(self, action: ModalityAction) -> ModalityAction:
        """후각 액션 개인화"""
        if self.emotion_profile.preferred_scents:
            # 선호 향수로 변경
            preferred_scent = random.choice(self.emotion_profile.preferred_scents)
            action.parameters["scent"] = preferred_scent
        
        return action

    async def _personalize_tactile_action(self, action: ModalityAction) -> ModalityAction:
        """촉각 액션 개인화"""
        # 마사지 선호도 적용
        massage_pref = self.emotion_profile.massage_preference
        
        if massage_pref == "light":
            action.intensity *= 0.7
        elif massage_pref == "strong":
            action.intensity *= 1.3
        
        return action

    async def _start_care_session(self, care_mode: CareMode, 
                                 care_plan: List[ModalityAction]) -> CareSession:
        """케어 세션 시작"""
        session_id = f"care_session_{int(time.time())}"
        
        session = CareSession(
            session_id=session_id,
            start_time=time.time(),
            target_emotion=EmotionState.CALM,
            care_mode=care_mode,
            active_modalities=[action.modality for action in care_plan],
            actions=care_plan
        )
        
        self.current_session = session
        self.is_care_active = True
        self.current_care_mode = care_mode
        
        # 모달리티 컨트롤러에 액션 실행 지시
        await self._execute_care_actions(care_plan)
        
        print(f"🎭 감성 케어 세션 시작: {session_id}")
        print(f"   케어 모드: {care_mode.value}")
        print(f"   활성 모달리티: {[m.value for m in session.active_modalities]}")
        
        return session

    async def _execute_care_actions(self, care_plan: List[ModalityAction]):
        """케어 액션 실행"""
        for action in care_plan:
            controller = self.modality_controllers[action.modality]
            
            # 비동기로 액션 실행
            asyncio.create_task(
                controller.execute_action(action)
            )

    async def monitor_care_effectiveness(self, ui_state: UIState) -> float:
        """케어 효과성 모니터링"""
        if not self.current_session or not self.is_care_active:
            return 0.0
        
        # 현재 감정 분석
        current_emotion = await self.emotion_analyzer.analyze_emotion(ui_state)
        
        # 케어 시작 시점과 비교
        if len(self.emotion_history) >= 10:
            baseline_emotion = self.emotion_history[-10]  # 10프레임 전
            
            # 효과성 계산
            effectiveness = await self._calculate_effectiveness(
                baseline_emotion, current_emotion
            )
            
            self.current_session.effectiveness_score = effectiveness
            
            # 효과가 없거나 역효과인 경우 조정
            if effectiveness < 0.3:
                await self._adjust_care_strategy(current_emotion)
            
            return effectiveness
        
        return 0.5  # 기본값

    async def _calculate_effectiveness(self, baseline: EmotionData, 
                                     current: EmotionData) -> float:
        """케어 효과성 계산"""
        
        # 감정가(valence) 개선도
        valence_improvement = current.valence - baseline.valence
        
        # 스트레스 감소도
        stress_reduction = baseline.stress_level - current.stress_level
        
        # 각성도 안정화 (목표 각성도에 따라)
        target_arousal = 0.5  # 중간 수준이 목표
        arousal_stability = 1.0 - abs(current.arousal - target_arousal)
        
        # 종합 효과성
        effectiveness = (
            valence_improvement * 0.4 +
            stress_reduction * 0.4 +
            arousal_stability * 0.2
        )
        
        # 0-1 범위로 정규화
        return max(0.0, min(1.0, effectiveness + 0.5))

    async def _adjust_care_strategy(self, current_emotion: EmotionData):
        """케어 전략 조정"""
        if not self.current_session:
            return
        
        print(f"🔄 케어 전략 조정 중... 현재 효과성: {self.current_session.effectiveness_score:.2f}")
        
        # 현재 액션 중단
        await self._stop_current_actions()
        
        # 새로운 케어 모드 결정
        new_care_mode = await self._determine_alternative_care_mode(current_emotion)
        
        if new_care_mode != self.current_care_mode:
            # 새로운 케어 계획 생성
            new_care_plan = await self._generate_care_plan(current_emotion, new_care_mode)
            
            # 케어 계획 업데이트
            self.current_session.care_mode = new_care_mode
            self.current_session.actions.extend(new_care_plan)
            
            # 새로운 액션 실행
            await self._execute_care_actions(new_care_plan)
            
            self.current_care_mode = new_care_mode
            print(f"   새로운 케어 모드: {new_care_mode.value}")

    async def _determine_alternative_care_mode(self, emotion_data: EmotionData) -> CareMode:
        """대안 케어 모드 결정"""
        current_mode = self.current_care_mode
        
        # 현재 모드가 효과 없는 경우의 대안
        alternatives = {
            CareMode.STRESS_RELIEF: CareMode.RELAXATION,
            CareMode.RELAXATION: CareMode.ENERGIZING,
            CareMode.ENERGIZING: CareMode.FOCUS,
            CareMode.COMFORT: CareMode.MOOD_BOOST,
            CareMode.MOOD_BOOST: CareMode.RELAXATION,
            CareMode.FOCUS: CareMode.STRESS_RELIEF
        }
        
        return alternatives.get(current_mode, CareMode.RELAXATION)

    async def _stop_current_actions(self):
        """현재 실행 중인 액션 중단"""
        for modality, controller in self.modality_controllers.items():
            await controller.stop_all_actions()

    async def end_care_session(self) -> Dict[str, Any]:
        """케어 세션 종료"""
        if not self.current_session:
            return {"error": "활성 세션이 없습니다"}
        
        self.current_session.end_time = time.time()
        session_duration = self.current_session.end_time - self.current_session.start_time
        
        # 모든 액션 중단
        await self._stop_current_actions()
        
        # 세션 결과 저장
        await self._save_care_session()
        
        # 학습 데이터 업데이트
        await self.adaptation_engine.learn_from_session(self.current_session)
        
        result = {
            "session_duration": session_duration,
            "care_mode": self.current_session.care_mode.value,
            "effectiveness_score": self.current_session.effectiveness_score,
            "modalities_used": [m.value for m in self.current_session.active_modalities]
        }
        
        print(f"🎭 감성 케어 세션 종료")
        print(f"   지속시간: {session_duration:.1f}초")
        print(f"   효과성: {self.current_session.effectiveness_score:.2f}")
        
        self.current_session = None
        self.is_care_active = False
        self.current_care_mode = None
        
        return result

    async def _save_care_session(self):
        """케어 세션 저장"""
        if not self.current_session:
            return
        
        sessions_dir = Path("profiles/care_sessions")
        sessions_dir.mkdir(exist_ok=True)
        
        session_file = sessions_dir / f"{self.current_session.session_id}.json"
        
        session_data = {
            "session_id": self.current_session.session_id,
            "start_time": self.current_session.start_time,
            "end_time": self.current_session.end_time,
            "care_mode": self.current_session.care_mode.value,
            "effectiveness_score": self.current_session.effectiveness_score,
            "modalities_used": [m.value for m in self.current_session.active_modalities],
            "actions_count": len(self.current_session.actions)
        }
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

    def get_care_statistics(self) -> Dict[str, Any]:
        """케어 시스템 통계"""
        stats = {
            "is_active": self.is_care_active,
            "current_mode": self.current_care_mode.value if self.current_care_mode else None,
            "successful_patterns": len(self.emotion_profile.successful_care_patterns),
            "preferred_modalities": [],
            "effectiveness_history": []
        }
        
        # 선호 모달리티 분석
        modality_usage = {}
        for pattern in self.emotion_profile.successful_care_patterns:
            for modality in pattern.get("modalities", []):
                modality_usage[modality] = modality_usage.get(modality, 0) + 1
        
        stats["preferred_modalities"] = sorted(
            modality_usage.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        return stats


class EmotionAnalyzer:
    """감정 분석 엔진"""
    
    async def analyze_emotion(self, ui_state: UIState) -> EmotionData:
        """UI 상태로부터 감정 분석"""
        
        # 기본 감정 상태
        primary_emotion = ui_state.face.emotion_state
        
        # 감정 강도 계산
        intensity = self._calculate_emotion_intensity(ui_state)
        
        # 감정가 계산 (긍정/부정)
        valence = self._calculate_valence(ui_state)
        
        # 각성도 계산
        arousal = self._calculate_arousal(ui_state)
        
        # 스트레스 레벨
        stress_level = ui_state.biometrics.stress_level or 0.0
        
        return EmotionData(
            primary_emotion=primary_emotion,
            intensity=intensity,
            valence=valence,
            arousal=arousal,
            stress_level=stress_level
        )
    
    def _calculate_emotion_intensity(self, ui_state: UIState) -> EmotionIntensity:
        """감정 강도 계산"""
        # 여러 지표를 종합하여 강도 계산
        
        # 표정 변화 정도
        facial_intensity = getattr(ui_state.face, 'expression_intensity', 0.5)
        
        # 심박수 변화
        hr_baseline = 70  # 기준 심박수
        hr_current = ui_state.biometrics.heart_rate or hr_baseline
        hr_deviation = abs(hr_current - hr_baseline) / hr_baseline
        
        # 스트레스 레벨
        stress_level = ui_state.biometrics.stress_level or 0.0
        
        # 종합 강도
        combined_intensity = (facial_intensity + hr_deviation + stress_level) / 3.0
        
        if combined_intensity < 0.2:
            return EmotionIntensity.VERY_LOW
        elif combined_intensity < 0.4:
            return EmotionIntensity.LOW
        elif combined_intensity < 0.6:
            return EmotionIntensity.MEDIUM
        elif combined_intensity < 0.8:
            return EmotionIntensity.HIGH
        else:
            return EmotionIntensity.VERY_HIGH
    
    def _calculate_valence(self, ui_state: UIState) -> float:
        """감정가 계산 (긍정/부정)"""
        
        # 감정 상태별 기본 감정가
        emotion_valences = {
            EmotionState.HAPPINESS: 0.8,
            EmotionState.JOY: 1.0,
            EmotionState.CALM: 0.3,
            EmotionState.NEUTRAL: 0.0,
            EmotionState.SADNESS: -0.6,
            EmotionState.ANGER: -0.8,
            EmotionState.FEAR: -0.7,
            EmotionState.DISGUST: -0.5,
            EmotionState.SURPRISE: 0.1,
            EmotionState.ANXIETY: -0.6,
            EmotionState.STRESS_LOW: -0.3,
            EmotionState.STRESS_MEDIUM: -0.5,
            EmotionState.STRESS_HIGH: -0.8,
            EmotionState.FATIGUE_MILD: -0.2,
            EmotionState.FATIGUE_MODERATE: -0.4,
            EmotionState.FATIGUE_EXTREME: -0.6
        }
        
        base_valence = emotion_valences.get(ui_state.face.emotion_state, 0.0)
        
        # 스트레스 레벨로 조정
        stress_adjustment = -(ui_state.biometrics.stress_level or 0.0) * 0.3
        
        return max(-1.0, min(1.0, base_valence + stress_adjustment))
    
    def _calculate_arousal(self, ui_state: UIState) -> float:
        """각성도 계산"""
        
        # 심박수 기반 각성도
        hr_baseline = 70
        hr_current = ui_state.biometrics.heart_rate or hr_baseline
        hr_arousal = min(1.0, (hr_current - 50) / 50.0)
        
        # 주의력 기반 각성도
        attention_arousal = ui_state.gaze.attention_score
        
        # 평균 각성도
        arousal = (hr_arousal + attention_arousal) / 2.0
        
        return max(0.0, min(1.0, arousal))


class AdaptationEngine:
    """적응 학습 엔진"""
    
    def __init__(self, emotion_profile: PersonalEmotionProfile):
        self.emotion_profile = emotion_profile
    
    async def learn_from_session(self, session: CareSession):
        """세션으로부터 학습"""
        
        # 효과적인 세션인 경우 패턴 저장
        if session.effectiveness_score > 0.7:
            pattern = {
                "care_mode": session.care_mode.value,
                "modalities": [m.value for m in session.active_modalities],
                "effectiveness": session.effectiveness_score,
                "timestamp": session.end_time
            }
            
            self.emotion_profile.successful_care_patterns.append(pattern)
            
            # 최대 50개 패턴만 유지
            if len(self.emotion_profile.successful_care_patterns) > 50:
                self.emotion_profile.successful_care_patterns = \
                    self.emotion_profile.successful_care_patterns[-50:]
            
            print(f"📚 성공적인 케어 패턴 학습: {session.care_mode.value} (효과성: {session.effectiveness_score:.2f})")


# 모달리티 컨트롤러들

class VisualController:
    """시각 모달리티 컨트롤러"""
    
    def __init__(self):
        self.active_actions = {}
    
    async def execute_action(self, action: ModalityAction):
        """시각 액션 실행"""
        print(f"🎨 시각 액션 실행: {action.action_type}")
        print(f"   색상: {action.parameters.get('color', 'N/A')}")
        print(f"   밝기: {action.parameters.get('brightness', 'N/A')}")
        
        self.active_actions[action.action_type] = action
        
        # 실제 구현 시: LED 조명 시스템 제어
        # await self._control_ambient_lighting(action.parameters)
        
        # 지속시간 후 자동 종료
        await asyncio.sleep(action.duration)
        await self._fade_out_action(action)
    
    async def _fade_out_action(self, action: ModalityAction):
        """액션 페이드 아웃"""
        if action.action_type in self.active_actions:
            print(f"🎨 시각 액션 페이드 아웃: {action.action_type}")
            del self.active_actions[action.action_type]
    
    async def stop_all_actions(self):
        """모든 액션 중단"""
        for action_type in list(self.active_actions.keys()):
            print(f"🎨 시각 액션 중단: {action_type}")
            del self.active_actions[action_type]


class AudioController:
    """청각 모달리티 컨트롤러"""
    
    def __init__(self):
        self.active_actions = {}
        self.current_playlist = None
    
    async def execute_action(self, action: ModalityAction):
        """청각 액션 실행"""
        print(f"🎵 청각 액션 실행: {action.action_type}")
        print(f"   플레이리스트: {action.parameters.get('playlist', 'N/A')}")
        print(f"   볼륨: {action.parameters.get('volume', 'N/A')}")
        
        self.active_actions[action.action_type] = action
        self.current_playlist = action.parameters.get('playlist')
        
        # 실제 구현 시: 오디오 시스템 제어
        # await self._play_music(action.parameters)
        
        await asyncio.sleep(action.duration)
        await self._stop_music()
    
    async def _stop_music(self):
        """음악 중단"""
        print(f"🎵 음악 중단")
        self.current_playlist = None
    
    async def stop_all_actions(self):
        """모든 액션 중단"""
        await self._stop_music()
        self.active_actions.clear()


class TactileController:
    """촉각 모달리티 컨트롤러"""
    
    def __init__(self):
        self.active_actions = {}
    
    async def execute_action(self, action: ModalityAction):
        """촉각 액션 실행"""
        print(f"👋 촉각 액션 실행: {action.action_type}")
        print(f"   패턴: {action.parameters.get('pattern', 'N/A')}")
        print(f"   강도: {action.intensity}")
        
        self.active_actions[action.action_type] = action
        
        # 실제 구현 시: 시트 마사지/진동 시스템 제어
        # await self._control_seat_massage(action.parameters)
        
        await asyncio.sleep(action.duration)
        await self._stop_tactile_feedback()
    
    async def _stop_tactile_feedback(self):
        """촉각 피드백 중단"""
        print(f"👋 촉각 피드백 중단")
    
    async def stop_all_actions(self):
        """모든 액션 중단"""
        await self._stop_tactile_feedback()
        self.active_actions.clear()


class OlfactoryController:
    """후각 모달리티 컨트롤러"""
    
    def __init__(self):
        self.active_actions = {}
        self.current_scent = None
    
    async def execute_action(self, action: ModalityAction):
        """후각 액션 실행"""
        print(f"👃 후각 액션 실행: {action.action_type}")
        print(f"   향수: {action.parameters.get('scent', 'N/A')}")
        print(f"   농도: {action.parameters.get('concentration', 'N/A')}")
        
        self.active_actions[action.action_type] = action
        self.current_scent = action.parameters.get('scent')
        
        # 실제 구현 시: 아로마 디퓨저 제어
        # await self._release_scent(action.parameters)
        
        await asyncio.sleep(action.duration)
        await self._stop_scent_release()
    
    async def _stop_scent_release(self):
        """향수 방출 중단"""
        print(f"👃 향수 방출 중단")
        self.current_scent = None
    
    async def stop_all_actions(self):
        """모든 액션 중단"""
        await self._stop_scent_release()
        self.active_actions.clear()


class ThermalController:
    """온감 모달리티 컨트롤러"""
    
    def __init__(self):
        self.active_actions = {}
        self.current_temperature = 22.0  # 기본 온도
    
    async def execute_action(self, action: ModalityAction):
        """온감 액션 실행"""
        print(f"🌡️ 온감 액션 실행: {action.action_type}")
        print(f"   목표 온도: {action.parameters.get('target_temp', 'N/A')}°C")
        
        self.active_actions[action.action_type] = action
        target_temp = action.parameters.get('target_temp', 22.0)
        
        # 실제 구현 시: 시트 히팅/쿨링 시스템 제어
        # await self._adjust_seat_temperature(target_temp)
        
        self.current_temperature = target_temp
        
        await asyncio.sleep(action.duration)
        await self._reset_temperature()
    
    async def _reset_temperature(self):
        """온도 초기화"""
        print(f"🌡️ 온도 초기화")
        self.current_temperature = 22.0
    
    async def stop_all_actions(self):
        """모든 액션 중단"""
        await self._reset_temperature()
        self.active_actions.clear()