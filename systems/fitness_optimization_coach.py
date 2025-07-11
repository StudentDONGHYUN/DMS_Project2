"""
S-Class DMS v19.0 - 운전자 체력 최적화 코치
장거리 운전자의 체력 관리와 운동 부족 해소를 위한 통합 솔루션입니다.
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
import random

from config.settings import get_config
from models.data_structures import UIState, PostureData


class ExerciseType(Enum):
    """운동 유형"""
    NECK_STRETCH = "neck_stretch"              # 목 스트레칭
    SHOULDER_ROLL = "shoulder_roll"            # 어깨 돌리기
    BACK_ARCH = "back_arch"                   # 등 젖히기
    SPINAL_TWIST = "spinal_twist"             # 척추 비틀기
    ANKLE_PUMP = "ankle_pump"                 # 발목 펌프
    CALF_RAISE = "calf_raise"                 # 종아리 들기
    DEEP_BREATHING = "deep_breathing"          # 심호흡
    EYE_EXERCISE = "eye_exercise"             # 눈 운동
    HAND_STRETCH = "hand_stretch"             # 손목 스트레칭
    HIP_FLEX = "hip_flex"                     # 엉덩이 굽히기


class ExerciseIntensity(Enum):
    """운동 강도"""
    GENTLE = "gentle"                         # 부드러운 (초보자)
    MODERATE = "moderate"                     # 보통 (중급자)
    VIGOROUS = "vigorous"                     # 활발한 (고급자)


class PostureIssue(Enum):
    """자세 문제"""
    FORWARD_HEAD = "forward_head"             # 거북목
    ROUNDED_SHOULDERS = "rounded_shoulders"    # 둥근 어깨
    SLOUCHED_BACK = "slouched_back"           # 구부정한 등
    TILTED_PELVIS = "tilted_pelvis"           # 기울어진 골반
    POOR_LUMBAR_SUPPORT = "poor_lumbar_support"  # 허리 지지 부족


class FitnessGoal(Enum):
    """체력 목표"""
    PAIN_RELIEF = "pain_relief"               # 통증 완화
    FLEXIBILITY = "flexibility"               # 유연성 향상
    STRENGTH = "strength"                     # 근력 강화
    ENDURANCE = "endurance"                   # 지구력 향상
    POSTURE_CORRECTION = "posture_correction"  # 자세 교정
    CIRCULATION = "circulation"               # 혈액순환 개선


@dataclass
class ExerciseRoutine:
    """운동 루틴"""
    routine_id: str
    exercise_type: ExerciseType
    intensity: ExerciseIntensity
    duration: float  # 초 단위
    repetitions: int
    rest_intervals: float  # 세트 간 휴식 (초)
    instructions: List[str]  # 단계별 지시사항
    target_muscles: List[str]  # 대상 근육
    benefits: List[str]  # 기대 효과
    precautions: List[str]  # 주의사항


@dataclass
class ExerciseSession:
    """운동 세션"""
    session_id: str
    start_time: float
    exercises: List[ExerciseRoutine]
    total_duration: float
    completion_rate: float = 0.0  # 완료율
    effectiveness_rating: Optional[int] = None  # 1-5 효과 평가
    discomfort_level: Optional[int] = None  # 1-5 불편함 정도
    end_time: Optional[float] = None


@dataclass
class PhysicalProfile:
    """신체 프로필"""
    user_id: str
    age: Optional[int] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    fitness_level: str = "beginner"  # "beginner", "intermediate", "advanced"
    
    # 체력 목표
    primary_goals: List[FitnessGoal] = field(default_factory=list)
    
    # 신체 제약사항
    physical_limitations: List[str] = field(default_factory=list)
    injury_history: List[str] = field(default_factory=list)
    
    # 운전 패턴
    average_driving_hours_per_day: float = 2.0
    longest_continuous_drive: float = 4.0  # 최장 연속 운전 시간
    
    # 진행 상황
    total_exercise_time: float = 0.0  # 총 운동 시간 (분)
    consecutive_exercise_days: int = 0
    fitness_score: float = 0.5  # 0-1 체력 점수
    
    last_updated: float = field(default_factory=time.time)


@dataclass
class PostureAnalysis:
    """자세 분석 결과"""
    timestamp: float
    overall_score: float  # 전체 자세 점수 (0-1)
    identified_issues: List[PostureIssue]
    severity_scores: Dict[PostureIssue, float]  # 문제별 심각도
    improvement_suggestions: List[str]
    risk_assessment: str  # "low", "medium", "high"


@dataclass
class CirculationMetrics:
    """혈액순환 지표"""
    timestamp: float
    sitting_duration: float  # 연속 앉은 시간 (분)
    movement_frequency: float  # 움직임 빈도
    lower_body_stiffness: float  # 하체 경직도 (0-1)
    swelling_risk: float  # 부종 위험도 (0-1)


class FitnessOptimizationCoach:
    """운전자 체력 최적화 코치 메인 시스템"""
    
    def __init__(self, user_id: str = "default"):
        self.config = get_config()
        self.user_id = user_id
        
        # 신체 프로필 로드
        self.physical_profile = self._load_physical_profile()
        
        # 운동 데이터베이스
        self.exercise_database = self._initialize_exercise_database()
        
        # 현재 세션
        self.current_session: Optional[ExerciseSession] = None
        
        # 실시간 데이터 추적
        self.posture_history = deque(maxlen=1800)  # 30분 @ 1Hz
        self.circulation_history = deque(maxlen=300)  # 10분
        self.exercise_history = self._load_exercise_history()
        
        # 자세 분석기
        self.posture_analyzer = PostureAnalyzer()
        
        # 혈액순환 모니터
        self.circulation_monitor = CirculationMonitor()
        
        # 운동 추천 엔진
        self.exercise_recommender = ExerciseRecommendationEngine(self.physical_profile)
        
        # 상태 추적
        self.last_exercise_reminder = 0.0
        self.current_sitting_time = 0.0
        self.movement_alerts_today = 0
        
        print(f"💪 체력 최적화 코치 초기화 완료 - 사용자: {user_id}")
        print(f"   체력 레벨: {self.physical_profile.fitness_level}")
        print(f"   주요 목표: {[goal.value for goal in self.physical_profile.primary_goals]}")
        print(f"   총 운동 시간: {self.physical_profile.total_exercise_time:.1f}분")

    def _load_physical_profile(self) -> PhysicalProfile:
        """신체 프로필 로드"""
        profile_path = Path(f"profiles/physical_profile_{self.user_id}.json")
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return PhysicalProfile(
                        user_id=data.get('user_id', self.user_id),
                        age=data.get('age'),
                        height_cm=data.get('height_cm'),
                        weight_kg=data.get('weight_kg'),
                        fitness_level=data.get('fitness_level', 'beginner'),
                        primary_goals=[FitnessGoal(goal) for goal in data.get('primary_goals', [])],
                        physical_limitations=data.get('physical_limitations', []),
                        injury_history=data.get('injury_history', []),
                        average_driving_hours_per_day=data.get('average_driving_hours_per_day', 2.0),
                        longest_continuous_drive=data.get('longest_continuous_drive', 4.0),
                        total_exercise_time=data.get('total_exercise_time', 0.0),
                        consecutive_exercise_days=data.get('consecutive_exercise_days', 0),
                        fitness_score=data.get('fitness_score', 0.5),
                        last_updated=data.get('last_updated', time.time())
                    )
            except Exception as e:
                print(f"신체 프로필 로드 실패: {e}")
        
        # 기본 프로필 생성
        return PhysicalProfile(
            user_id=self.user_id,
            primary_goals=[FitnessGoal.POSTURE_CORRECTION, FitnessGoal.CIRCULATION]
        )

    def _load_exercise_history(self) -> List[ExerciseSession]:
        """운동 히스토리 로드"""
        history_path = Path(f"logs/exercise_history_{self.user_id}.json")
        
        if history_path.exists():
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return [
                        ExerciseSession(
                            session_id=entry['session_id'],
                            start_time=entry['start_time'],
                            exercises=[],  # 간단화를 위해 생략
                            total_duration=entry['total_duration'],
                            completion_rate=entry['completion_rate'],
                            effectiveness_rating=entry.get('effectiveness_rating'),
                            discomfort_level=entry.get('discomfort_level'),
                            end_time=entry.get('end_time')
                        ) for entry in data
                    ]
            except Exception as e:
                print(f"운동 히스토리 로드 실패: {e}")
        
        return []

    def _initialize_exercise_database(self) -> Dict[ExerciseType, ExerciseRoutine]:
        """운동 데이터베이스 초기화"""
        
        exercises = {}
        
        # 목 스트레칭
        exercises[ExerciseType.NECK_STRETCH] = ExerciseRoutine(
            routine_id="neck_stretch_basic",
            exercise_type=ExerciseType.NECK_STRETCH,
            intensity=ExerciseIntensity.GENTLE,
            duration=60.0,
            repetitions=5,
            rest_intervals=10.0,
            instructions=[
                "어깨를 이완하고 똑바로 앉으세요",
                "고개를 천천히 오른쪽으로 기울이세요 (10초 유지)",
                "중앙으로 돌아온 후 왼쪽으로 기울이세요 (10초 유지)",
                "앞뒤로도 천천히 움직여주세요",
                "동작은 부드럽게, 절대 급격하게 하지 마세요"
            ],
            target_muscles=["목 근육", "승모근", "후두하근"],
            benefits=["목 긴장 완화", "거북목 예방", "혈액순환 개선"],
            precautions=["목 디스크가 있으면 의사와 상담", "통증이 있으면 즉시 중단"]
        )
        
        # 어깨 돌리기
        exercises[ExerciseType.SHOULDER_ROLL] = ExerciseRoutine(
            routine_id="shoulder_roll_basic",
            exercise_type=ExerciseType.SHOULDER_ROLL,
            intensity=ExerciseIntensity.GENTLE,
            duration=45.0,
            repetitions=10,
            rest_intervals=5.0,
            instructions=[
                "양 어깨를 편안하게 이완하세요",
                "어깨를 앞에서 뒤로 천천히 돌리세요",
                "큰 원을 그리듯이 부드럽게 움직이세요",
                "반대 방향으로도 같은 횟수만큼 돌리세요",
                "동작 중 깊게 호흡하세요"
            ],
            target_muscles=["삼각근", "승모근", "능형근"],
            benefits=["어깨 긴장 해소", "둥근 어깨 교정", "상체 혈액순환"],
            precautions=["어깨 탈구 병력이 있으면 주의", "통증 시 즉시 중단"]
        )
        
        # 등 젖히기
        exercises[ExerciseType.BACK_ARCH] = ExerciseRoutine(
            routine_id="back_arch_seated",
            exercise_type=ExerciseType.BACK_ARCH,
            intensity=ExerciseIntensity.MODERATE,
            duration=30.0,
            repetitions=8,
            rest_intervals=15.0,
            instructions=[
                "의자에 깊숙이 앉아 등받이에 기대세요",
                "양손을 머리 뒤로 깍지 끼세요",
                "천천히 가슴을 펴며 등을 뒤로 젖히세요",
                "5초간 유지한 후 천천히 원위치",
                "호흡을 멈추지 말고 자연스럽게 하세요"
            ],
            target_muscles=["척추기립근", "광배근", "능형근"],
            benefits=["척추 유연성", "자세 교정", "등 통증 완화"],
            precautions=["허리 디스크 주의", "과도한 신전 금지"]
        )
        
        # 발목 펌프
        exercises[ExerciseType.ANKLE_PUMP] = ExerciseRoutine(
            routine_id="ankle_pump_seated",
            exercise_type=ExerciseType.ANKLE_PUMP,
            intensity=ExerciseIntensity.GENTLE,
            duration=60.0,
            repetitions=15,
            rest_intervals=5.0,
            instructions=[
                "발뒤꿈치를 바닥에 고정하세요",
                "발끝을 위로 올렸다가 아래로 내리세요",
                "발목을 시계방향으로 천천히 돌리세요",
                "반시계방향으로도 같은 횟수만큼 돌리세요",
                "양발을 번갈아가며 실시하세요"
            ],
            target_muscles=["종아리근", "전경골근", "발목 주변근"],
            benefits=["혈액순환", "부종 방지", "하지정맥류 예방"],
            precautions=["발목 부상 시 주의", "급격한 동작 금지"]
        )
        
        # 눈 운동
        exercises[ExerciseType.EYE_EXERCISE] = ExerciseRoutine(
            routine_id="eye_exercise_basic",
            exercise_type=ExerciseType.EYE_EXERCISE,
            intensity=ExerciseIntensity.GENTLE,
            duration=90.0,
            repetitions=1,
            rest_intervals=0.0,
            instructions=[
                "먼 곳(100m 이상)을 20초간 바라보세요",
                "눈을 감고 10초간 휴식하세요",
                "눈동자를 상하좌우로 천천히 움직이세요",
                "시계방향, 반시계방향으로 눈동자를 돌리세요",
                "의식적으로 깜박임을 10회 반복하세요"
            ],
            target_muscles=["안구근육", "외안근", "눈꺼풀근"],
            benefits=["눈 피로 해소", "안구건조 완화", "시력 보호"],
            precautions=["심한 시력 장애 시 안과 상담", "현기증 시 중단"]
        )
        
        # 더 많은 운동들... (간단화를 위해 일부만 구현)
        
        return exercises

    async def analyze_posture_and_circulation(self, ui_state: UIState) -> Tuple[PostureAnalysis, CirculationMetrics]:
        """자세 및 혈액순환 분석"""
        
        # 자세 분석
        posture_analysis = await self.posture_analyzer.analyze_posture(ui_state.posture)
        self.posture_history.append(posture_analysis)
        
        # 혈액순환 분석
        circulation_metrics = await self.circulation_monitor.analyze_circulation(
            ui_state, self.current_sitting_time
        )
        self.circulation_history.append(circulation_metrics)
        
        # 앉은 시간 업데이트
        self.current_sitting_time += 1.0/60.0  # 1분 증가 (1Hz 호출 가정)
        
        return posture_analysis, circulation_metrics

    async def check_exercise_need(
        self, 
        posture_analysis: PostureAnalysis, 
        circulation_metrics: CirculationMetrics
    ) -> Optional[str]:
        """운동 필요성 체크"""
        
        current_time = time.time()
        
        # 너무 자주 알림 방지 (30분 간격)
        if current_time - self.last_exercise_reminder < 1800:
            return None
        
        exercise_needed = False
        exercise_reason = ""
        
        # 자세 문제 체크
        if posture_analysis.overall_score < 0.6:
            exercise_needed = True
            exercise_reason = f"자세 점수가 {posture_analysis.overall_score:.1f}로 낮습니다"
        
        # 장시간 앉기 체크
        if circulation_metrics.sitting_duration > 60:  # 1시간 이상
            exercise_needed = True
            exercise_reason = f"{circulation_metrics.sitting_duration:.0f}분간 앉아 계십니다"
        
        # 혈액순환 문제 체크
        if circulation_metrics.swelling_risk > 0.7:
            exercise_needed = True
            exercise_reason = "혈액순환이 좋지 않습니다"
        
        # 특정 자세 문제 체크
        if PostureIssue.FORWARD_HEAD in posture_analysis.identified_issues:
            severity = posture_analysis.severity_scores.get(PostureIssue.FORWARD_HEAD, 0)
            if severity > 0.7:
                exercise_needed = True
                exercise_reason = "거북목 증상이 심각합니다"
        
        if exercise_needed:
            self.last_exercise_reminder = current_time
            return exercise_reason
        
        return None

    async def recommend_exercise_routine(
        self, 
        posture_analysis: PostureAnalysis, 
        circulation_metrics: CirculationMetrics,
        available_time: float = 300.0  # 기본 5분
    ) -> List[ExerciseRoutine]:
        """맞춤형 운동 루틴 추천"""
        
        return await self.exercise_recommender.generate_personalized_routine(
            posture_analysis=posture_analysis,
            circulation_metrics=circulation_metrics,
            available_time=available_time,
            user_goals=self.physical_profile.primary_goals,
            fitness_level=self.physical_profile.fitness_level,
            limitations=self.physical_profile.physical_limitations
        )

    async def start_exercise_session(self, exercises: List[ExerciseRoutine]) -> str:
        """운동 세션 시작"""
        
        if self.current_session:
            await self.end_exercise_session()
        
        session_id = f"exercise_{int(time.time())}"
        total_duration = sum(ex.duration + ex.rest_intervals for ex in exercises)
        
        self.current_session = ExerciseSession(
            session_id=session_id,
            start_time=time.time(),
            exercises=exercises,
            total_duration=total_duration
        )
        
        print(f"💪 운동 세션 시작: {len(exercises)}개 운동, 예상 시간: {total_duration/60:.1f}분")
        
        # 첫 번째 운동 안내
        if exercises:
            await self._announce_exercise_start(exercises[0])
        
        return session_id

    async def _announce_exercise_start(self, exercise: ExerciseRoutine):
        """운동 시작 안내"""
        
        print(f"🏃 운동 시작: {exercise.exercise_type.value}")
        print(f"   목표: {', '.join(exercise.benefits)}")
        print(f"   지시사항:")
        for i, instruction in enumerate(exercise.instructions, 1):
            print(f"     {i}. {instruction}")
        
        if exercise.precautions:
            print(f"   주의사항: {', '.join(exercise.precautions)}")

    async def monitor_exercise_progress(self, ui_state: UIState) -> Dict[str, Any]:
        """운동 진행 상황 모니터링"""
        
        if not self.current_session:
            return {"session_active": False}
        
        elapsed_time = time.time() - self.current_session.start_time
        progress_percentage = min((elapsed_time / self.current_session.total_duration) * 100, 100.0)
        
        # 운동 중 자세 모니터링
        form_quality = await self._assess_exercise_form(ui_state)
        
        # 세션 완료 체크
        if elapsed_time >= self.current_session.total_duration:
            await self.end_exercise_session()
            return {"session_completed": True}
        
        return {
            "session_active": True,
            "progress_percentage": progress_percentage,
            "elapsed_time": elapsed_time,
            "remaining_time": self.current_session.total_duration - elapsed_time,
            "form_quality": form_quality,
            "current_exercise": self.current_session.exercises[0].exercise_type.value if self.current_session.exercises else None
        }

    async def _assess_exercise_form(self, ui_state: UIState) -> Dict[str, Any]:
        """운동 자세 평가"""
        
        # 간단한 자세 평가 (실제로는 더 정교한 분석 필요)
        form_quality = {
            "overall_score": ui_state.posture.spinal_alignment_score,
            "stability": 1.0 - (ui_state.hands.tremor_frequency or 0.0),
            "alignment": ui_state.posture.spinal_alignment_score,
            "feedback": []
        }
        
        if form_quality["overall_score"] < 0.6:
            form_quality["feedback"].append("자세를 더 바르게 유지하세요")
        
        if form_quality["stability"] < 0.7:
            form_quality["feedback"].append("동작을 더 천천히, 안정적으로 하세요")
        
        return form_quality

    async def end_exercise_session(self, user_feedback: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """운동 세션 종료"""
        
        if not self.current_session:
            return {"error": "활성 세션이 없습니다"}
        
        self.current_session.end_time = time.time()
        actual_duration = self.current_session.end_time - self.current_session.start_time
        
        # 완료율 계산
        completion_rate = min(1.0, actual_duration / self.current_session.total_duration)
        self.current_session.completion_rate = completion_rate
        
        # 사용자 피드백 기록
        if user_feedback:
            self.current_session.effectiveness_rating = user_feedback.get('effectiveness', 3)
            self.current_session.discomfort_level = user_feedback.get('discomfort', 1)
        
        # 프로필 업데이트
        await self._update_physical_profile()
        
        # 세션 저장
        await self._save_exercise_session()
        
        # 운동 효과 분석
        benefits = await self._analyze_exercise_benefits()
        
        print(f"💪 운동 세션 완료!")
        print(f"   지속 시간: {actual_duration/60:.1f}분")
        print(f"   완료율: {completion_rate*100:.1f}%")
        
        result = {
            "session_completed": True,
            "duration_minutes": actual_duration / 60.0,
            "completion_rate": completion_rate,
            "estimated_benefits": benefits
        }
        
        # 세션 정리
        self.current_session = None
        self.current_sitting_time = 0.0  # 앉은 시간 리셋
        
        return result

    async def _update_physical_profile(self):
        """신체 프로필 업데이트"""
        
        if not self.current_session:
            return
        
        # 운동 시간 누적
        session_minutes = (self.current_session.end_time - self.current_session.start_time) / 60.0
        self.physical_profile.total_exercise_time += session_minutes * self.current_session.completion_rate
        
        # 연속 운동 일수 계산
        await self._update_consecutive_exercise_days()
        
        # 체력 점수 업데이트
        await self._update_fitness_score()

    async def _update_consecutive_exercise_days(self):
        """연속 운동 일수 업데이트"""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        if self.exercise_history:
            last_session_date = datetime.datetime.fromtimestamp(
                self.exercise_history[-1].start_time
            ).strftime("%Y-%m-%d")
            
            if last_session_date == today:
                # 오늘 이미 운동했으면 연속 일수 유지
                pass
            else:
                # 어제 운동했으면 연속 일수 증가, 아니면 리셋
                yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                
                if last_session_date == yesterday:
                    self.physical_profile.consecutive_exercise_days += 1
                else:
                    self.physical_profile.consecutive_exercise_days = 1
        else:
            self.physical_profile.consecutive_exercise_days = 1

    async def _update_fitness_score(self):
        """체력 점수 업데이트"""
        
        # 체력 점수 계산 요소들
        total_hours = self.physical_profile.total_exercise_time / 60.0
        consecutive_days = self.physical_profile.consecutive_exercise_days
        recent_sessions = len([s for s in self.exercise_history[-7:] if s.completion_rate > 0.7])
        
        # 점수 계산 (0-1)
        new_score = min(1.0, max(0.1, 
            (total_hours * 0.01) +      # 총 운동 시간 기여
            (consecutive_days * 0.02) +  # 꾸준함 기여
            (recent_sessions * 0.05)     # 최근 활동성 기여
        ))
        
        # 점진적 업데이트
        self.physical_profile.fitness_score = (
            self.physical_profile.fitness_score * 0.8 + new_score * 0.2
        )

    async def _analyze_exercise_benefits(self) -> Dict[str, str]:
        """운동 효과 분석"""
        
        if not self.current_session:
            return {}
        
        benefits = {}
        
        # 자세 개선 예상
        posture_exercises = [ex for ex in self.current_session.exercises 
                           if ex.exercise_type in [ExerciseType.NECK_STRETCH, ExerciseType.BACK_ARCH]]
        if posture_exercises:
            benefits["posture"] = "자세가 10-15% 개선될 것으로 예상됩니다"
        
        # 혈액순환 개선
        circulation_exercises = [ex for ex in self.current_session.exercises 
                               if ex.exercise_type in [ExerciseType.ANKLE_PUMP, ExerciseType.CALF_RAISE]]
        if circulation_exercises:
            benefits["circulation"] = "하체 혈액순환이 20-30% 개선될 것으로 예상됩니다"
        
        # 근육 긴장 완화
        stretch_exercises = [ex for ex in self.current_session.exercises 
                           if "stretch" in ex.exercise_type.value]
        if stretch_exercises:
            benefits["tension_relief"] = "근육 긴장이 25-40% 완화될 것으로 예상됩니다"
        
        # 피로 감소
        if self.current_session.completion_rate > 0.7:
            benefits["fatigue_reduction"] = "운전 피로가 15-25% 감소할 것으로 예상됩니다"
        
        return benefits

    async def _save_exercise_session(self):
        """운동 세션 저장"""
        
        # 히스토리에 추가
        self.exercise_history.append(self.current_session)
        
        # 최근 50개만 유지
        if len(self.exercise_history) > 50:
            self.exercise_history = self.exercise_history[-50:]
        
        # 파일 저장
        history_path = Path(f"logs/exercise_history_{self.user_id}.json")
        history_path.parent.mkdir(exist_ok=True)
        
        history_data = [
            {
                'session_id': session.session_id,
                'start_time': session.start_time,
                'total_duration': session.total_duration,
                'completion_rate': session.completion_rate,
                'effectiveness_rating': session.effectiveness_rating,
                'discomfort_level': session.discomfort_level,
                'end_time': session.end_time
            } for session in self.exercise_history
        ]
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
        
        # 프로필 저장
        await self._save_physical_profile()

    async def _save_physical_profile(self):
        """신체 프로필 저장"""
        profile_path = Path(f"profiles/physical_profile_{self.user_id}.json")
        profile_path.parent.mkdir(exist_ok=True)
        
        data = {
            'user_id': self.physical_profile.user_id,
            'age': self.physical_profile.age,
            'height_cm': self.physical_profile.height_cm,
            'weight_kg': self.physical_profile.weight_kg,
            'fitness_level': self.physical_profile.fitness_level,
            'primary_goals': [goal.value for goal in self.physical_profile.primary_goals],
            'physical_limitations': self.physical_profile.physical_limitations,
            'injury_history': self.physical_profile.injury_history,
            'average_driving_hours_per_day': self.physical_profile.average_driving_hours_per_day,
            'longest_continuous_drive': self.physical_profile.longest_continuous_drive,
            'total_exercise_time': self.physical_profile.total_exercise_time,
            'consecutive_exercise_days': self.physical_profile.consecutive_exercise_days,
            'fitness_score': self.physical_profile.fitness_score,
            'last_updated': time.time()
        }
        
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_fitness_statistics(self) -> Dict[str, Any]:
        """체력 통계 정보 반환"""
        
        recent_sessions = self.exercise_history[-7:] if len(self.exercise_history) >= 7 else self.exercise_history
        
        avg_completion_rate = np.mean([s.completion_rate for s in recent_sessions]) if recent_sessions else 0.0
        total_sessions = len(self.exercise_history)
        
        return {
            "fitness_level": self.physical_profile.fitness_level,
            "fitness_score": self.physical_profile.fitness_score,
            "total_exercise_time_hours": self.physical_profile.total_exercise_time / 60.0,
            "consecutive_exercise_days": self.physical_profile.consecutive_exercise_days,
            "total_sessions": total_sessions,
            "average_completion_rate": avg_completion_rate,
            "primary_goals": [goal.value for goal in self.physical_profile.primary_goals],
            "current_sitting_time_minutes": self.current_sitting_time,
            "movement_alerts_today": self.movement_alerts_today
        }


class PostureAnalyzer:
    """자세 분석기"""
    
    async def analyze_posture(self, posture_data: PostureData) -> PostureAnalysis:
        """자세 분석"""
        
        current_time = time.time()
        identified_issues = []
        severity_scores = {}
        
        # 전체 자세 점수
        overall_score = posture_data.spinal_alignment_score
        
        # 구체적인 문제 식별
        if overall_score < 0.7:
            if hasattr(posture_data, 'head_forward_angle') and posture_data.head_forward_angle > 15:
                identified_issues.append(PostureIssue.FORWARD_HEAD)
                severity_scores[PostureIssue.FORWARD_HEAD] = min(1.0, posture_data.head_forward_angle / 30.0)
            
            if hasattr(posture_data, 'shoulder_elevation') and posture_data.shoulder_elevation > 10:
                identified_issues.append(PostureIssue.ROUNDED_SHOULDERS)
                severity_scores[PostureIssue.ROUNDED_SHOULDERS] = min(1.0, posture_data.shoulder_elevation / 20.0)
            
            if overall_score < 0.5:
                identified_issues.append(PostureIssue.SLOUCHED_BACK)
                severity_scores[PostureIssue.SLOUCHED_BACK] = 1.0 - overall_score
        
        # 개선 제안
        improvement_suggestions = []
        if PostureIssue.FORWARD_HEAD in identified_issues:
            improvement_suggestions.append("목을 뒤로 당기고 턱을 아래로 당기세요")
        if PostureIssue.ROUNDED_SHOULDERS in identified_issues:
            improvement_suggestions.append("어깨를 뒤로 펴고 가슴을 열어주세요")
        if PostureIssue.SLOUCHED_BACK in identified_issues:
            improvement_suggestions.append("등을 곧게 펴고 허리를 세워주세요")
        
        # 위험도 평가
        if overall_score >= 0.7:
            risk_assessment = "low"
        elif overall_score >= 0.5:
            risk_assessment = "medium"
        else:
            risk_assessment = "high"
        
        return PostureAnalysis(
            timestamp=current_time,
            overall_score=overall_score,
            identified_issues=identified_issues,
            severity_scores=severity_scores,
            improvement_suggestions=improvement_suggestions,
            risk_assessment=risk_assessment
        )


class CirculationMonitor:
    """혈액순환 모니터"""
    
    async def analyze_circulation(self, ui_state: UIState, sitting_duration: float) -> CirculationMetrics:
        """혈액순환 분석"""
        
        current_time = time.time()
        
        # 움직임 빈도 (손 떨림, 자세 변화 등으로 추정)
        movement_frequency = 1.0 - min(1.0, sitting_duration / 120.0)  # 2시간 기준
        
        # 하체 경직도 (앉은 시간 기반)
        lower_body_stiffness = min(1.0, sitting_duration / 180.0)  # 3시간 기준
        
        # 부종 위험도
        swelling_risk = min(1.0, sitting_duration / 240.0)  # 4시간 기준
        
        return CirculationMetrics(
            timestamp=current_time,
            sitting_duration=sitting_duration,
            movement_frequency=movement_frequency,
            lower_body_stiffness=lower_body_stiffness,
            swelling_risk=swelling_risk
        )


class ExerciseRecommendationEngine:
    """운동 추천 엔진"""
    
    def __init__(self, physical_profile: PhysicalProfile):
        self.profile = physical_profile

    async def generate_personalized_routine(
        self,
        posture_analysis: PostureAnalysis,
        circulation_metrics: CirculationMetrics,
        available_time: float,
        user_goals: List[FitnessGoal],
        fitness_level: str,
        limitations: List[str]
    ) -> List[ExerciseRoutine]:
        """개인화된 운동 루틴 생성"""
        
        recommended_exercises = []
        
        # 자세 문제 기반 운동 선택
        if PostureIssue.FORWARD_HEAD in posture_analysis.identified_issues:
            recommended_exercises.append(ExerciseType.NECK_STRETCH)
        
        if PostureIssue.ROUNDED_SHOULDERS in posture_analysis.identified_issues:
            recommended_exercises.append(ExerciseType.SHOULDER_ROLL)
        
        if PostureIssue.SLOUCHED_BACK in posture_analysis.identified_issues:
            recommended_exercises.append(ExerciseType.BACK_ARCH)
        
        # 혈액순환 문제 기반 운동 선택
        if circulation_metrics.swelling_risk > 0.5:
            recommended_exercises.append(ExerciseType.ANKLE_PUMP)
        
        if circulation_metrics.sitting_duration > 60:
            recommended_exercises.append(ExerciseType.CALF_RAISE)
        
        # 목표 기반 운동 추가
        if FitnessGoal.CIRCULATION in user_goals:
            if ExerciseType.ANKLE_PUMP not in recommended_exercises:
                recommended_exercises.append(ExerciseType.ANKLE_PUMP)
        
        if FitnessGoal.POSTURE_CORRECTION in user_goals:
            if ExerciseType.BACK_ARCH not in recommended_exercises:
                recommended_exercises.append(ExerciseType.BACK_ARCH)
        
        # 눈 피로가 있을 때 눈 운동 추가
        recommended_exercises.append(ExerciseType.EYE_EXERCISE)
        
        # 시간 제약에 맞춰 운동 선택 및 조정
        final_routine = []
        total_time = 0.0
        
        # 기본 운동 데이터베이스에서 루틴 가져오기 (간단화)
        exercise_db = {
            ExerciseType.NECK_STRETCH: 60.0,
            ExerciseType.SHOULDER_ROLL: 45.0,
            ExerciseType.BACK_ARCH: 30.0,
            ExerciseType.ANKLE_PUMP: 60.0,
            ExerciseType.EYE_EXERCISE: 90.0
        }
        
        for exercise_type in recommended_exercises:
            exercise_time = exercise_db.get(exercise_type, 60.0)
            if total_time + exercise_time <= available_time:
                # 실제 운동 루틴 객체 생성 (간단화)
                routine = ExerciseRoutine(
                    routine_id=f"{exercise_type.value}_personalized",
                    exercise_type=exercise_type,
                    intensity=ExerciseIntensity.GENTLE if fitness_level == "beginner" else ExerciseIntensity.MODERATE,
                    duration=exercise_time,
                    repetitions=5,
                    rest_intervals=10.0,
                    instructions=[f"{exercise_type.value} 운동을 실시하세요"],
                    target_muscles=["해당 근육군"],
                    benefits=["해당 효과"],
                    precautions=["주의사항"]
                )
                final_routine.append(routine)
                total_time += exercise_time
        
        return final_routine