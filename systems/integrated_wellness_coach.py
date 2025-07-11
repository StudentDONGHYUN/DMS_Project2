"""
S-Class DMS v19.0 - 통합 웰니스 코치
모든 웰니스 코칭 시스템을 통합하여 관리하는 종합 솔루션
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from config.settings import get_config
from models.data_structures import UIState

# 개별 코칭 시스템들 import
from systems.mindfulness_coach import MindfulnessCoach, MeditationType
from systems.sleep_optimization_coach import SleepOptimizationCoach, FatigueLevel
from systems.fitness_optimization_coach import FitnessOptimizationCoach, PostureIssue
from systems.ai_driving_coach import AIDrivingCoach, CoachingFeedback
from systems.emotional_care_system import EmotionalCareSystem, CareMode
from systems.v2d_healthcare import V2DHealthcareSystem


class WellnessArea(Enum):
    """웰니스 영역"""
    MINDFULNESS = "mindfulness"           # 마음챙김
    SLEEP = "sleep"                      # 수면
    FITNESS = "fitness"                  # 체력
    DRIVING_SKILLS = "driving_skills"    # 운전 기술
    EMOTIONAL_CARE = "emotional_care"    # 감정 케어
    HEALTH_MONITORING = "health_monitoring"  # 건강 모니터링


class CoachingPriority(Enum):
    """코칭 우선순위"""
    EMERGENCY = "emergency"              # 응급 (즉시 대응 필요)
    HIGH = "high"                       # 높음 (빠른 대응 필요)
    MEDIUM = "medium"                   # 보통 (적절한 시점에 대응)
    LOW = "low"                         # 낮음 (시간이 있을 때 대응)
    BACKGROUND = "background"           # 백그라운드 (자동 실행)


@dataclass
class IntegratedRecommendation:
    """통합 추천사항"""
    area: WellnessArea
    priority: CoachingPriority
    title: str
    message: str
    action_type: str  # "meditation", "exercise", "rest", "coaching", etc.
    estimated_duration: float  # 예상 소요 시간 (분)
    expected_benefit: str
    urgency_score: float  # 0-1 긴급도 점수
    timestamp: float = field(default_factory=time.time)


@dataclass
class WellnessProfile:
    """통합 웰니스 프로필"""
    user_id: str
    
    # 전체 웰니스 점수 (0-1)
    overall_wellness_score: float = 0.5
    
    # 영역별 점수
    mindfulness_score: float = 0.5
    sleep_score: float = 0.5
    fitness_score: float = 0.5
    driving_score: float = 0.5
    emotional_score: float = 0.5
    health_score: float = 0.5
    
    # 개인 우선순위 (사용자가 설정)
    priority_areas: List[WellnessArea] = field(default_factory=list)
    
    # 활성화된 코칭 시스템
    active_coaches: List[WellnessArea] = field(default_factory=list)
    
    # 코칭 성과
    total_coaching_hours: float = 0.0
    improvement_rate: float = 0.0  # 개선율
    
    last_updated: float = field(default_factory=time.time)


class IntegratedWellnessCoach:
    """통합 웰니스 코치 메인 시스템"""
    
    def __init__(self, user_id: str = "default"):
        self.config = get_config()
        self.user_id = user_id
        
        # 통합 프로필
        self.wellness_profile = self._load_wellness_profile()
        
        # 개별 코칭 시스템들 초기화
        self.coaches = self._initialize_coaches()
        
        # 실시간 상태 추적
        self.current_recommendations = []
        self.active_sessions = {}  # 현재 진행 중인 세션들
        
        # 우선순위 관리
        self.priority_manager = WellnessPriorityManager()
        
        # 일정 관리
        self.schedule_manager = WellnessScheduleManager()
        
        print(f"🌟 통합 웰니스 코치 초기화 완료 - 사용자: {user_id}")
        print(f"   전체 웰니스 점수: {self.wellness_profile.overall_wellness_score:.2f}")
        print(f"   활성 코치: {[area.value for area in self.wellness_profile.active_coaches]}")

    def _load_wellness_profile(self) -> WellnessProfile:
        """통합 웰니스 프로필 로드"""
        profile_path = Path(f"profiles/wellness_profile_{self.user_id}.json")
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return WellnessProfile(
                        user_id=data.get('user_id', self.user_id),
                        overall_wellness_score=data.get('overall_wellness_score', 0.5),
                        mindfulness_score=data.get('mindfulness_score', 0.5),
                        sleep_score=data.get('sleep_score', 0.5),
                        fitness_score=data.get('fitness_score', 0.5),
                        driving_score=data.get('driving_score', 0.5),
                        emotional_score=data.get('emotional_score', 0.5),
                        health_score=data.get('health_score', 0.5),
                        priority_areas=[WellnessArea(area) for area in data.get('priority_areas', [])],
                        active_coaches=[WellnessArea(area) for area in data.get('active_coaches', [])],
                        total_coaching_hours=data.get('total_coaching_hours', 0.0),
                        improvement_rate=data.get('improvement_rate', 0.0),
                        last_updated=data.get('last_updated', time.time())
                    )
            except Exception as e:
                print(f"웰니스 프로필 로드 실패: {e}")
        
        # 기본 프로필 생성
        return WellnessProfile(
            user_id=self.user_id,
            priority_areas=[WellnessArea.MINDFULNESS, WellnessArea.FITNESS, WellnessArea.SLEEP],
            active_coaches=[WellnessArea.MINDFULNESS, WellnessArea.FITNESS, WellnessArea.SLEEP, 
                          WellnessArea.DRIVING_SKILLS, WellnessArea.EMOTIONAL_CARE, WellnessArea.HEALTH_MONITORING]
        )

    def _initialize_coaches(self) -> Dict[WellnessArea, Any]:
        """개별 코칭 시스템들 초기화"""
        coaches = {}
        
        # 각 영역별 코치 초기화
        if WellnessArea.MINDFULNESS in self.wellness_profile.active_coaches:
            coaches[WellnessArea.MINDFULNESS] = MindfulnessCoach(self.user_id)
        
        if WellnessArea.SLEEP in self.wellness_profile.active_coaches:
            coaches[WellnessArea.SLEEP] = SleepOptimizationCoach(self.user_id)
        
        if WellnessArea.FITNESS in self.wellness_profile.active_coaches:
            coaches[WellnessArea.FITNESS] = FitnessOptimizationCoach(self.user_id)
        
        if WellnessArea.DRIVING_SKILLS in self.wellness_profile.active_coaches:
            coaches[WellnessArea.DRIVING_SKILLS] = AIDrivingCoach(self.user_id)
        
        if WellnessArea.EMOTIONAL_CARE in self.wellness_profile.active_coaches:
            coaches[WellnessArea.EMOTIONAL_CARE] = EmotionalCareSystem(self.user_id)
        
        if WellnessArea.HEALTH_MONITORING in self.wellness_profile.active_coaches:
            coaches[WellnessArea.HEALTH_MONITORING] = V2DHealthcareSystem(self.user_id)
        
        return coaches

    async def analyze_wellness_state(self, ui_state: UIState) -> List[IntegratedRecommendation]:
        """종합 웰니스 상태 분석 및 추천"""
        
        recommendations = []
        
        # 각 코칭 시스템별 분석 병렬 실행
        analysis_tasks = []
        
        # 마음챙김 분석
        if WellnessArea.MINDFULNESS in self.coaches:
            analysis_tasks.append(self._analyze_mindfulness(ui_state))
        
        # 수면 분석
        if WellnessArea.SLEEP in self.coaches:
            analysis_tasks.append(self._analyze_sleep(ui_state))
        
        # 체력 분석
        if WellnessArea.FITNESS in self.coaches:
            analysis_tasks.append(self._analyze_fitness(ui_state))
        
        # 운전 기술 분석
        if WellnessArea.DRIVING_SKILLS in self.coaches:
            analysis_tasks.append(self._analyze_driving_skills(ui_state))
        
        # 감정 케어 분석
        if WellnessArea.EMOTIONAL_CARE in self.coaches:
            analysis_tasks.append(self._analyze_emotional_care(ui_state))
        
        # 건강 모니터링 분석
        if WellnessArea.HEALTH_MONITORING in self.coaches:
            analysis_tasks.append(self._analyze_health_monitoring(ui_state))
        
        # 모든 분석 병렬 실행
        if analysis_tasks:
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # 결과 수집
            for result in analysis_results:
                if isinstance(result, list):
                    recommendations.extend(result)
        
        # 우선순위에 따라 정렬
        recommendations = await self.priority_manager.prioritize_recommendations(
            recommendations, self.wellness_profile
        )
        
        # 현재 추천사항 업데이트
        self.current_recommendations = recommendations
        
        return recommendations

    async def _analyze_mindfulness(self, ui_state: UIState) -> List[IntegratedRecommendation]:
        """마음챙김 분석"""
        recommendations = []
        
        coach = self.coaches[WellnessArea.MINDFULNESS]
        
        # 명상 필요성 분석
        meditation_type = await coach.analyze_mindfulness_need(ui_state)
        
        if meditation_type:
            urgency_score = 0.7 if ui_state.biometrics.stress_level and ui_state.biometrics.stress_level > 0.7 else 0.5
            
            # 명상 유형별 추천 생성
            meditation_names = {
                MeditationType.STRESS_RELIEF: "스트레스 해소 명상",
                MeditationType.BREATH_AWARENESS: "호흡 집중 명상", 
                MeditationType.FOCUS_ENHANCEMENT: "집중력 향상 명상",
                MeditationType.MICRO_MEDITATION: "마이크로 명상",
                MeditationType.BODY_SCAN: "바디 스캔 명상"
            }
            
            duration_map = {
                MeditationType.MICRO_MEDITATION: 1.0,
                MeditationType.BREATH_AWARENESS: 3.0,
                MeditationType.STRESS_RELIEF: 5.0,
                MeditationType.FOCUS_ENHANCEMENT: 5.0,
                MeditationType.BODY_SCAN: 8.0
            }
            
            recommendations.append(IntegratedRecommendation(
                area=WellnessArea.MINDFULNESS,
                priority=CoachingPriority.HIGH if urgency_score > 0.6 else CoachingPriority.MEDIUM,
                title="마음챙김 명상 추천",
                message=f"{meditation_names.get(meditation_type, '명상')}으로 마음을 진정시켜보세요",
                action_type="meditation",
                estimated_duration=duration_map.get(meditation_type, 5.0),
                expected_benefit="스트레스 감소, 집중력 향상, 정서적 안정",
                urgency_score=urgency_score
            ))
        
        return recommendations

    async def _analyze_sleep(self, ui_state: UIState) -> List[IntegratedRecommendation]:
        """수면 분석"""
        recommendations = []
        
        coach = self.coaches[WellnessArea.SLEEP]
        
        # 현재 피로도 분석
        fatigue_level = await coach.analyze_current_fatigue(ui_state)
        
        # 파워 낮잠 제안
        if fatigue_level in [FatigueLevel.HIGH, FatigueLevel.CRITICAL]:
            nap_suggestion = await coach.suggest_power_nap()
            
            if nap_suggestion:
                urgency_score = 0.9 if fatigue_level == FatigueLevel.CRITICAL else 0.7
                
                recommendations.append(IntegratedRecommendation(
                    area=WellnessArea.SLEEP,
                    priority=CoachingPriority.HIGH if fatigue_level == FatigueLevel.CRITICAL else CoachingPriority.MEDIUM,
                    title="파워 낮잠 추천",
                    message=f"{nap_suggestion.optimal_duration:.0f}분간의 낮잠으로 피로를 회복하세요",
                    action_type="power_nap",
                    estimated_duration=nap_suggestion.optimal_duration,
                    expected_benefit=f"피로도 {nap_suggestion.fatigue_reduction_expected*100:.0f}% 감소",
                    urgency_score=urgency_score
                ))
        
        # 수면 추천사항
        sleep_recommendations = await coach.generate_sleep_recommendations()
        
        for sleep_rec in sleep_recommendations:
            priority = CoachingPriority.HIGH if sleep_rec.priority <= 2 else CoachingPriority.MEDIUM
            
            recommendations.append(IntegratedRecommendation(
                area=WellnessArea.SLEEP,
                priority=priority,
                title=f"수면 최적화 - {sleep_rec.type}",
                message=sleep_rec.message,
                action_type=sleep_rec.type,
                estimated_duration=sleep_rec.duration or 5.0,
                expected_benefit=sleep_rec.expected_benefit,
                urgency_score=0.8 if sleep_rec.priority == 1 else 0.5
            ))
        
        return recommendations

    async def _analyze_fitness(self, ui_state: UIState) -> List[IntegratedRecommendation]:
        """체력 분석"""
        recommendations = []
        
        coach = self.coaches[WellnessArea.FITNESS]
        
        # 자세 및 혈액순환 분석
        posture_analysis, circulation_metrics = await coach.analyze_posture_and_circulation(ui_state)
        
        # 운동 필요성 체크
        exercise_reason = await coach.check_exercise_need(posture_analysis, circulation_metrics)
        
        if exercise_reason:
            # 맞춤형 운동 루틴 추천
            exercise_routines = await coach.recommend_exercise_routine(
                posture_analysis, circulation_metrics, available_time=300.0
            )
            
            if exercise_routines:
                total_duration = sum(ex.duration for ex in exercise_routines) / 60.0  # 분 단위
                
                urgency_score = 0.8 if posture_analysis.risk_assessment == "high" else 0.6
                
                recommendations.append(IntegratedRecommendation(
                    area=WellnessArea.FITNESS,
                    priority=CoachingPriority.HIGH if urgency_score > 0.7 else CoachingPriority.MEDIUM,
                    title="체력 운동 추천",
                    message=f"{exercise_reason}. {len(exercise_routines)}가지 운동으로 개선해보세요",
                    action_type="exercise_routine",
                    estimated_duration=total_duration,
                    expected_benefit="자세 개선, 혈액순환 증진, 근육 긴장 완화",
                    urgency_score=urgency_score
                ))
        
        return recommendations

    async def _analyze_driving_skills(self, ui_state: UIState) -> List[IntegratedRecommendation]:
        """운전 기술 분석"""
        recommendations = []
        
        coach = self.coaches[WellnessArea.DRIVING_SKILLS]
        
        # 실시간 운전 분석
        feedback_list = await coach.process_real_time_data(ui_state)
        
        for feedback in feedback_list:
            if feedback.priority <= 2:  # 높은 우선순위만
                priority = CoachingPriority.HIGH if feedback.priority == 1 else CoachingPriority.MEDIUM
                
                recommendations.append(IntegratedRecommendation(
                    area=WellnessArea.DRIVING_SKILLS,
                    priority=priority,
                    title=f"운전 기술 코칭 - {feedback.category.value}",
                    message=feedback.message,
                    action_type="driving_coaching",
                    estimated_duration=1.0,  # 실시간 피드백
                    expected_benefit=feedback.suggestion,
                    urgency_score=1.0 - (feedback.priority / 5.0)
                ))
        
        return recommendations

    async def _analyze_emotional_care(self, ui_state: UIState) -> List[IntegratedRecommendation]:
        """감정 케어 분석"""
        recommendations = []
        
        coach = self.coaches[WellnessArea.EMOTIONAL_CARE]
        
        # 감정 상태 분석 및 케어 실행
        care_session = await coach.process_emotion_data(ui_state)
        
        if care_session:
            care_mode_names = {
                CareMode.STRESS_RELIEF: "스트레스 해소",
                CareMode.RELAXATION: "이완",
                CareMode.ENERGIZING: "활성화",
                CareMode.COMFORT: "위로",
                CareMode.MOOD_BOOST: "기분 향상",
                CareMode.FOCUS: "집중"
            }
            
            recommendations.append(IntegratedRecommendation(
                area=WellnessArea.EMOTIONAL_CARE,
                priority=CoachingPriority.MEDIUM,
                title=f"감정 케어 - {care_mode_names.get(care_session.care_mode, '케어')}",
                message="현재 감정 상태에 맞는 멀티모달 케어를 시작합니다",
                action_type="emotional_care",
                estimated_duration=5.0,
                expected_benefit="정서적 안정, 스트레스 완화, 기분 개선",
                urgency_score=0.6
            ))
        
        return recommendations

    async def _analyze_health_monitoring(self, ui_state: UIState) -> List[IntegratedRecommendation]:
        """건강 모니터링 분석"""
        recommendations = []
        
        coach = self.coaches[WellnessArea.HEALTH_MONITORING]
        
        # 생체 데이터 처리
        health_alerts = await coach.process_biometric_data(ui_state)
        
        for alert in health_alerts:
            if alert.severity.value in ['high', 'critical', 'emergency']:
                priority_map = {
                    'emergency': CoachingPriority.EMERGENCY,
                    'critical': CoachingPriority.HIGH,
                    'high': CoachingPriority.HIGH
                }
                
                recommendations.append(IntegratedRecommendation(
                    area=WellnessArea.HEALTH_MONITORING,
                    priority=priority_map[alert.severity.value],
                    title=f"건강 경고 - {alert.metric_type.value}",
                    message=alert.message,
                    action_type="health_alert",
                    estimated_duration=0.5,  # 즉시 확인
                    expected_benefit=alert.recommendation,
                    urgency_score=1.0 if alert.severity.value == 'emergency' else 0.8
                ))
        
        return recommendations

    async def execute_recommendation(self, recommendation: IntegratedRecommendation) -> Dict[str, Any]:
        """추천사항 실행"""
        
        result = {"success": False, "message": ""}
        
        try:
            if recommendation.area not in self.coaches:
                result["message"] = f"{recommendation.area.value} 코치가 활성화되지 않았습니다"
                return result
            
            coach = self.coaches[recommendation.area]
            
            # 액션 타입별 실행
            if recommendation.action_type == "meditation":
                if hasattr(coach, 'start_meditation_session'):
                    meditation_type = self._get_meditation_type_from_title(recommendation.title)
                    session_id = await coach.start_meditation_session(meditation_type)
                    self.active_sessions[recommendation.area] = session_id
                    result = {"success": True, "session_id": session_id}
            
            elif recommendation.action_type == "exercise_routine":
                if hasattr(coach, 'start_exercise_session'):
                    # 간단화: 기본 운동 루틴 실행
                    from systems.fitness_optimization_coach import ExerciseType, ExerciseRoutine, ExerciseIntensity
                    basic_routine = [
                        ExerciseRoutine(
                            routine_id="integrated_basic",
                            exercise_type=ExerciseType.NECK_STRETCH,
                            intensity=ExerciseIntensity.GENTLE,
                            duration=60.0,
                            repetitions=5,
                            rest_intervals=10.0,
                            instructions=["목 스트레칭을 실시하세요"],
                            target_muscles=["목 근육"],
                            benefits=["목 긴장 완화"],
                            precautions=["통증 시 중단"]
                        )
                    ]
                    session_id = await coach.start_exercise_session(basic_routine)
                    self.active_sessions[recommendation.area] = session_id
                    result = {"success": True, "session_id": session_id}
            
            elif recommendation.action_type == "power_nap":
                result = {
                    "success": True, 
                    "message": f"{recommendation.estimated_duration:.0f}분간 휴식을 취하세요",
                    "action": "start_nap_timer"
                }
            
            else:
                result = {
                    "success": True,
                    "message": f"{recommendation.title} 실행 완료",
                    "action": "background_execution"
                }
            
            print(f"🌟 추천사항 실행: {recommendation.title}")
            
        except Exception as e:
            result = {"success": False, "message": f"실행 중 오류: {e}"}
        
        return result

    def _get_meditation_type_from_title(self, title: str) -> MeditationType:
        """제목에서 명상 유형 추출"""
        if "스트레스" in title:
            return MeditationType.STRESS_RELIEF
        elif "호흡" in title:
            return MeditationType.BREATH_AWARENESS
        elif "집중" in title:
            return MeditationType.FOCUS_ENHANCEMENT
        elif "마이크로" in title:
            return MeditationType.MICRO_MEDITATION
        else:
            return MeditationType.BREATH_AWARENESS

    async def get_wellness_dashboard(self) -> Dict[str, Any]:
        """웰니스 대시보드 데이터 반환"""
        
        # 각 코치별 통계 수집
        coach_stats = {}
        
        for area, coach in self.coaches.items():
            try:
                if hasattr(coach, 'get_mindfulness_statistics'):
                    coach_stats[area.value] = coach.get_mindfulness_statistics()
                elif hasattr(coach, 'get_sleep_statistics'):
                    coach_stats[area.value] = coach.get_sleep_statistics()
                elif hasattr(coach, 'get_fitness_statistics'):
                    coach_stats[area.value] = coach.get_fitness_statistics()
                elif hasattr(coach, 'get_driving_statistics'):
                    coach_stats[area.value] = coach.get_driving_statistics()
                elif hasattr(coach, 'get_care_statistics'):
                    coach_stats[area.value] = coach.get_care_statistics()
                elif hasattr(coach, 'get_health_statistics'):
                    coach_stats[area.value] = coach.get_health_statistics()
            except Exception as e:
                print(f"통계 수집 실패 - {area.value}: {e}")
        
        # 전체 웰니스 점수 업데이트
        await self._update_wellness_scores()
        
        return {
            "overall_wellness_score": self.wellness_profile.overall_wellness_score,
            "area_scores": {
                "mindfulness": self.wellness_profile.mindfulness_score,
                "sleep": self.wellness_profile.sleep_score,
                "fitness": self.wellness_profile.fitness_score,
                "driving": self.wellness_profile.driving_score,
                "emotional": self.wellness_profile.emotional_score,
                "health": self.wellness_profile.health_score
            },
            "active_sessions": self.active_sessions,
            "current_recommendations": [
                {
                    "area": rec.area.value,
                    "priority": rec.priority.value,
                    "title": rec.title,
                    "message": rec.message,
                    "duration": rec.estimated_duration,
                    "urgency": rec.urgency_score
                } for rec in self.current_recommendations[:5]  # 상위 5개
            ],
            "coach_statistics": coach_stats,
            "total_coaching_hours": self.wellness_profile.total_coaching_hours,
            "improvement_rate": self.wellness_profile.improvement_rate
        }

    async def _update_wellness_scores(self):
        """웰니스 점수 업데이트"""
        
        # 각 영역별 점수 수집
        area_scores = {}
        
        for area, coach in self.coaches.items():
            try:
                if area == WellnessArea.MINDFULNESS and hasattr(coach, 'mindfulness_profile'):
                    area_scores[area] = coach.mindfulness_profile.mindfulness_level / 10.0
                elif area == WellnessArea.SLEEP and hasattr(coach, 'physical_profile'):
                    area_scores[area] = coach.circadian_profile.natural_sleep_duration / 10.0  # 간단화
                elif area == WellnessArea.FITNESS and hasattr(coach, 'physical_profile'):
                    area_scores[area] = coach.physical_profile.fitness_score
                elif area == WellnessArea.DRIVING_SKILLS and hasattr(coach, 'driving_profile'):
                    # 운전 점수는 최근 세션 기반으로 계산
                    recent_scores = []
                    if hasattr(coach.driving_profile, 'historical_scores'):
                        for skill_scores in coach.driving_profile.historical_scores.values():
                            if skill_scores:
                                recent_scores.extend(skill_scores[-3:])  # 최근 3개
                    area_scores[area] = sum(recent_scores) / len(recent_scores) / 100.0 if recent_scores else 0.5
                else:
                    area_scores[area] = 0.5  # 기본값
            except Exception:
                area_scores[area] = 0.5
        
        # 개별 영역 점수 업데이트
        self.wellness_profile.mindfulness_score = area_scores.get(WellnessArea.MINDFULNESS, 0.5)
        self.wellness_profile.sleep_score = area_scores.get(WellnessArea.SLEEP, 0.5)
        self.wellness_profile.fitness_score = area_scores.get(WellnessArea.FITNESS, 0.5)
        self.wellness_profile.driving_score = area_scores.get(WellnessArea.DRIVING_SKILLS, 0.5)
        self.wellness_profile.emotional_score = area_scores.get(WellnessArea.EMOTIONAL_CARE, 0.5)
        self.wellness_profile.health_score = area_scores.get(WellnessArea.HEALTH_MONITORING, 0.5)
        
        # 전체 웰니스 점수 계산 (가중 평균)
        weights = {
            WellnessArea.MINDFULNESS: 0.2,
            WellnessArea.SLEEP: 0.2,
            WellnessArea.FITNESS: 0.15,
            WellnessArea.DRIVING_SKILLS: 0.15,
            WellnessArea.EMOTIONAL_CARE: 0.15,
            WellnessArea.HEALTH_MONITORING: 0.15
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for area, score in area_scores.items():
            weight = weights.get(area, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        self.wellness_profile.overall_wellness_score = weighted_sum / total_weight if total_weight > 0 else 0.5

    async def _save_wellness_profile(self):
        """웰니스 프로필 저장"""
        profile_path = Path(f"profiles/wellness_profile_{self.user_id}.json")
        profile_path.parent.mkdir(exist_ok=True)
        
        data = {
            'user_id': self.wellness_profile.user_id,
            'overall_wellness_score': self.wellness_profile.overall_wellness_score,
            'mindfulness_score': self.wellness_profile.mindfulness_score,
            'sleep_score': self.wellness_profile.sleep_score,
            'fitness_score': self.wellness_profile.fitness_score,
            'driving_score': self.wellness_profile.driving_score,
            'emotional_score': self.wellness_profile.emotional_score,
            'health_score': self.wellness_profile.health_score,
            'priority_areas': [area.value for area in self.wellness_profile.priority_areas],
            'active_coaches': [area.value for area in self.wellness_profile.active_coaches],
            'total_coaching_hours': self.wellness_profile.total_coaching_hours,
            'improvement_rate': self.wellness_profile.improvement_rate,
            'last_updated': time.time()
        }
        
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class WellnessPriorityManager:
    """웰니스 우선순위 관리자"""
    
    async def prioritize_recommendations(
        self, 
        recommendations: List[IntegratedRecommendation], 
        wellness_profile: WellnessProfile
    ) -> List[IntegratedRecommendation]:
        """추천사항 우선순위 정렬"""
        
        # 우선순위 점수 계산
        for rec in recommendations:
            priority_score = self._calculate_priority_score(rec, wellness_profile)
            rec.urgency_score = priority_score
        
        # 우선순위에 따라 정렬 (높은 점수 먼저)
        recommendations.sort(key=lambda x: x.urgency_score, reverse=True)
        
        return recommendations

    def _calculate_priority_score(
        self, 
        recommendation: IntegratedRecommendation, 
        wellness_profile: WellnessProfile
    ) -> float:
        """우선순위 점수 계산"""
        
        base_score = recommendation.urgency_score
        
        # 사용자 우선 영역 가점
        if recommendation.area in wellness_profile.priority_areas:
            base_score += 0.2
        
        # 우선순위 레벨별 가중치
        priority_weights = {
            CoachingPriority.EMERGENCY: 1.0,
            CoachingPriority.HIGH: 0.8,
            CoachingPriority.MEDIUM: 0.6,
            CoachingPriority.LOW: 0.4,
            CoachingPriority.BACKGROUND: 0.2
        }
        
        weight = priority_weights.get(recommendation.priority, 0.5)
        final_score = base_score * weight
        
        return min(1.0, final_score)


class WellnessScheduleManager:
    """웰니스 일정 관리자"""
    
    def __init__(self):
        self.scheduled_sessions = {}
    
    async def schedule_wellness_session(
        self, 
        area: WellnessArea, 
        session_type: str, 
        preferred_time: Optional[float] = None
    ) -> str:
        """웰니스 세션 예약"""
        
        session_id = f"{area.value}_{session_type}_{int(time.time())}"
        
        # 간단한 스케줄링 로직
        scheduled_time = preferred_time or (time.time() + 3600)  # 1시간 후 기본값
        
        self.scheduled_sessions[session_id] = {
            "area": area,
            "session_type": session_type,
            "scheduled_time": scheduled_time,
            "status": "scheduled"
        }
        
        return session_id
    
    async def get_upcoming_sessions(self) -> List[Dict[str, Any]]:
        """예정된 세션 목록 반환"""
        
        current_time = time.time()
        upcoming = []
        
        for session_id, session_info in self.scheduled_sessions.items():
            if session_info["scheduled_time"] > current_time:
                upcoming.append({
                    "session_id": session_id,
                    "area": session_info["area"].value,
                    "type": session_info["session_type"],
                    "scheduled_time": session_info["scheduled_time"],
                    "time_until": session_info["scheduled_time"] - current_time
                })
        
        # 시간순 정렬
        upcoming.sort(key=lambda x: x["scheduled_time"])
        
        return upcoming