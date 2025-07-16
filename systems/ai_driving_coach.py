"""
S-Class DMS v19.0 - AI 드라이빙 코치 시스템
실시간 운전 습관 분석 및 개인화된 피드백 제공
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
from pathlib import Path

from config.settings import get_config
from models.data_structures import UIState


class DrivingSkillCategory(Enum):
    """운전 기술 카테고리"""

    STEERING_SMOOTHNESS = "steering_smoothness"  # 핸들 조작 부드러움
    POSTURE_STABILITY = "posture_stability"  # 자세 안정성
    ATTENTION_MANAGEMENT = "attention_management"  # 주의 관리
    SPEED_CONTROL = "speed_control"  # 속도 조절
    CORNERING = "cornering"  # 코너링 기술
    OVERALL_SAFETY = "overall_safety"  # 전반적 안전성


class CoachingLevel(Enum):
    """코칭 수준"""

    BEGINNER = "beginner"  # 초보자
    INTERMEDIATE = "intermediate"  # 중급자
    ADVANCED = "advanced"  # 고급자
    EXPERT = "expert"  # 전문가


class FeedbackType(Enum):
    """피드백 유형"""

    REAL_TIME = "real_time"  # 실시간 피드백
    SESSION_END = "session_end"  # 운전 종료 후 피드백
    WEEKLY_REPORT = "weekly_report"  # 주간 리포트
    ACHIEVEMENT = "achievement"  # 성취 달성


@dataclass
class DrivingMetrics:
    """운전 메트릭"""

    steering_smoothness: float = 0.0  # 0.0-1.0 (부드러움)
    steering_jerk: float = 0.0  # 핸들 저크 수치
    posture_score: float = 0.0  # 자세 점수
    attention_score: float = 0.0  # 주의집중 점수
    gaze_distribution: float = 0.0  # 시선 분산도
    reaction_time: float = 0.0  # 반응 시간 (ms)
    session_duration: float = 0.0  # 세션 지속 시간 (분)
    timestamp: float = field(default_factory=time.time)


@dataclass
class CoachingFeedback:
    """코칭 피드백"""

    category: DrivingSkillCategory
    feedback_type: FeedbackType
    priority: int  # 1(높음) - 5(낮음)
    message: str  # 피드백 메시지
    suggestion: str  # 개선 제안
    achievement_points: int = 0  # 성취 포인트
    improvement_percentage: float = 0.0  # 개선율


@dataclass
class DrivingSession:
    """운전 세션"""

    session_id: str
    start_time: float
    end_time: Optional[float] = None
    metrics: List[DrivingMetrics] = field(default_factory=list)
    feedback: List[CoachingFeedback] = field(default_factory=list)
    overall_score: float = 0.0
    improvements: Dict[str, float] = field(default_factory=dict)


@dataclass
class DrivingProfile:
    """운전자 프로필"""

    user_id: str
    coaching_level: CoachingLevel
    preferred_feedback_style: str  # "gentle", "direct", "motivational"
    target_skills: List[DrivingSkillCategory] = field(default_factory=list)
    historical_scores: Dict[str, List[float]] = field(default_factory=dict)
    total_driving_hours: float = 0.0
    achievements: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


class AIDrivingCoach:
    """AI 드라이빙 코치 메인 시스템"""

    def __init__(self, user_id: str = "default"):
        self.config = get_config()
        self.user_id = user_id
        self.current_session: Optional[DrivingSession] = None
        self.driving_profile = self._load_driving_profile()

        # 실시간 데이터 버퍼
        self.metrics_buffer = deque(maxlen=300)  # 10초 @ 30fps
        self.feedback_history = deque(maxlen=100)

        # 코칭 규칙 및 임계값
        self.coaching_rules = self._initialize_coaching_rules()
        self.feedback_cooldowns = {}  # 피드백 쿨다운 관리

        # 성과 추적
        self.baseline_metrics = self._calculate_baseline_metrics()

        print(f"🎓 AI 드라이빙 코치 초기화 완료 - 사용자: {user_id}")
        print(f"   현재 레벨: {self.driving_profile.coaching_level.value}")
        print(f"   총 운전 시간: {self.driving_profile.total_driving_hours:.1f}시간")

    def _load_driving_profile(self) -> DrivingProfile:
        """운전자 프로필 로드"""
        profile_path = Path(f"profiles/driving_coach_{self.user_id}.json")

        if profile_path.exists():
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return DrivingProfile(
                        user_id=data.get("user_id", self.user_id),
                        coaching_level=CoachingLevel(
                            data.get("coaching_level", "beginner")
                        ),
                        preferred_feedback_style=data.get(
                            "preferred_feedback_style", "gentle"
                        ),
                        target_skills=[
                            DrivingSkillCategory(s)
                            for s in data.get("target_skills", [])
                        ],
                        historical_scores=data.get("historical_scores", {}),
                        total_driving_hours=data.get("total_driving_hours", 0.0),
                        achievements=data.get("achievements", []),
                        last_updated=data.get("last_updated", time.time()),
                    )
            except Exception as e:
                print(f"프로필 로드 실패: {e}, 기본 프로필 생성")

        # 기본 프로필 생성
        return DrivingProfile(
            user_id=self.user_id,
            coaching_level=CoachingLevel.BEGINNER,
            preferred_feedback_style="gentle",
            target_skills=[
                DrivingSkillCategory.STEERING_SMOOTHNESS,
                DrivingSkillCategory.ATTENTION_MANAGEMENT,
            ],
        )

    def _save_driving_profile(self):
        """운전자 프로필 저장"""
        profile_path = Path(f"profiles/driving_coach_{self.user_id}.json")
        profile_path.parent.mkdir(exist_ok=True)

        data = {
            "user_id": self.driving_profile.user_id,
            "coaching_level": self.driving_profile.coaching_level.value,
            "preferred_feedback_style": self.driving_profile.preferred_feedback_style,
            "target_skills": [s.value for s in self.driving_profile.target_skills],
            "historical_scores": self.driving_profile.historical_scores,
            "total_driving_hours": self.driving_profile.total_driving_hours,
            "achievements": self.driving_profile.achievements,
            "last_updated": self.driving_profile.last_updated,
        }

        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _initialize_coaching_rules(self) -> Dict[str, Dict]:
        """코칭 규칙 초기화"""
        return {
            DrivingSkillCategory.STEERING_SMOOTHNESS.value: {
                "excellent_threshold": 0.9,
                "good_threshold": 0.75,
                "needs_improvement_threshold": 0.5,
                "feedback_messages": {
                    "gentle": {
                        "improvement": "핸들을 조금 더 부드럽게 조작해보세요. 급격한 움직임보다는 천천히 회전시키면 더 안정적입니다.",
                        "good": "핸들 조작이 안정적이네요! 이 상태를 유지해주세요.",
                        "excellent": "완벽한 핸들 컨트롤입니다! 매우 부드럽고 안정적이에요.",
                    },
                    "direct": {
                        "improvement": "핸들 조작이 너무 급격합니다. 저크를 줄이세요.",
                        "good": "핸들 조작 양호. 현재 수준 유지하세요.",
                        "excellent": "탁월한 핸들 컨트롤. 전문가 수준입니다.",
                    },
                    "motivational": {
                        "improvement": "핸들링 스킬을 향상시킬 좋은 기회입니다! 부드러운 조작으로 더 나은 드라이버가 되어보세요!",
                        "good": "멋진 핸들링이에요! 계속 이런 식으로 해보세요!",
                        "excellent": "환상적인 핸들 컨트롤! 당신은 정말 숙련된 드라이버군요!",
                    },
                },
            },
            DrivingSkillCategory.POSTURE_STABILITY.value: {
                "excellent_threshold": 0.85,
                "good_threshold": 0.7,
                "needs_improvement_threshold": 0.5,
                "feedback_messages": {
                    "gentle": {
                        "improvement": "허리를 펴고 바른 자세를 유지하면 피로를 줄일 수 있습니다. 등받이에 기대어 편안한 자세로 앉아보세요.",
                        "good": "좋은 운전 자세를 유지하고 계시네요!",
                        "excellent": "완벽한 운전 자세입니다! 척추 건강에도 좋아요.",
                    }
                },
            },
            DrivingSkillCategory.ATTENTION_MANAGEMENT.value: {
                "excellent_threshold": 0.9,
                "good_threshold": 0.75,
                "needs_improvement_threshold": 0.6,
                "feedback_messages": {
                    "gentle": {
                        "improvement": "시야를 넓게 보고, 좌우 미러도 주기적으로 확인해주세요. 한 곳에만 집중하지 마시고 전체적으로 살펴보는 것이 좋습니다.",
                        "good": "주의집중이 잘 되고 있어요! 계속 주변을 살피며 운전해주세요.",
                        "excellent": "훌륭한 상황 인식 능력입니다! 모든 방향을 골고루 체크하고 계시네요.",
                    }
                },
            },
        }

    def _calculate_baseline_metrics(self) -> Dict[str, float]:
        """사용자 기준 메트릭 계산"""
        if not self.driving_profile.historical_scores:
            return {skill.value: 0.5 for skill in DrivingSkillCategory}

        baseline = {}
        for skill in DrivingSkillCategory:
            scores = self.driving_profile.historical_scores.get(skill.value, [0.5])
            baseline[skill.value] = np.mean(scores[-10:])  # 최근 10개 평균

        return baseline

    async def start_driving_session(self) -> str:
        """운전 세션 시작"""
        session_id = f"session_{int(time.time())}"
        self.current_session = DrivingSession(
            session_id=session_id, start_time=time.time()
        )

        print(f"🚗 운전 세션 시작: {session_id}")
        print(f"   목표 기술: {[s.value for s in self.driving_profile.target_skills]}")

        return session_id

    async def process_real_time_data(self, ui_state: UIState) -> List[CoachingFeedback]:
        """실시간 데이터 처리 및 코칭"""
        if not self.current_session:
            return []

        # UI 상태에서 운전 메트릭 추출
        metrics = self._extract_driving_metrics(ui_state)
        self.metrics_buffer.append(metrics)
        self.current_session.metrics.append(metrics)

        # 실시간 피드백 생성
        feedback = await self._generate_real_time_feedback(metrics)

        if feedback:
            self.current_session.feedback.extend(feedback)
            self.feedback_history.extend(feedback)

        return feedback

    def _extract_driving_metrics(self, ui_state: UIState) -> DrivingMetrics:
        """UI 상태에서 운전 메트릭 추출"""
        # 핸들 조작 부드러움 (손 데이터에서)
        steering_smoothness = 1.0 - min(1.0, ui_state.hands.tremor_frequency or 0.0)
        steering_jerk = ui_state.hands.tremor_frequency or 0.0

        # 자세 점수 (자세 데이터에서)
        posture_score = ui_state.posture.spinal_alignment_score

        # 주의집중 점수 (시선 데이터에서)
        attention_score = ui_state.gaze.attention_score

        # 시선 분산도 (주의산만 수준 역수)
        gaze_distribution = 1.0 - ui_state.gaze.distraction_level

        return DrivingMetrics(
            steering_smoothness=steering_smoothness,
            steering_jerk=steering_jerk,
            posture_score=posture_score,
            attention_score=attention_score,
            gaze_distribution=gaze_distribution,
            session_duration=(time.time() - self.current_session.start_time) / 60.0,
        )

    async def _generate_real_time_feedback(
        self, metrics: DrivingMetrics
    ) -> List[CoachingFeedback]:
        """실시간 피드백 생성"""
        feedback_list = []
        current_time = time.time()

        # 각 기술 카테고리별 평가
        for skill in self.driving_profile.target_skills:
            if skill == DrivingSkillCategory.STEERING_SMOOTHNESS:
                score = metrics.steering_smoothness
                feedback = self._evaluate_skill_performance(skill, score, current_time)
                if feedback:
                    feedback_list.append(feedback)

            elif skill == DrivingSkillCategory.POSTURE_STABILITY:
                score = metrics.posture_score
                feedback = self._evaluate_skill_performance(skill, score, current_time)
                if feedback:
                    feedback_list.append(feedback)

            elif skill == DrivingSkillCategory.ATTENTION_MANAGEMENT:
                score = metrics.attention_score
                feedback = self._evaluate_skill_performance(skill, score, current_time)
                if feedback:
                    feedback_list.append(feedback)

        return feedback_list

    def _evaluate_skill_performance(
        self, skill: DrivingSkillCategory, score: float, current_time: float
    ) -> Optional[CoachingFeedback]:
        """기술 성능 평가 및 피드백 생성"""
        skill_name = skill.value
        rules = self.coaching_rules.get(skill_name, {})

        # 쿨다운 체크 (같은 기술에 대해 30초 간격)
        last_feedback_time = self.feedback_cooldowns.get(skill_name, 0)
        if current_time - last_feedback_time < 30.0:
            return None

        # 성능 수준 판정
        excellent_threshold = rules.get("excellent_threshold", 0.9)
        good_threshold = rules.get("good_threshold", 0.75)
        needs_improvement_threshold = rules.get("needs_improvement_threshold", 0.5)

        if score >= excellent_threshold:
            performance_level = "excellent"
            priority = 5  # 낮은 우선순위 (칭찬)
        elif score >= good_threshold:
            performance_level = "good"
            priority = 4
        elif score >= needs_improvement_threshold:
            performance_level = "improvement"
            priority = 2  # 높은 우선순위 (개선 필요)
        else:
            performance_level = "improvement"
            priority = 1  # 최고 우선순위 (즉시 개선 필요)

        # 피드백 메시지 생성
        messages = rules.get("feedback_messages", {}).get(
            self.driving_profile.preferred_feedback_style, {}
        )
        message = messages.get(performance_level, f"{skill_name} 성능: {score:.2f}")

        # 개선 정도 계산
        baseline = self.baseline_metrics.get(skill_name, 0.5)
        improvement = ((score - baseline) / baseline) * 100 if baseline > 0 else 0

        self.feedback_cooldowns[skill_name] = current_time

        return CoachingFeedback(
            category=skill,
            feedback_type=FeedbackType.REAL_TIME,
            priority=priority,
            message=message,
            suggestion=self._generate_improvement_suggestion(skill, score),
            improvement_percentage=improvement,
        )

    def _generate_improvement_suggestion(
        self, skill: DrivingSkillCategory, score: float
    ) -> str:
        """개선 제안 생성"""
        suggestions = {
            DrivingSkillCategory.STEERING_SMOOTHNESS: [
                "핸들을 더 부드럽게 돌려보세요",
                "급격한 조작을 피하고 천천히 회전시키세요",
                "핸들을 양손으로 균등하게 잡고 조작하세요",
            ],
            DrivingSkillCategory.POSTURE_STABILITY: [
                "등받이에 등을 기대고 바른 자세를 유지하세요",
                "어깨의 힘을 빼고 편안하게 앉으세요",
                "주기적으로 어깨를 돌려 긴장을 풀어주세요",
            ],
            DrivingSkillCategory.ATTENTION_MANAGEMENT: [
                "전방뿐만 아니라 좌우 미러도 확인하세요",
                "시선을 한 곳에 고정하지 말고 전체적으로 살펴보세요",
                "정기적으로 후방 미러도 체크하세요",
            ],
        }

        skill_suggestions = suggestions.get(skill, ["지속적인 연습이 필요합니다"])

        # 점수에 따라 적절한 제안 선택
        if score < 0.5:
            return skill_suggestions[0]  # 기본 제안
        elif score < 0.75:
            return (
                skill_suggestions[1]
                if len(skill_suggestions) > 1
                else skill_suggestions[0]
            )
        else:
            return skill_suggestions[-1]  # 고급 제안

    async def end_driving_session(self) -> Dict[str, Any]:
        """운전 세션 종료 및 세션 리포트 생성"""
        if not self.current_session:
            return {"error": "활성 세션이 없습니다"}

        self.current_session.end_time = time.time()
        session_duration = (
            self.current_session.end_time - self.current_session.start_time
        ) / 60.0

        # 세션 분석
        session_report = await self._analyze_driving_session()

        # 프로필 업데이트
        self._update_driving_profile(session_report)

        # 성취 달성 체크
        achievements = self._check_achievements(session_report)

        # 세션 저장
        self._save_driving_session()

        print(f"🏁 운전 세션 종료 - 지속시간: {session_duration:.1f}분")
        print(f"   전체 점수: {session_report['overall_score']:.1f}/100")

        self.current_session = None

        return {
            "session_report": session_report,
            "achievements": achievements,
            "duration_minutes": session_duration,
        }

    async def _analyze_driving_session(self) -> Dict[str, Any]:
        """운전 세션 분석"""
        if not self.current_session.metrics:
            return {"overall_score": 0.0, "category_scores": {}}

        # 카테고리별 점수 계산
        category_scores = {}

        # 핸들 조작 부드러움
        steering_scores = [m.steering_smoothness for m in self.current_session.metrics]
        category_scores[DrivingSkillCategory.STEERING_SMOOTHNESS.value] = (
            np.mean(steering_scores) * 100
        )

        # 자세 안정성
        posture_scores = [m.posture_score for m in self.current_session.metrics]
        category_scores[DrivingSkillCategory.POSTURE_STABILITY.value] = (
            np.mean(posture_scores) * 100
        )

        # 주의 관리
        attention_scores = [m.attention_score for m in self.current_session.metrics]
        category_scores[DrivingSkillCategory.ATTENTION_MANAGEMENT.value] = (
            np.mean(attention_scores) * 100
        )

        # 전체 점수 (가중 평균)
        weights = {
            DrivingSkillCategory.STEERING_SMOOTHNESS.value: 0.3,
            DrivingSkillCategory.POSTURE_STABILITY.value: 0.2,
            DrivingSkillCategory.ATTENTION_MANAGEMENT.value: 0.5,
        }

        overall_score = sum(
            category_scores[cat] * weights.get(cat, 0.2) for cat in category_scores
        )

        # 개선사항 계산
        improvements = {}
        for category, score in category_scores.items():
            baseline = self.baseline_metrics.get(category, 50.0) * 100
            improvement = ((score - baseline) / baseline) * 100 if baseline > 0 else 0
            improvements[category] = improvement

        self.current_session.overall_score = overall_score
        self.current_session.improvements = improvements

        return {
            "overall_score": overall_score,
            "category_scores": category_scores,
            "improvements": improvements,
            "feedback_count": len(self.current_session.feedback),
            "session_duration_minutes": (time.time() - self.current_session.start_time)
            / 60.0,
        }

    def _update_driving_profile(self, session_report: Dict[str, Any]):
        """운전 프로필 업데이트"""
        # 운전 시간 누적
        session_duration_hours = session_report["session_duration_minutes"] / 60.0
        self.driving_profile.total_driving_hours += session_duration_hours

        # 히스토리 점수 업데이트
        for category, score in session_report["category_scores"].items():
            if category not in self.driving_profile.historical_scores:
                self.driving_profile.historical_scores[category] = []

            self.driving_profile.historical_scores[category].append(score)

            # 최근 20개만 유지
            if len(self.driving_profile.historical_scores[category]) > 20:
                self.driving_profile.historical_scores[category] = (
                    self.driving_profile.historical_scores[category][-20:]
                )

        # 레벨 업 체크
        self._check_level_up()

        # 베이스라인 재계산
        self.baseline_metrics = self._calculate_baseline_metrics()

        # 프로필 저장
        self.driving_profile.last_updated = time.time()
        self._save_driving_profile()

    def _check_level_up(self):
        """레벨 업 체크"""
        total_hours = self.driving_profile.total_driving_hours
        avg_scores = {}

        for category, scores in self.driving_profile.historical_scores.items():
            if scores:
                avg_scores[category] = np.mean(scores[-5:])  # 최근 5세션 평균

        overall_avg = np.mean(list(avg_scores.values())) if avg_scores else 0

        # 레벨업 조건
        if (
            self.driving_profile.coaching_level == CoachingLevel.BEGINNER
            and total_hours >= 10
            and overall_avg >= 70
        ):
            self.driving_profile.coaching_level = CoachingLevel.INTERMEDIATE
            print("🎉 레벨 업! 중급자 레벨에 도달했습니다!")

        elif (
            self.driving_profile.coaching_level == CoachingLevel.INTERMEDIATE
            and total_hours >= 50
            and overall_avg >= 80
        ):
            self.driving_profile.coaching_level = CoachingLevel.ADVANCED
            print("🎉 레벨 업! 고급자 레벨에 도달했습니다!")

        elif (
            self.driving_profile.coaching_level == CoachingLevel.ADVANCED
            and total_hours >= 100
            and overall_avg >= 90
        ):
            self.driving_profile.coaching_level = CoachingLevel.EXPERT
            print("🎉 레벨 업! 전문가 레벨에 도달했습니다!")

    def _check_achievements(self, session_report: Dict[str, Any]) -> List[str]:
        """성취 달성 체크"""
        new_achievements = []

        # 성취 조건들
        achievements_config = {
            "perfect_steering": {
                "condition": session_report["category_scores"].get(
                    DrivingSkillCategory.STEERING_SMOOTHNESS.value, 0
                )
                >= 95,
                "title": "완벽한 핸들링 마스터",
                "description": "핸들 조작에서 95점 이상 달성",
            },
            "attention_master": {
                "condition": session_report["category_scores"].get(
                    DrivingSkillCategory.ATTENTION_MANAGEMENT.value, 0
                )
                >= 95,
                "title": "주의집중 달인",
                "description": "주의 관리에서 95점 이상 달성",
            },
            "consistent_driver": {
                "condition": all(
                    score >= 80 for score in session_report["category_scores"].values()
                ),
                "title": "일관성 있는 드라이버",
                "description": "모든 카테고리에서 80점 이상 달성",
            },
            "improvement_champion": {
                "condition": any(
                    improvement >= 20
                    for improvement in session_report["improvements"].values()
                ),
                "title": "개선의 챔피언",
                "description": "한 세션에서 20% 이상 개선",
            },
        }

        for achievement_id, config in achievements_config.items():
            if (
                config["condition"]
                and achievement_id not in self.driving_profile.achievements
            ):
                self.driving_profile.achievements.append(achievement_id)
                new_achievements.append(
                    {
                        "id": achievement_id,
                        "title": config["title"],
                        "description": config["description"],
                    }
                )
                print(f"🏆 새로운 성취 달성: {config['title']}")

        return new_achievements

    def _save_driving_session(self):
        """운전 세션 저장"""
        if not self.current_session:
            return

        sessions_dir = Path("profiles/driving_sessions")
        sessions_dir.mkdir(exist_ok=True)

        session_file = sessions_dir / f"{self.current_session.session_id}.json"

        # 세션 데이터 직렬화
        session_data = {
            "session_id": self.current_session.session_id,
            "user_id": self.user_id,
            "start_time": self.current_session.start_time,
            "end_time": self.current_session.end_time,
            "overall_score": self.current_session.overall_score,
            "improvements": self.current_session.improvements,
            "feedback_count": len(self.current_session.feedback),
            "metrics_count": len(self.current_session.metrics),
        }

        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

    def get_driving_statistics(self) -> Dict[str, Any]:
        """운전 통계 조회"""
        stats = {
            "total_hours": self.driving_profile.total_driving_hours,
            "current_level": self.driving_profile.coaching_level.value,
            "achievements_count": len(self.driving_profile.achievements),
            "target_skills": [s.value for s in self.driving_profile.target_skills],
            "recent_scores": {},
        }

        # 최근 점수
        for category, scores in self.driving_profile.historical_scores.items():
            if scores:
                stats["recent_scores"][category] = {
                    "latest": scores[-1],
                    "average": np.mean(scores[-5:]),  # 최근 5개 평균
                    "trend": "improving"
                    if len(scores) > 1 and scores[-1] > scores[-2]
                    else "stable",
                }

        return stats

    async def get_personalized_recommendations(self) -> List[str]:
        """개인화된 추천사항"""
        recommendations = []

        # 약점 분야 식별
        weak_areas = []
        for category, scores in self.driving_profile.historical_scores.items():
            if scores and np.mean(scores[-3:]) < 70:  # 최근 3개 평균이 70 미만
                weak_areas.append(category)

        # 추천사항 생성
        for area in weak_areas:
            if area == DrivingSkillCategory.STEERING_SMOOTHNESS.value:
                recommendations.append(
                    "핸들 조작 연습: 빈 주차장에서 8자 주행 연습을 해보세요"
                )
            elif area == DrivingSkillCategory.ATTENTION_MANAGEMENT.value:
                recommendations.append(
                    "주의력 향상: 운전 중 의식적으로 미러 체크 횟수를 늘려보세요"
                )
            elif area == DrivingSkillCategory.POSTURE_STABILITY.value:
                recommendations.append(
                    "자세 개선: 운전 전 좌석과 미러 위치를 정확히 조정해보세요"
                )

        # 레벨별 일반 추천
        if self.driving_profile.coaching_level == CoachingLevel.BEGINNER:
            recommendations.append(
                "기초 향상: 매일 30분씩 꾸준한 운전 연습이 실력 향상에 도움됩니다"
            )
        elif self.driving_profile.coaching_level == CoachingLevel.INTERMEDIATE:
            recommendations.append(
                "심화 연습: 다양한 도로 환경에서 운전 경험을 쌓아보세요"
            )

        return recommendations[:3]  # 최대 3개 추천
