"""
S-Class DMS v19 - 사용자 프로필 관리 시스템
개인화된 설정, 학습 데이터, 성능 메트릭을 관리합니다.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class BiometricBaseline:
    """사용자별 생체 기준값"""
    avg_heart_rate: float = 75.0
    avg_ear: float = 0.30
    avg_attention_focus: float = 0.8
    normal_head_pose_range: Dict[str, float] = None
    
    def __post_init__(self):
        if self.normal_head_pose_range is None:
            self.normal_head_pose_range = {
                "yaw": 15.0,
                "pitch": 10.0,
                "roll": 8.0
            }


@dataclass
class DrivingBehaviorProfile:
    """운전 행동 프로필"""
    driving_style: str = "normal"  # aggressive, cautious, normal, anxious, confident, inexperienced
    skill_level: str = "intermediate"  # beginner, intermediate, advanced, expert
    steering_smoothness: float = 0.7
    attention_consistency: float = 0.8
    reaction_time: float = 0.5  # 초
    preferred_settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.preferred_settings is None:
            self.preferred_settings = {
                "coaching_frequency": "normal",
                "alert_sensitivity": "medium", 
                "ui_complexity": "standard"
            }


@dataclass
class HealthProfile:
    """건강 프로필"""
    age_range: str = "adult"  # young, adult, senior
    medical_conditions: List[str] = None
    medications: List[str] = None
    emergency_contacts: List[Dict[str, str]] = None
    health_monitoring_consent: bool = True
    share_with_doctor: bool = False
    
    def __post_init__(self):
        if self.medical_conditions is None:
            self.medical_conditions = []
        if self.medications is None:
            self.medications = []
        if self.emergency_contacts is None:
            self.emergency_contacts = []


@dataclass
class PersonalizationData:
    """개인화 학습 데이터"""
    calibration_completed: bool = False
    calibration_date: str = ""
    session_count: int = 0
    total_drive_time: float = 0.0  # 분
    
    # 적응형 임계값
    personalized_thresholds: Dict[str, float] = None
    
    # 학습된 패턴
    learned_patterns: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.personalized_thresholds is None:
            self.personalized_thresholds = {
                "fatigue_threshold": 0.6,
                "distraction_threshold": 0.5,
                "stress_threshold": 0.7
            }
        
        if self.learned_patterns is None:
            self.learned_patterns = {
                "daily_fatigue_pattern": {},
                "attention_zones": {},
                "common_distractions": []
            }


@dataclass
class PerformanceMetrics:
    """성능 지표"""
    last_session_date: str = ""
    avg_fatigue_score: float = 0.0
    avg_distraction_score: float = 0.0
    improvement_trend: str = "stable"  # improving, stable, declining
    
    # 5대 혁신 시스템별 메트릭
    ai_coach_metrics: Dict[str, Any] = None
    healthcare_metrics: Dict[str, Any] = None
    emotional_care_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.ai_coach_metrics is None:
            self.ai_coach_metrics = {
                "coaching_score": 0.0,
                "improvement_areas": [],
                "achievements": []
            }
        
        if self.healthcare_metrics is None:
            self.healthcare_metrics = {
                "avg_heart_rate": 0.0,
                "stress_episodes": 0,
                "health_alerts": 0
            }
        
        if self.emotional_care_metrics is None:
            self.emotional_care_metrics = {
                "dominant_emotions": [],
                "care_sessions": 0,
                "effectiveness_score": 0.0
            }


@dataclass
class UserProfile:
    """통합 사용자 프로필"""
    user_id: str
    display_name: str = ""
    created_date: str = ""
    last_updated: str = ""
    
    # 하위 프로필들
    biometric_baseline: BiometricBaseline = None
    driving_behavior: DrivingBehaviorProfile = None
    health_profile: HealthProfile = None
    personalization_data: PersonalizationData = None
    performance_metrics: PerformanceMetrics = None
    
    # 시스템 설정
    preferred_edition: str = "RESEARCH"
    privacy_settings: Dict[str, bool] = None
    
    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.user_id.title()
        
        if not self.created_date:
            self.created_date = datetime.now().isoformat()
        
        if not self.last_updated:
            self.last_updated = self.created_date
        
        # 하위 프로필 초기화
        if self.biometric_baseline is None:
            self.biometric_baseline = BiometricBaseline()
        if self.driving_behavior is None:
            self.driving_behavior = DrivingBehaviorProfile()
        if self.health_profile is None:
            self.health_profile = HealthProfile()
        if self.personalization_data is None:
            self.personalization_data = PersonalizationData()
        if self.performance_metrics is None:
            self.performance_metrics = PerformanceMetrics()
        
        if self.privacy_settings is None:
            self.privacy_settings = {
                "save_biometric_data": True,
                "save_session_recordings": False,
                "share_analytics": False,
                "emergency_monitoring": True
            }


class ProfileManager:
    """사용자 프로필 관리자"""
    
    def __init__(self, profiles_dir: Path = Path("profiles")):
        self.profiles_dir = profiles_dir
        self.profiles_dir.mkdir(exist_ok=True)
        
        # 로드된 프로필 캐시
        self._profile_cache: Dict[str, UserProfile] = {}
        
        logger.info(f"프로필 매니저 초기화: {self.profiles_dir}")
    
    def create_profile(self, user_id: str, **kwargs) -> UserProfile:
        """새 사용자 프로필 생성"""
        if self.profile_exists(user_id):
            logger.warning(f"프로필이 이미 존재함: {user_id}")
            return self.load_profile(user_id)
        
        profile = UserProfile(user_id=user_id, **kwargs)
        self.save_profile(profile)
        
        logger.info(f"새 프로필 생성: {user_id}")
        return profile
    
    def load_profile(self, user_id: str) -> UserProfile:
        """사용자 프로필 로드"""
        # 캐시에서 먼저 확인
        if user_id in self._profile_cache:
            return self._profile_cache[user_id]
        
        profile_path = self.profiles_dir / f"{user_id}.json"
        
        if not profile_path.exists():
            logger.info(f"프로필이 없어 새로 생성: {user_id}")
            return self.create_profile(user_id)
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 데이터클래스로 복원
            profile = self._dict_to_profile(data)
            
            # 캐시에 저장
            self._profile_cache[user_id] = profile
            
            logger.debug(f"프로필 로드 완료: {user_id}")
            return profile
            
        except Exception as e:
            logger.error(f"프로필 로드 실패 ({user_id}): {e}")
            return self.create_profile(user_id)
    
    def save_profile(self, profile: UserProfile) -> bool:
        """사용자 프로필 저장"""
        try:
            profile.last_updated = datetime.now().isoformat()
            
            profile_path = self.profiles_dir / f"{profile.user_id}.json"
            
            # 데이터클래스를 딕셔너리로 변환
            data = self._profile_to_dict(profile)
            
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # 캐시 업데이트
            self._profile_cache[profile.user_id] = profile
            
            logger.debug(f"프로필 저장 완료: {profile.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"프로필 저장 실패 ({profile.user_id}): {e}")
            return False
    
    def profile_exists(self, user_id: str) -> bool:
        """프로필 존재 여부 확인"""
        profile_path = self.profiles_dir / f"{user_id}.json"
        return profile_path.exists()
    
    def list_profiles(self) -> List[str]:
        """모든 프로필 목록 반환"""
        profile_files = self.profiles_dir.glob("*.json")
        return [f.stem for f in profile_files]
    
    def delete_profile(self, user_id: str) -> bool:
        """프로필 삭제"""
        try:
            profile_path = self.profiles_dir / f"{user_id}.json"
            if profile_path.exists():
                profile_path.unlink()
            
            # 캐시에서도 제거
            if user_id in self._profile_cache:
                del self._profile_cache[user_id]
            
            logger.info(f"프로필 삭제 완료: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"프로필 삭제 실패 ({user_id}): {e}")
            return False
    
    def update_biometric_baseline(self, user_id: str, **updates) -> bool:
        """생체 기준값 업데이트"""
        profile = self.load_profile(user_id)
        
        for key, value in updates.items():
            if hasattr(profile.biometric_baseline, key):
                setattr(profile.biometric_baseline, key, value)
        
        return self.save_profile(profile)
    
    def update_performance_metrics(self, user_id: str, session_data: Dict[str, Any]) -> bool:
        """성능 지표 업데이트"""
        profile = self.load_profile(user_id)
        
        # 세션 데이터로 메트릭 업데이트
        profile.performance_metrics.last_session_date = datetime.now().isoformat()
        profile.personalization_data.session_count += 1
        
        # 평균값 업데이트 (이동평균)
        if 'fatigue_score' in session_data:
            current_avg = profile.performance_metrics.avg_fatigue_score
            new_score = session_data['fatigue_score']
            profile.performance_metrics.avg_fatigue_score = (current_avg * 0.9) + (new_score * 0.1)
        
        if 'distraction_score' in session_data:
            current_avg = profile.performance_metrics.avg_distraction_score  
            new_score = session_data['distraction_score']
            profile.performance_metrics.avg_distraction_score = (current_avg * 0.9) + (new_score * 0.1)
        
        return self.save_profile(profile)
    
    def get_personalized_thresholds(self, user_id: str) -> Dict[str, float]:
        """개인화된 임계값 반환"""
        profile = self.load_profile(user_id)
        return profile.personalization_data.personalized_thresholds.copy()
    
    def complete_calibration(self, user_id: str, calibration_results: Dict[str, Any]) -> bool:
        """캘리브레이션 완료 처리"""
        profile = self.load_profile(user_id)
        
        # 캘리브레이션 상태 업데이트
        profile.personalization_data.calibration_completed = True
        profile.personalization_data.calibration_date = datetime.now().isoformat()
        
        # 개인화된 임계값 설정
        if 'thresholds' in calibration_results:
            profile.personalization_data.personalized_thresholds.update(
                calibration_results['thresholds']
            )
        
        # 생체 기준값 설정
        if 'biometric_baseline' in calibration_results:
            baseline_data = calibration_results['biometric_baseline']
            for key, value in baseline_data.items():
                if hasattr(profile.biometric_baseline, key):
                    setattr(profile.biometric_baseline, key, value)
        
        return self.save_profile(profile)
    
    def get_profile_summary(self, user_id: str) -> Dict[str, Any]:
        """프로필 요약 정보 반환"""
        profile = self.load_profile(user_id)
        
        return {
            "user_id": profile.user_id,
            "display_name": profile.display_name,
            "created_date": profile.created_date,
            "session_count": profile.personalization_data.session_count,
            "calibration_completed": profile.personalization_data.calibration_completed,
            "driving_style": profile.driving_behavior.driving_style,
            "skill_level": profile.driving_behavior.skill_level,
            "avg_fatigue_score": profile.performance_metrics.avg_fatigue_score,
            "improvement_trend": profile.performance_metrics.improvement_trend
        }
    
    def export_profile(self, user_id: str, export_path: Path) -> bool:
        """프로필을 외부 파일로 내보내기"""
        try:
            profile = self.load_profile(user_id)
            data = self._profile_to_dict(profile)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"프로필 내보내기 완료: {user_id} -> {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"프로필 내보내기 실패 ({user_id}): {e}")
            return False
    
    def import_profile(self, import_path: Path) -> Optional[str]:
        """외부 파일에서 프로필 가져오기"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            profile = self._dict_to_profile(data)
            
            if self.save_profile(profile):
                logger.info(f"프로필 가져오기 완료: {profile.user_id}")
                return profile.user_id
            
        except Exception as e:
            logger.error(f"프로필 가져오기 실패: {e}")
        
        return None
    
    def _profile_to_dict(self, profile: UserProfile) -> Dict[str, Any]:
        """프로필을 딕셔너리로 변환"""
        return asdict(profile)
    
    def _dict_to_profile(self, data: Dict[str, Any]) -> UserProfile:
        """딕셔너리를 프로필로 변환"""
        # 중첩된 데이터클래스들을 재귀적으로 복원
        if 'biometric_baseline' in data and data['biometric_baseline']:
            data['biometric_baseline'] = BiometricBaseline(**data['biometric_baseline'])
        
        if 'driving_behavior' in data and data['driving_behavior']:
            data['driving_behavior'] = DrivingBehaviorProfile(**data['driving_behavior'])
        
        if 'health_profile' in data and data['health_profile']:
            data['health_profile'] = HealthProfile(**data['health_profile'])
        
        if 'personalization_data' in data and data['personalization_data']:
            data['personalization_data'] = PersonalizationData(**data['personalization_data'])
        
        if 'performance_metrics' in data and data['performance_metrics']:
            data['performance_metrics'] = PerformanceMetrics(**data['performance_metrics'])
        
        return UserProfile(**data)


# 전역 프로필 매니저 인스턴스
_global_profile_manager = None


def get_profile_manager(profiles_dir: Path = None) -> ProfileManager:
    """전역 프로필 매니저 인스턴스 반환"""
    global _global_profile_manager
    if _global_profile_manager is None:
        if profiles_dir is None:
            profiles_dir = Path("profiles")
        _global_profile_manager = ProfileManager(profiles_dir)
    return _global_profile_manager


if __name__ == "__main__":
    # 프로필 매니저 테스트
    print("🧪 프로필 매니저 테스트")
    print("=" * 50)
    
    pm = get_profile_manager()
    
    # 테스트 프로필 생성
    profile = pm.create_profile("test_user", display_name="테스트 사용자")
    print(f"✅ 프로필 생성: {profile.user_id}")
    
    # 프로필 요약 출력
    summary = pm.get_profile_summary("test_user")
    print(f"📊 프로필 요약: {summary}")
    
    # 기존 프로필 목록
    profiles = pm.list_profiles()
    print(f"📋 기존 프로필들: {profiles}")