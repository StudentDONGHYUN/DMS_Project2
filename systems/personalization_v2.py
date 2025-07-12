"""
Personalization Engine v2
통합 시스템과 호환되는 개인화 엔진
기존 코드와 개선된 코드를 통합하여 호환성과 성능을 모두 확보
"""

import time
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PersonalizationEngine:
    """통합 개인화 엔진 - 기존 코드와 개선된 코드의 통합"""

    def __init__(self, user_id: str):
        """
        개인화 엔진 초기화
        
        Args:
            user_id: 사용자 ID
        """
        self.user_id = user_id
        self.profiles_dir = Path("profiles")
        self.profiles_dir.mkdir(exist_ok=True)
        
        # User profile
        self.user_profile = {}
        self.baseline_established = False
        self.adaptation_level = 0.0
        
        # Session data
        self.session_start_time = time.time()
        self.session_data = []
        
        # Load user profile
        self._load_user_profile()
        
        logger.info(f"Personalization Engine initialized for user: {user_id}")

    def _load_user_profile(self):
        """사용자 프로필 로드"""
        try:
            profile_file = self.profiles_dir / f"{self.user_id}_profile.json"
            
            if profile_file.exists():
                with open(profile_file, 'r', encoding='utf-8') as f:
                    self.user_profile = json.load(f)
                
                self.baseline_established = self.user_profile.get('baseline_established', False)
                self.adaptation_level = self.user_profile.get('adaptation_level', 0.0)
                
                logger.info(f"User profile loaded: {profile_file}")
            else:
                logger.info(f"No existing profile found for user: {self.user_id}")
                
        except Exception as e:
            logger.error(f"Error loading user profile: {e}")

    def _save_user_profile(self):
        """사용자 프로필 저장"""
        try:
            profile_file = self.profiles_dir / f"{self.user_id}_profile.json"
            
            # Update profile data
            self.user_profile.update({
                'user_id': self.user_id,
                'baseline_established': self.baseline_established,
                'adaptation_level': self.adaptation_level,
                'last_updated': time.time()
            })
            
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_profile, f, indent=2, ensure_ascii=False)
            
            logger.info(f"User profile saved: {profile_file}")
            
        except Exception as e:
            logger.error(f"Error saving user profile: {e}")

    def update_baseline(self, data: Dict[str, Any]):
        """기준선 업데이트"""
        try:
            if not self.baseline_established:
                # Establish baseline
                self.user_profile['baseline'] = data.copy()
                self.baseline_established = True
                self.adaptation_level = 0.1  # Initial adaptation
                logger.info("Baseline established")
            else:
                # Update baseline gradually
                baseline = self.user_profile.get('baseline', {})
                for key, value in data.items():
                    if key in baseline:
                        # Gradual adaptation
                        baseline[key] = baseline[key] * 0.9 + value * 0.1
                    else:
                        baseline[key] = value
                
                self.adaptation_level = min(1.0, self.adaptation_level + 0.01)
            
            self._save_user_profile()
            
        except Exception as e:
            logger.error(f"Error updating baseline: {e}")

    def get_personalized_thresholds(self, base_thresholds: Dict[str, float]) -> Dict[str, float]:
        """개인화된 임계값 반환"""
        try:
            if not self.baseline_established:
                return base_thresholds
            
            baseline = self.user_profile.get('baseline', {})
            personalized = base_thresholds.copy()
            
            # Apply personalization based on baseline
            for key, base_value in base_thresholds.items():
                if key in baseline:
                    # Adjust threshold based on user's baseline
                    baseline_value = baseline[key]
                    adjustment = (baseline_value - base_value) * self.adaptation_level
                    personalized[key] = base_value + adjustment
            
            return personalized
            
        except Exception as e:
            logger.error(f"Error getting personalized thresholds: {e}")
            return base_thresholds

    def add_session_data(self, data: Dict[str, Any]):
        """세션 데이터 추가"""
        try:
            data['timestamp'] = time.time()
            self.session_data.append(data)
            
            # Keep only recent data (last 1000 entries)
            if len(self.session_data) > 1000:
                self.session_data = self.session_data[-1000:]
                
        except Exception as e:
            logger.error(f"Error adding session data: {e}")

    def get_session_summary(self) -> Dict[str, Any]:
        """세션 요약 반환"""
        try:
            if not self.session_data:
                return {}
            
            # Calculate session statistics
            session_duration = time.time() - self.session_start_time
            
            # Get unique keys from session data
            all_keys = set()
            for entry in self.session_data:
                all_keys.update(entry.keys())
            
            summary = {
                'session_duration': session_duration,
                'data_points': len(self.session_data),
                'baseline_established': self.baseline_established,
                'adaptation_level': self.adaptation_level
            }
            
            # Calculate averages for numeric values
            for key in all_keys:
                if key != 'timestamp':
                    values = [entry.get(key) for entry in self.session_data if isinstance(entry.get(key), (int, float))]
                    if values:
                        summary[f'avg_{key}'] = sum(values) / len(values)
                        summary[f'min_{key}'] = min(values)
                        summary[f'max_{key}'] = max(values)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting session summary: {e}")
            return {}

    def is_baseline_established(self) -> bool:
        """기준선이 설정되었는지 확인"""
        return self.baseline_established

    def get_adaptation_level(self) -> float:
        """적응 수준 반환"""
        return self.adaptation_level

    def reset_profile(self):
        """프로필 초기화"""
        try:
            self.user_profile = {}
            self.baseline_established = False
            self.adaptation_level = 0.0
            self.session_data.clear()
            
            # Remove profile file
            profile_file = self.profiles_dir / f"{self.user_id}_profile.json"
            if profile_file.exists():
                profile_file.unlink()
            
            logger.info(f"Profile reset for user: {self.user_id}")
            
        except Exception as e:
            logger.error(f"Error resetting profile: {e}")

    def get_user_profile(self) -> Dict[str, Any]:
        """사용자 프로필 반환"""
        return self.user_profile.copy()

    def close(self):
        """엔진 종료"""
        try:
            # Save final profile
            self._save_user_profile()
            
            # Save session data
            if self.session_data:
                session_file = self.profiles_dir / f"{self.user_id}_session_{int(time.time())}.json"
                with open(session_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'user_id': self.user_id,
                        'session_duration': time.time() - self.session_start_time,
                        'data_count': len(self.session_data),
                        'data': self.session_data
                    }, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Session data saved: {session_file}")
            
            logger.info("Personalization Engine closed")
            
        except Exception as e:
            logger.error(f"Error closing Personalization Engine: {e}")