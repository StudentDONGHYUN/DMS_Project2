"""
Personalization Engine - Enhanced System
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
                
                logger.info(f"User profile loaded for {self.user_id}")
            else:
                self._create_default_profile()
                logger.info(f"Created default profile for new user: {self.user_id}")
                
        except Exception as e:
            logger.error(f"Error loading user profile: {e}")
            self._create_default_profile()

    def _create_default_profile(self):
        """기본 프로필 생성"""
        self.user_profile = {
            'user_id': self.user_id,
            'created_at': time.time(),
            'baseline_established': False,
            'adaptation_level': 0.0,
            'baseline_data': {},
            'thresholds': {},
            'preferences': {},
            'session_count': 0
        }

    def _save_user_profile(self):
        """사용자 프로필 저장"""
        try:
            self.user_profile['last_updated'] = time.time()
            self.user_profile['baseline_established'] = self.baseline_established
            self.user_profile['adaptation_level'] = self.adaptation_level
            
            profile_file = self.profiles_dir / f"{self.user_id}_profile.json"
            
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_profile, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"User profile saved for {self.user_id}")
            
        except Exception as e:
            logger.error(f"Error saving user profile: {e}")

    def update_baseline(self, data: Dict[str, Any]):
        """베이스라인 데이터 업데이트"""
        try:
            if 'baseline_data' not in self.user_profile:
                self.user_profile['baseline_data'] = {}
            
            # Update baseline data
            for key, value in data.items():
                if key in self.user_profile['baseline_data']:
                    # Running average
                    current = self.user_profile['baseline_data'][key]
                    self.user_profile['baseline_data'][key] = (current + value) / 2
                else:
                    self.user_profile['baseline_data'][key] = value
            
            # Check if we have enough data to establish baseline
            required_keys = ['drowsiness_avg', 'emotion_baseline', 'gaze_pattern']
            if all(key in self.user_profile['baseline_data'] for key in required_keys):
                self.baseline_established = True
                self.adaptation_level = min(1.0, self.adaptation_level + 0.1)
                logger.info(f"Baseline established for user {self.user_id}")
            
            self._save_user_profile()
            
        except Exception as e:
            logger.error(f"Error updating baseline: {e}")

    def get_personalized_thresholds(self, base_thresholds: Dict[str, float]) -> Dict[str, float]:
        """개인화된 임계값 반환"""
        try:
            if not self.baseline_established:
                return base_thresholds
            
            personalized = base_thresholds.copy()
            baseline_data = self.user_profile.get('baseline_data', {})
            
            # Adjust thresholds based on user's baseline
            if 'drowsiness_avg' in baseline_data:
                drowsiness_factor = baseline_data['drowsiness_avg'] / 0.5  # Normalize to expected baseline
                personalized['drowsiness_threshold'] = base_thresholds.get('drowsiness_threshold', 0.7) * drowsiness_factor
            
            if 'emotion_baseline' in baseline_data:
                emotion_factor = baseline_data['emotion_baseline'] / 0.5
                personalized['emotion_threshold'] = base_thresholds.get('emotion_threshold', 0.6) * emotion_factor
            
            # Apply adaptation level
            for key in personalized:
                adaptation_factor = 1.0 - (self.adaptation_level * 0.1)  # Max 10% adjustment
                personalized[key] *= adaptation_factor
            
            return personalized
            
        except Exception as e:
            logger.error(f"Error getting personalized thresholds: {e}")
            return base_thresholds

    def add_session_data(self, data: Dict[str, Any]):
        """세션 데이터 추가"""
        try:
            data['timestamp'] = time.time()
            self.session_data.append(data)
            
            # Keep only recent session data
            if len(self.session_data) > 1000:
                self.session_data = self.session_data[-1000:]
                
        except Exception as e:
            logger.error(f"Error adding session data: {e}")

    def get_session_summary(self) -> Dict[str, Any]:
        """세션 요약 반환"""
        try:
            if not self.session_data:
                return {}
            
            session_duration = time.time() - self.session_start_time
            
            # Calculate averages
            drowsiness_values = [d.get('drowsiness_score', 0) for d in self.session_data if 'drowsiness_score' in d]
            emotion_values = [d.get('emotion_score', 0) for d in self.session_data if 'emotion_score' in d]
            attention_values = [d.get('attention_score', 0) for d in self.session_data if 'attention_score' in d]
            
            summary = {
                'session_duration_seconds': session_duration,
                'data_points': len(self.session_data),
                'avg_drowsiness': sum(drowsiness_values) / len(drowsiness_values) if drowsiness_values else 0,
                'avg_emotion': sum(emotion_values) / len(emotion_values) if emotion_values else 0,
                'avg_attention': sum(attention_values) / len(attention_values) if attention_values else 0,
                'alerts_count': len([d for d in self.session_data if d.get('alert_triggered', False)]),
                'baseline_established': self.baseline_established,
                'adaptation_level': self.adaptation_level
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting session summary: {e}")
            return {}

    def is_baseline_established(self) -> bool:
        """베이스라인 확립 여부 반환"""
        return self.baseline_established

    def get_adaptation_level(self) -> float:
        """적응 수준 반환"""
        return self.adaptation_level

    def reset_profile(self):
        """프로필 리셋"""
        try:
            # Backup current profile
            backup_file = self.profiles_dir / f"{self.user_id}_profile_backup_{int(time.time())}.json"
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_profile, f, indent=2, ensure_ascii=False)
            
            # Reset to default
            self._create_default_profile()
            self.baseline_established = False
            self.adaptation_level = 0.0
            self._save_user_profile()
            
            logger.info(f"Profile reset for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Error resetting profile: {e}")

    def get_user_profile(self) -> Dict[str, Any]:
        """사용자 프로필 반환"""
        return self.user_profile.copy()

    def close(self):
        """세션 종료 시 정리 작업"""
        try:
            # Update session count
            self.user_profile['session_count'] = self.user_profile.get('session_count', 0) + 1
            
            # Save final session summary
            session_summary = self.get_session_summary()
            if session_summary:
                if 'recent_sessions' not in self.user_profile:
                    self.user_profile['recent_sessions'] = []
                
                self.user_profile['recent_sessions'].append(session_summary)
                
                # Keep only last 10 sessions
                if len(self.user_profile['recent_sessions']) > 10:
                    self.user_profile['recent_sessions'] = self.user_profile['recent_sessions'][-10:]
            
            self._save_user_profile()
            logger.info(f"Personalization session closed for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Error closing personalization session: {e}")

    # Legacy compatibility methods
    def adapt_thresholds(self, base_thresholds):
        """Legacy compatibility method"""
        return self.get_personalized_thresholds(base_thresholds)

    def update_user_data(self, data):
        """Legacy compatibility method"""
        self.update_baseline(data)
