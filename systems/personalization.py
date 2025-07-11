import json
from pathlib import Path
from datetime import datetime
import logging
import aiofiles

logger = logging.getLogger(__name__)

class PersonalizationEngine:
    """개인화 학습 엔진"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.adaptive_thresholds = {
            "blink_rate_normal": 20.0,
            "yawn_threshold": 0.6,
            "perclos_critical": 0.8,
            "gaze_deviation_limit": 3.0,
            "ear_threshold": 0.25,
            "emotion_sensitivity": 0.7,
            "distraction_tolerance": 0.5,
        }
        self._load_user_profile()

    def _load_user_profile(self):
        profile_path = Path("profiles") / f"{self.user_id}_profile.json"
        if profile_path.exists():
            try:
                with open(profile_path, "r") as f:
                    data = json.load(f)
                    self.adaptive_thresholds.update(data.get("thresholds", {}))
                logger.info(f"사용자 프로필 로드됨: {self.user_id}")
            except Exception as e:
                logger.warning(f"프로필 로드 실패: {e}")

    async def initialize(self):
        """
        비동기 초기화: 사용자 프로필 파일이 없으면 생성하고, 임계값을 최신 상태로 동기화
        향후 비동기 I/O 확장 고려
        """
        profile_path = Path("profiles") / f"{self.user_id}_profile.json"
        if not profile_path.exists():
            # 프로필 파일이 없으면 기본값으로 생성
            await self._async_save_profile()
        else:
            # 기존 동기 로더로 임계값 동기화
            self._load_user_profile()
        logger.info(f"PersonalizationEngine for {self.user_id} initialized.")

    async def _async_save_profile(self):
        """비동기 프로필 저장 (aiofiles 사용)"""
        profile_dir = Path("profiles")
        profile_dir.mkdir(exist_ok=True)
        profile_data = {
            "user_id": self.user_id,
            "thresholds": self.adaptive_thresholds,
            "last_updated": datetime.now().isoformat(),
        }
        try:
            async with aiofiles.open(profile_dir / f"{self.user_id}_profile.json", "w") as f:
                await f.write(json.dumps(profile_data, indent=2))
        except Exception as e:
            logger.error(f"프로필 저장 실패: {e}")

    def update_threshold(self, threshold_name: str, value: float):
        if threshold_name in self.adaptive_thresholds:
            self.adaptive_thresholds[threshold_name] = value
            logger.info(f"임계값 업데이트: {threshold_name} = {value}")

    def get_threshold(self, threshold_name: str) -> float:
        return self.adaptive_thresholds.get(threshold_name, 0.5)
