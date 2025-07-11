import logging
logger = logging.getLogger(__name__)
import math
from typing import Dict

class SensorBackupManager:
    def __init__(self):
        self.backup_modes = {
            "face_backup_active": False,
            "pose_backup_active": False,
            "hand_backup_active": False,
        }
        self.backup_quality = {
            "face_from_pose": 0.6,
            "pose_from_face": 0.4,
            "hand_from_pose": 0.8,
        }
        self.pose_face_points = {
            "nose": 0,
            "left_eye_inner": 1,
            "left_eye": 2,
            "left_eye_outer": 3,
            "right_eye_inner": 4,
            "right_eye": 5,
            "right_eye_outer": 6,
            "left_ear": 7,
            "right_ear": 8,
            "mouth_left": 9,
            "mouth_right": 10,
        }
        logger.info("SensorBackupManager 초기화 완료")

    def analyze_face_from_pose(self, pose_landmarks, pose_world_landmarks) -> Dict:
        try:
            if not pose_landmarks or len(pose_landmarks) < 11:
                return {"success": False, "reason": "insufficient_pose_landmarks"}
            pfp = self.pose_face_points
            nose, left_ear, right_ear = (
                pose_landmarks[pfp["nose"]],
                pose_landmarks[pfp["left_ear"]],
                pose_landmarks[pfp["right_ear"]],
            )
            ear_center = [
                (left_ear.x + right_ear.x) / 2,
                (left_ear.y + right_ear.y) / 2,
            ]
            yaw = math.degrees(math.atan2(nose.x - ear_center[0], 0.1))
            pitch = math.degrees(math.atan2(nose.y - ear_center[1], 0.5))
            return {
                "success": True,
                "data": {
                    "head_pose": {
                        "yaw": max(-90, min(90, yaw)),
                        "pitch": max(-60, min(60, pitch)),
                        "roll": 0.0,
                    }
                },
                "backup_quality": self.backup_quality["face_from_pose"],
            }
        except Exception as e:
            return {"success": False, "reason": str(e)}

    def analyze_hands_from_pose(self, pose_landmarks, pose_world_landmarks) -> Dict:
        try:
            if not pose_landmarks or len(pose_landmarks) < 17:
                return {"success": False, "reason": "insufficient_pose_landmarks"}
            lw, rw = pose_landmarks[15], pose_landmarks[16]
            return {
                "success": True,
                "data": {
                    "hand_positions": [
                        {
                            "handedness": "Left",
                            "x": lw.x,
                            "y": lw.y,
                            "z": getattr(lw, "z", 0.0),
                        },
                        {
                            "handedness": "Right",
                            "x": rw.x,
                            "y": rw.y,
                            "z": getattr(rw, "z", 0.0),
                        },
                    ]
                },
                "backup_quality": self.backup_quality["hand_from_pose"],
            }
        except Exception as e:
            return {"success": False, "reason": str(e)}

    def get_backup_status(self) -> Dict:
        return {
            "active_backups": [m for m, a in self.backup_modes.items() if a],
            "qualities": self.backup_quality.copy(),
        }

    def activate_backup(self, backup_type: str):
        if backup_type in self.backup_modes and not self.backup_modes[backup_type]:
            logger.warning(f"백업 모드 활성화: {backup_type}")
            self.backup_modes[backup_type] = True

    def deactivate_backup(self, backup_type: str):
        if backup_type in self.backup_modes and self.backup_modes[backup_type]:
            logger.info(f"백업 모드 비활성화: {backup_type} - 원본 센서 복구됨")
            self.backup_modes[backup_type] = False
