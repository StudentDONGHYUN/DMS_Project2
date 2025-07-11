import numpy as np
import math
from collections import deque
import logging
from core.definitions import GazeZone

logger = logging.getLogger(__name__)

class EnhancedSphericalGazeClassifier:
    """향상된 3D 구 형상 시선 분석 모델"""

    def __init__(self):
        self.zones = [
            (GazeZone.FRONT, self._normalize([0.0, 0.0, 1.0]), 25.0),
            (GazeZone.REARVIEW_MIRROR, self._normalize([0.0, 0.5, 1.0]), 15.0),
            (GazeZone.ROOF, self._normalize([0.0, 1.0, 0.3]), 20.0),
            (GazeZone.INSTRUMENT_CLUSTER, self._normalize([0.0, -0.3, 1.0]), 20.0),
            (GazeZone.CENTER_STACK, self._normalize([0.3, -0.2, 0.8]), 18.0),
            (GazeZone.FLOOR, self._normalize([0.0, -1.0, 0.5]), 25.0),
            (GazeZone.LEFT_SIDE_MIRROR, self._normalize([-0.8, 0.2, 0.6]), 12.0),
            (GazeZone.DRIVER_WINDOW, self._normalize([-1.0, 0.0, 0.2]), 30.0),
            (GazeZone.BLIND_SPOT_LEFT, self._normalize([-0.6, -0.2, -0.8]), 20.0),
            (GazeZone.RIGHT_SIDE_MIRROR, self._normalize([0.8, 0.2, 0.6]), 12.0),
            (GazeZone.PASSENGER, self._normalize([1.0, 0.0, 0.5]), 25.0),
        ]
        self.gaze_history = deque(maxlen=30)
        self.zone_duration_tracker = {}
        logger.info(f"EnhancedSphericalGazeClassifier 초기화: {len(self.zones)}개 구역 정의됨")

    def _normalize(self, v):
        v = np.array(v, dtype=np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v

    def _angles_to_vector(self, yaw, pitch):
        yaw_rad, pitch_rad = math.radians(yaw), math.radians(pitch)
        x = math.sin(yaw_rad) * math.cos(pitch_rad)
        y = math.sin(pitch_rad)
        z = math.cos(yaw_rad) * math.cos(pitch_rad)
        return self._normalize([x, y, z])

    def classify(self, yaw, pitch, timestamp=None):
        gaze_vector = self._angles_to_vector(yaw, pitch)
        best_match, min_angle = GazeZone.UNKNOWN, float("inf")

        for zone, ref_vec, radius in self.zones:
            dot_product = np.clip(np.dot(gaze_vector, ref_vec), -1.0, 1.0)
            angle_deg = math.degrees(math.acos(abs(dot_product)))
            if angle_deg <= radius and angle_deg < min_angle:
                min_angle, best_match = angle_deg, zone

        if timestamp:
            self.gaze_history.append(
                {
                    "timestamp": timestamp,
                    "zone": best_match,
                    "confidence": 1.0 - (min_angle / 90.0),
                }
            )

        return best_match

    def get_gaze_stability(self) -> float:
        if len(self.gaze_history) < 10:
            return 1.0
        recent_zones = [entry["zone"] for entry in list(self.gaze_history)[-10:]]
        unique_zones = len(set(recent_zones))
        stability = max(0.0, 1.0 - (unique_zones - 1) / 9.0)
        return stability

    def get_attention_focus_score(self) -> float:
        if len(self.gaze_history) < 5:
            return 1.0
        front_gaze_count = sum(
            1 for entry in self.gaze_history if entry["zone"] == GazeZone.FRONT
        )
        focus_ratio = front_gaze_count / len(self.gaze_history)
        mirror_zones = {
            GazeZone.REARVIEW_MIRROR,
            GazeZone.LEFT_SIDE_MIRROR,
            GazeZone.RIGHT_SIDE_MIRROR,
        }
        mirror_gaze_count = sum(
            1 for entry in self.gaze_history if entry["zone"] in mirror_zones
        )
        mirror_ratio = mirror_gaze_count / len(self.gaze_history)
        if 0.1 <= mirror_ratio <= 0.3:
            focus_ratio += mirror_ratio * 0.5
        return min(1.0, focus_ratio)
