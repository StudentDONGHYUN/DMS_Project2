import numpy as np
import math
from collections import deque
import logging
from core.definitions import GazeZone

logger = logging.getLogger(__name__)

class EnhancedSphericalGazeClassifier:
    """향상된 3D 구 형상 시선 분석 모델 (동적 분류 방식 지원)"""

    def __init__(self, mode='3d'):
        # mode: '3d', 'lut', 'bbox' 중 선택
        self.mode = mode
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
        # LUT(룩업 테이블) 예시: 5도 간격, (yaw, pitch) -> zone
        self.lut_resolution = 5
        self.lut = self._build_lut()
        # BBox(경계상자) 예시: (yaw_min, yaw_max, pitch_min, pitch_max, zone)
        self.bboxes = self._define_bboxes()
        logger.info(f"EnhancedSphericalGazeClassifier 초기화: {len(self.zones)}개 구역 정의됨, mode={self.mode}")

    def set_mode(self, mode):
        assert mode in ('3d', 'lut', 'bbox')
        self.mode = mode
        logger.info(f"시선 분류 모드 변경: {mode}")

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

    def _build_lut(self):
        # LUT: (yaw, pitch) 5도 간격, -90~90 pitch, -90~90 yaw
        lut = {}
        for yaw in range(-90, 91, self.lut_resolution):
            for pitch in range(-90, 91, self.lut_resolution):
                zone = self._classify_3d(yaw, pitch)
                lut[(yaw, pitch)] = zone
        return lut

    def _lut_lookup(self, yaw, pitch):
        # 가장 가까운 LUT 인덱스 찾기
        y = int(round(yaw / self.lut_resolution) * self.lut_resolution)
        p = int(round(pitch / self.lut_resolution) * self.lut_resolution)
        y = max(-90, min(90, y))
        p = max(-90, min(90, p))
        return self.lut.get((y, p), GazeZone.UNKNOWN)

    def _define_bboxes(self):
        # 실제 차량 환경에 맞춘 yaw/pitch 경계값 (단위: 도)
        # (yaw_min, yaw_max, pitch_min, pitch_max, zone, priority)
        # priority: 낮을수록 우선 적용
        # 참고: yaw=0은 정면, pitch=0은 수평, pitch>0은 위, pitch<0은 아래
        return [
            # 1. 정면 (가장 우선)
            (-15, 15, -10, 10, GazeZone.FRONT, 1),
            # 2. 계기판 (정면보다 약간 아래)
            (-20, 20, -25, -11, GazeZone.INSTRUMENT_CLUSTER, 2),
            # 3. 센터스택 (정면보다 약간 오른쪽, 아래)
            (16, 40, -25, -5, GazeZone.CENTER_STACK, 3),
            # 4. 룸미러 (정면보다 약간 위)
            (-15, 15, 11, 25, GazeZone.REARVIEW_MIRROR, 2),
            # 5. 루프 (룸미러보다 더 위)
            (-20, 20, 26, 50, GazeZone.ROOF, 3),
            # 6. 바닥 (정면보다 아래)
            (-20, 20, -50, -26, GazeZone.FLOOR, 3),
            # 7. 좌측 사이드미러
            (-50, -16, -15, 15, GazeZone.LEFT_SIDE_MIRROR, 2),
            # 8. 우측 사이드미러
            (16, 50, -15, 15, GazeZone.RIGHT_SIDE_MIRROR, 2),
            # 9. 운전석 창문 (좌측 끝)
            (-90, -51, -30, 30, GazeZone.DRIVER_WINDOW, 2),
            # 10. 조수석 (우측 끝)
            (51, 90, -30, 30, GazeZone.PASSENGER, 2),
            # 11. 블라인드 스팟(좌)
            (-90, -51, -90, -31, GazeZone.BLIND_SPOT_LEFT, 3),
            # 12. 블라인드 스팟(우)
            (51, 90, -90, -31, GazeZone.BLIND_SPOT_LEFT, 3),
            # 기타: UNKNOWN (catch-all)
        ]

    def _bbox_lookup(self, yaw, pitch):
        # 우선순위(priority) 순서대로 탐색
        for (ymin, ymax, pmin, pmax, zone, priority) in sorted(self.bboxes, key=lambda x: x[5]):
            if ymin <= yaw <= ymax and pmin <= pitch <= pmax:
                return zone
        return GazeZone.UNKNOWN

    def _classify_3d(self, yaw, pitch):
        # 기존 3D 벡터/삼각함수 방식 (내부용)
        gaze_vector = self._angles_to_vector(yaw, pitch)
        best_match, min_angle = GazeZone.UNKNOWN, float("inf")
        for zone, ref_vec, radius in self.zones:
            dot_product = np.clip(np.dot(gaze_vector, ref_vec), -1.0, 1.0)
            angle_deg = math.degrees(math.acos(abs(dot_product)))
            if angle_deg <= radius and angle_deg < min_angle:
                min_angle, best_match = angle_deg, zone
        return best_match

    def classify(self, yaw, pitch, timestamp=None):
        # 동적으로 분류 방식 선택
        if self.mode == '3d':
            zone = self._classify_3d(yaw, pitch)
        elif self.mode == 'lut':
            zone = self._lut_lookup(yaw, pitch)
        elif self.mode == 'bbox':
            zone = self._bbox_lookup(yaw, pitch)
        else:
            zone = GazeZone.UNKNOWN
        if timestamp:
            self.gaze_history.append(
                {
                    "timestamp": timestamp,
                    "zone": zone,
                    "confidence": 1.0,  # 간단화
                }
            )
        return zone

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
