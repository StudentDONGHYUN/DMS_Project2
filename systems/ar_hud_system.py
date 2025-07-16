"""
S-Class DMS v19.0 - 상황인지형 증강현실 HUD 시스템
전면 유리에 직접 정보를 투사하는 지능형 AR 인터페이스
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
import math
import cv2

from config.settings import get_config
from models.data_structures import UIState, GazeData


class ARObjectType(Enum):
    """AR 객체 유형"""

    WARNING_BOX = "warning_box"
    NAVIGATION_ARROW = "navigation_arrow"
    LANE_HIGHLIGHT = "lane_highlight"
    VEHICLE_INFO = "vehicle_info"
    SPEED_LIMIT = "speed_limit"
    DISTANCE_INDICATOR = "distance_indicator"
    HAZARD_MARKER = "hazard_marker"
    BLIND_SPOT_ALERT = "blind_spot_alert"
    PARKING_GUIDE = "parking_guide"
    BIOMETRIC_OVERLAY = "biometric_overlay"


class ARPriority(Enum):
    """AR 표시 우선순위"""

    CRITICAL = 1  # 즉시 위험 (빨간색)
    HIGH = 2  # 높은 위험 (주황색)
    MEDIUM = 3  # 주의 필요 (노란색)
    LOW = 4  # 일반 정보 (파란색)
    INFO = 5  # 참고 정보 (회색)


class GazeRegion(Enum):
    """시선 영역"""

    CENTER = "center"  # 중앙 (전방)
    LEFT_MIRROR = "left_mirror"  # 좌측 미러
    RIGHT_MIRROR = "right_mirror"  # 우측 미러
    REAR_MIRROR = "rear_mirror"  # 후방 미러
    DASHBOARD = "dashboard"  # 대시보드
    LEFT_BLIND = "left_blind"  # 좌측 사각지대
    RIGHT_BLIND = "right_blind"  # 우측 사각지대


@dataclass
class ARObject:
    """AR 객체"""

    object_id: str
    object_type: ARObjectType
    priority: ARPriority

    # 3D 위치 (월드 좌표)
    world_position: Tuple[float, float, float]  # (x, y, z)

    # 2D 화면 위치 (픽셀 좌표)
    screen_position: Tuple[int, int]  # (x, y)

    # 시각적 속성
    size: Tuple[int, int]  # (width, height)
    color: Tuple[int, int, int, int]  # RGBA
    thickness: int = 2

    # 내용
    text: Optional[str] = None
    icon: Optional[str] = None

    # 동작
    blink_rate: float = 0.0  # 깜빡임 주기 (0 = 고정)
    fade_duration: float = 0.0  # 페이드 지속시간 (0 = 영구)

    # 상태
    is_visible: bool = True
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    # 조건부 표시
    show_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriverGazeState:
    """운전자 시선 상태"""

    current_region: GazeRegion
    gaze_point: Tuple[float, float]  # 정규화된 좌표 (0-1)
    attention_score: float
    fixation_duration: float  # 현재 영역 고정 시간
    saccade_velocity: float  # 사카드 속도
    predicted_next_region: Optional[GazeRegion] = None
    confidence: float = 1.0


@dataclass
class VehicleContext:
    """차량 상황 정보"""

    speed_kmh: float = 0.0
    steering_angle: float = 0.0
    turn_signal: Optional[str] = None  # "left", "right", None
    gear: str = "P"  # P, R, N, D, S

    # 주변 환경
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)
    lane_info: Dict[str, Any] = field(default_factory=dict)
    traffic_signs: List[Dict[str, Any]] = field(default_factory=list)

    # 네비게이션
    next_maneuver: Optional[Dict[str, Any]] = None
    distance_to_maneuver: float = 0.0


class ARHUDSystem:
    """상황인지형 AR HUD 메인 시스템"""

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.config = get_config()
        self.screen_width = screen_width
        self.screen_height = screen_height

        # AR 객체 관리
        self.ar_objects: Dict[str, ARObject] = {}
        self.object_update_queue = deque(maxlen=1000)

        # 시선 추적
        self.gaze_tracker = GazeRegionTracker()
        self.current_gaze_state = DriverGazeState(
            current_region=GazeRegion.CENTER,
            gaze_point=(0.5, 0.5),
            attention_score=1.0,
            fixation_duration=0.0,
            saccade_velocity=0.0,
        )

        # 상황 인식 엔진
        self.context_analyzer = ContextAnalyzer()
        self.intention_predictor = IntentionPredictor()

        # 렌더링 엔진
        self.ar_renderer = ARRenderer(screen_width, screen_height)

        # 설정
        self.enable_predictive_highlighting = True
        self.adaptive_brightness = True
        self.min_fixation_time = 0.2  # 최소 시선 고정 시간

        print(f"🥽 AR HUD 시스템 초기화 완료")
        print(f"   화면 해상도: {screen_width}x{screen_height}")
        print(
            f"   예측 하이라이팅: {'ON' if self.enable_predictive_highlighting else 'OFF'}"
        )

    async def process_frame(
        self, ui_state: UIState, vehicle_context: VehicleContext
    ) -> np.ndarray:
        """프레임 처리 및 AR 오버레이 생성"""

        # 1. 시선 상태 업데이트
        await self._update_gaze_state(ui_state.gaze)

        # 2. 상황 분석
        context_info = await self.context_analyzer.analyze_situation(
            vehicle_context, self.current_gaze_state
        )

        # 3. 운전자 의도 예측
        driver_intention = await self.intention_predictor.predict_intention(
            self.current_gaze_state, vehicle_context, context_info
        )

        # 4. AR 객체 생성/업데이트
        await self._update_ar_objects(
            ui_state, vehicle_context, context_info, driver_intention
        )

        # 5. 시선 기반 적응형 표시
        await self._adapt_display_to_gaze()

        # 6. AR 프레임 렌더링
        ar_frame = await self.ar_renderer.render_frame(
            self.ar_objects,
            self.current_gaze_state,
            adaptive_brightness=self.adaptive_brightness,
        )

        return ar_frame

    async def _update_gaze_state(self, gaze_data: GazeData):
        """시선 상태 업데이트"""
        # 시선 영역 판정
        current_region = self.gaze_tracker.determine_gaze_region(
            gaze_data.gaze_x, gaze_data.gaze_y
        )

        # 고정 시간 계산
        if current_region == self.current_gaze_state.current_region:
            self.current_gaze_state.fixation_duration += 1 / 30.0  # 30fps 가정
        else:
            self.current_gaze_state.fixation_duration = 0.0
            self.current_gaze_state.current_region = current_region

        # 시선 좌표 업데이트
        self.current_gaze_state.gaze_point = (gaze_data.gaze_x, gaze_data.gaze_y)
        self.current_gaze_state.attention_score = gaze_data.attention_score

        # 사카드 속도 계산 (시선 이동 속도)
        # 실제 구현에서는 이전 프레임과의 차이로 계산
        self.current_gaze_state.saccade_velocity = gaze_data.saccade_velocity

    async def _update_ar_objects(
        self,
        ui_state: UIState,
        vehicle_context: VehicleContext,
        context_info: Dict[str, Any],
        driver_intention: Dict[str, Any],
    ):
        """AR 객체 생성 및 업데이트"""

        # 기존 객체 정리 (만료된 객체 제거)
        await self._cleanup_expired_objects()

        # 1. 안전 경고 객체 생성
        await self._create_safety_warnings(ui_state, context_info)

        # 2. 네비게이션 가이드 생성
        await self._create_navigation_guides(vehicle_context)

        # 3. 차선 변경 지원
        await self._create_lane_change_assistance(driver_intention, vehicle_context)

        # 4. 위험 요소 하이라이팅
        await self._create_hazard_highlighting(context_info)

        # 5. 생체 정보 오버레이
        await self._create_biometric_overlay(ui_state)

    async def _create_safety_warnings(
        self, ui_state: UIState, context_info: Dict[str, Any]
    ):
        """안전 경고 AR 객체 생성"""

        # 주의산만 경고
        if ui_state.gaze.distraction_level > 0.7:
            await self._create_attention_warning()

        # 졸음 경고
        if ui_state.face.drowsiness_level > 0.6:
            await self._create_drowsiness_warning()

        # 위험 물체 경고
        high_risk_objects = context_info.get("high_risk_objects", [])
        for obj in high_risk_objects:
            await self._create_object_warning_box(obj)

    async def _create_attention_warning(self):
        """주의산만 경고 생성"""
        warning_box = ARObject(
            object_id="attention_warning",
            object_type=ARObjectType.WARNING_BOX,
            priority=ARPriority.HIGH,
            world_position=(0.0, 0.0, 5.0),  # 전방 5m
            screen_position=(self.screen_width // 2, self.screen_height // 3),
            size=(400, 100),
            color=(255, 140, 0, 200),  # 주황색
            text="주의력 집중 필요!",
            blink_rate=2.0,  # 2Hz 깜빡임
            fade_duration=3.0,
        )
        self.ar_objects["attention_warning"] = warning_box

    async def _create_drowsiness_warning(self):
        """졸음 경고 생성"""
        warning_box = ARObject(
            object_id="drowsiness_warning",
            object_type=ARObjectType.WARNING_BOX,
            priority=ARPriority.CRITICAL,
            world_position=(0.0, 0.0, 3.0),
            screen_position=(self.screen_width // 2, self.screen_height // 4),
            size=(500, 120),
            color=(255, 0, 0, 220),  # 빨간색
            text="졸음 감지! 휴식 필요",
            blink_rate=3.0,
            fade_duration=5.0,
        )
        self.ar_objects["drowsiness_warning"] = warning_box

    async def _create_object_warning_box(self, risk_object: Dict[str, Any]):
        """위험 객체 경고 박스 생성"""
        obj_id = f"risk_object_{risk_object.get('id', int(time.time()))}"

        # 객체 위치를 화면 좌표로 변환
        screen_x, screen_y = self._world_to_screen(
            risk_object.get("position", (0, 0, 10))
        )

        warning_box = ARObject(
            object_id=obj_id,
            object_type=ARObjectType.WARNING_BOX,
            priority=ARPriority.HIGH,
            world_position=risk_object.get("position", (0, 0, 10)),
            screen_position=(screen_x, screen_y),
            size=(150, 100),
            color=(255, 0, 0, 180),
            text=f"위험: {risk_object.get('type', '물체')}",
            blink_rate=2.5,
            fade_duration=2.0,
        )
        self.ar_objects[obj_id] = warning_box

    async def _create_navigation_guides(self, vehicle_context: VehicleContext):
        """네비게이션 가이드 AR 객체 생성"""
        if not vehicle_context.next_maneuver:
            return

        maneuver = vehicle_context.next_maneuver
        distance = vehicle_context.distance_to_maneuver

        # 네비게이션 화살표 생성
        if maneuver.get("type") == "turn_right":
            await self._create_turn_arrow("right", distance)
        elif maneuver.get("type") == "turn_left":
            await self._create_turn_arrow("left", distance)
        elif maneuver.get("type") == "straight":
            await self._create_straight_arrow(distance)

    async def _create_turn_arrow(self, direction: str, distance: float):
        """회전 화살표 생성"""
        # 실제 도로의 회전 지점에 AR 화살표 표시
        if distance <= 100:  # 100m 이내에서만 표시
            arrow_id = f"nav_arrow_{direction}"

            # 회전 지점 좌표 계산 (실제로는 GPS + 맵 데이터 활용)
            target_position = self._calculate_maneuver_position(direction, distance)
            screen_x, screen_y = self._world_to_screen(target_position)

            arrow = ARObject(
                object_id=arrow_id,
                object_type=ARObjectType.NAVIGATION_ARROW,
                priority=ARPriority.MEDIUM,
                world_position=target_position,
                screen_position=(screen_x, screen_y),
                size=(80, 80),
                color=(0, 255, 255, 200),  # 시안색
                icon=f"arrow_{direction}",
                text=f"{distance:.0f}m",
            )
            self.ar_objects[arrow_id] = arrow

    async def _create_lane_change_assistance(
        self, driver_intention: Dict[str, Any], vehicle_context: VehicleContext
    ):
        """차선 변경 지원 AR 생성"""
        intended_direction = driver_intention.get("lane_change_direction")

        if intended_direction:
            # 목표 차선 하이라이팅
            await self._highlight_target_lane(intended_direction)

            # 사각지대 경고
            if self._check_blind_spot_risk(intended_direction, vehicle_context):
                await self._create_blind_spot_warning(intended_direction)

    async def _highlight_target_lane(self, direction: str):
        """목표 차선 하이라이팅"""
        lane_id = f"target_lane_{direction}"

        # 차선 영역 계산 (실제로는 차선 인식 데이터 활용)
        lane_points = self._calculate_lane_boundary(direction)

        lane_highlight = ARObject(
            object_id=lane_id,
            object_type=ARObjectType.LANE_HIGHLIGHT,
            priority=ARPriority.MEDIUM,
            world_position=(0.0, 0.0, 20.0),  # 전방 20m
            screen_position=(self.screen_width // 2, self.screen_height // 2),
            size=(200, 600),
            color=(0, 255, 0, 100),  # 반투명 녹색
            blink_rate=1.0,
            fade_duration=5.0,
        )
        self.ar_objects[lane_id] = lane_highlight

    async def _create_blind_spot_warning(self, direction: str):
        """사각지대 경고 생성"""
        warning_id = f"blind_spot_{direction}"

        # 사각지대 위치에 경고 표시
        if direction == "left":
            screen_pos = (100, self.screen_height // 2)
        else:
            screen_pos = (self.screen_width - 100, self.screen_height // 2)

        blind_spot_warning = ARObject(
            object_id=warning_id,
            object_type=ARObjectType.BLIND_SPOT_ALERT,
            priority=ARPriority.HIGH,
            world_position=(-3.0 if direction == "left" else 3.0, 0.0, 0.0),
            screen_position=screen_pos,
            size=(120, 120),
            color=(255, 0, 0, 200),
            text="사각지대 위험!",
            blink_rate=3.0,
            fade_duration=3.0,
        )
        self.ar_objects[warning_id] = blind_spot_warning

    async def _create_hazard_highlighting(self, context_info: Dict[str, Any]):
        """위험 요소 하이라이팅"""
        hazards = context_info.get("potential_hazards", [])

        for hazard in hazards:
            hazard_id = f"hazard_{hazard.get('id', int(time.time()))}"

            screen_x, screen_y = self._world_to_screen(
                hazard.get("position", (0, 0, 15))
            )

            hazard_marker = ARObject(
                object_id=hazard_id,
                object_type=ARObjectType.HAZARD_MARKER,
                priority=ARPriority.MEDIUM,
                world_position=hazard.get("position", (0, 0, 15)),
                screen_position=(screen_x, screen_y),
                size=(60, 60),
                color=(255, 255, 0, 160),  # 노란색
                icon="warning",
                text=hazard.get("type", "주의"),
                fade_duration=4.0,
            )
            self.ar_objects[hazard_id] = hazard_marker

    async def _create_biometric_overlay(self, ui_state: UIState):
        """생체 정보 오버레이 생성"""
        # 심박수 표시 (옵션)
        if ui_state.biometrics.heart_rate and ui_state.biometrics.heart_rate > 0:
            hr_overlay = ARObject(
                object_id="heart_rate_overlay",
                object_type=ARObjectType.BIOMETRIC_OVERLAY,
                priority=ARPriority.INFO,
                world_position=(0.0, 0.0, 0.0),
                screen_position=(self.screen_width - 150, 50),
                size=(120, 30),
                color=(255, 255, 255, 150),
                text=f"♥ {ui_state.biometrics.heart_rate:.0f} BPM",
            )
            self.ar_objects["heart_rate_overlay"] = hr_overlay

    async def _adapt_display_to_gaze(self):
        """시선 기반 적응형 표시"""
        if not self.enable_predictive_highlighting:
            return

        gaze_region = self.current_gaze_state.current_region
        fixation_time = self.current_gaze_state.fixation_duration

        # 시선 영역별 객체 밝기 조정
        for obj_id, ar_obj in self.ar_objects.items():
            # 시선이 오래 고정된 영역의 객체는 밝기 감소
            if self._is_object_in_gaze_region(ar_obj, gaze_region):
                if fixation_time > 2.0:  # 2초 이상 고정
                    # 밝기 감소 (주의산만 방지)
                    r, g, b, a = ar_obj.color
                    ar_obj.color = (r, g, b, max(50, a - 30))
            else:
                # 시선이 없는 영역의 중요 객체는 밝기 증가
                if ar_obj.priority in [ARPriority.CRITICAL, ARPriority.HIGH]:
                    r, g, b, a = ar_obj.color
                    ar_obj.color = (r, g, b, min(255, a + 20))

    def _is_object_in_gaze_region(
        self, ar_obj: ARObject, gaze_region: GazeRegion
    ) -> bool:
        """AR 객체가 시선 영역에 있는지 확인"""
        obj_x, obj_y = ar_obj.screen_position

        # 간단한 영역 판정 (실제로는 더 정교한 계산 필요)
        if gaze_region == GazeRegion.CENTER:
            return (
                self.screen_width // 4 <= obj_x <= 3 * self.screen_width // 4
                and self.screen_height // 4 <= obj_y <= 3 * self.screen_height // 4
            )
        elif gaze_region == GazeRegion.LEFT_MIRROR:
            return obj_x <= self.screen_width // 4
        elif gaze_region == GazeRegion.RIGHT_MIRROR:
            return obj_x >= 3 * self.screen_width // 4

        return False

    async def _cleanup_expired_objects(self):
        """만료된 AR 객체 정리"""
        current_time = time.time()
        expired_objects = []

        for obj_id, ar_obj in self.ar_objects.items():
            # 페이드 지속시간이 있는 객체 체크
            if (
                ar_obj.fade_duration > 0
                and current_time - ar_obj.created_at > ar_obj.fade_duration
            ):
                expired_objects.append(obj_id)

        for obj_id in expired_objects:
            del self.ar_objects[obj_id]

    def _world_to_screen(
        self, world_pos: Tuple[float, float, float]
    ) -> Tuple[int, int]:
        """월드 좌표를 화면 좌표로 변환"""
        x, y, z = world_pos

        # 간단한 투영 변환 (실제로는 카메라 캘리브레이션 필요)
        if z > 0:
            # 원근 투영
            focal_length = 800  # 가상 초점거리
            screen_x = int(self.screen_width // 2 + (x * focal_length / z))
            screen_y = int(self.screen_height // 2 - (y * focal_length / z))
        else:
            screen_x = self.screen_width // 2
            screen_y = self.screen_height // 2

        # 화면 범위 클램핑
        screen_x = max(0, min(self.screen_width, screen_x))
        screen_y = max(0, min(self.screen_height, screen_y))

        return screen_x, screen_y

    def _calculate_maneuver_position(
        self, direction: str, distance: float
    ) -> Tuple[float, float, float]:
        """조작 위치 계산"""
        # 간단한 계산 (실제로는 GPS + HD맵 데이터 활용)
        if direction == "left":
            return (-2.0, 0.0, distance)
        elif direction == "right":
            return (2.0, 0.0, distance)
        else:
            return (0.0, 0.0, distance)

    def _calculate_lane_boundary(self, direction: str) -> List[Tuple[float, float]]:
        """차선 경계 계산"""
        # 차선 인식 데이터를 활용한 실제 차선 좌표 계산
        # 여기서는 간단한 예시
        if direction == "left":
            return [(-2.0, 0.0), (-2.0, 50.0), (-5.0, 50.0), (-5.0, 0.0)]
        else:
            return [(2.0, 0.0), (2.0, 50.0), (5.0, 50.0), (5.0, 0.0)]

    def _check_blind_spot_risk(
        self, direction: str, vehicle_context: VehicleContext
    ) -> bool:
        """사각지대 위험 체크"""
        # 실제로는 레이더/카메라 센서 데이터 활용
        detected_objects = vehicle_context.detected_objects

        for obj in detected_objects:
            obj_pos = obj.get("position", (0, 0, 0))
            obj_x = obj_pos[0]

            if direction == "left" and -5.0 <= obj_x <= -1.0:
                return True
            elif direction == "right" and 1.0 <= obj_x <= 5.0:
                return True

        return False

    def get_ar_statistics(self) -> Dict[str, Any]:
        """AR 시스템 통계"""
        stats = {
            "active_objects": len(self.ar_objects),
            "objects_by_type": {},
            "objects_by_priority": {},
            "current_gaze_region": self.current_gaze_state.current_region.value,
            "gaze_fixation_time": self.current_gaze_state.fixation_duration,
            "adaptive_features_enabled": self.enable_predictive_highlighting,
        }

        # 타입별 통계
        for ar_obj in self.ar_objects.values():
            obj_type = ar_obj.object_type.value
            priority = ar_obj.priority.value

            stats["objects_by_type"][obj_type] = (
                stats["objects_by_type"].get(obj_type, 0) + 1
            )
            stats["objects_by_priority"][priority] = (
                stats["objects_by_priority"].get(priority, 0) + 1
            )

        return stats


class GazeRegionTracker:
    """시선 영역 추적기"""

    def __init__(self):
        self.region_boundaries = self._initialize_region_boundaries()

    def _initialize_region_boundaries(self) -> Dict[GazeRegion, Dict[str, float]]:
        """시선 영역 경계 설정"""
        return {
            GazeRegion.CENTER: {
                "x_min": 0.25,
                "x_max": 0.75,
                "y_min": 0.25,
                "y_max": 0.75,
            },
            GazeRegion.LEFT_MIRROR: {
                "x_min": 0.0,
                "x_max": 0.2,
                "y_min": 0.3,
                "y_max": 0.6,
            },
            GazeRegion.RIGHT_MIRROR: {
                "x_min": 0.8,
                "x_max": 1.0,
                "y_min": 0.3,
                "y_max": 0.6,
            },
            GazeRegion.REAR_MIRROR: {
                "x_min": 0.4,
                "x_max": 0.6,
                "y_min": 0.0,
                "y_max": 0.2,
            },
            GazeRegion.DASHBOARD: {
                "x_min": 0.3,
                "x_max": 0.7,
                "y_min": 0.8,
                "y_max": 1.0,
            },
        }

    def determine_gaze_region(self, gaze_x: float, gaze_y: float) -> GazeRegion:
        """시선 좌표로부터 영역 판정"""
        for region, bounds in self.region_boundaries.items():
            if (
                bounds["x_min"] <= gaze_x <= bounds["x_max"]
                and bounds["y_min"] <= gaze_y <= bounds["y_max"]
            ):
                return region

        return GazeRegion.CENTER  # 기본값


class ContextAnalyzer:
    """상황 분석 엔진"""

    async def analyze_situation(
        self, vehicle_context: VehicleContext, gaze_state: DriverGazeState
    ) -> Dict[str, Any]:
        """상황 분석"""
        analysis = {
            "risk_level": "low",
            "high_risk_objects": [],
            "potential_hazards": [],
            "driver_attention_state": "normal",
            "environmental_factors": {},
        }

        # 속도 기반 위험도 분석
        if vehicle_context.speed_kmh > 80:
            analysis["risk_level"] = "high"
        elif vehicle_context.speed_kmh > 50:
            analysis["risk_level"] = "medium"

        # 주의력 상태 분석
        if gaze_state.attention_score < 0.5:
            analysis["driver_attention_state"] = "distracted"
        elif gaze_state.fixation_duration > 3.0:
            analysis["driver_attention_state"] = "tunnel_vision"

        # 감지된 객체 위험도 평가
        for obj in vehicle_context.detected_objects:
            risk_score = self._calculate_object_risk(obj, vehicle_context)
            if risk_score > 0.7:
                analysis["high_risk_objects"].append({**obj, "risk_score": risk_score})

        return analysis

    def _calculate_object_risk(
        self, obj: Dict[str, Any], context: VehicleContext
    ) -> float:
        """객체 위험도 계산"""
        base_risk = 0.0

        # 객체 타입별 기본 위험도
        obj_type = obj.get("type", "unknown")
        if obj_type == "pedestrian":
            base_risk = 0.8
        elif obj_type == "vehicle":
            base_risk = 0.6
        elif obj_type == "cyclist":
            base_risk = 0.7

        # 거리 기반 위험도 조정
        distance = obj.get("distance", 100)
        if distance < 10:
            base_risk *= 1.5
        elif distance < 20:
            base_risk *= 1.2

        # 속도 기반 조정
        if context.speed_kmh > 60:
            base_risk *= 1.3

        return min(1.0, base_risk)


class IntentionPredictor:
    """운전자 의도 예측기"""

    async def predict_intention(
        self,
        gaze_state: DriverGazeState,
        vehicle_context: VehicleContext,
        context_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """운전자 의도 예측"""
        intention = {
            "lane_change_direction": None,
            "lane_change_probability": 0.0,
            "turn_intention": None,
            "stopping_probability": 0.0,
            "confidence": 0.0,
        }

        # 차선 변경 의도 예측
        if vehicle_context.turn_signal:
            intention["lane_change_direction"] = vehicle_context.turn_signal
            intention["lane_change_probability"] = 0.9
            intention["confidence"] = 0.9

        # 시선 패턴 기반 예측
        if gaze_state.current_region == GazeRegion.LEFT_MIRROR:
            if gaze_state.fixation_duration > 1.0:
                intention["lane_change_direction"] = "left"
                intention["lane_change_probability"] = 0.7
                intention["confidence"] = 0.7

        elif gaze_state.current_region == GazeRegion.RIGHT_MIRROR:
            if gaze_state.fixation_duration > 1.0:
                intention["lane_change_direction"] = "right"
                intention["lane_change_probability"] = 0.7
                intention["confidence"] = 0.7

        # 핸들 조작 패턴 고려
        if abs(vehicle_context.steering_angle) > 5:
            if vehicle_context.steering_angle > 0:
                intention["turn_intention"] = "right"
            else:
                intention["turn_intention"] = "left"

        return intention


class ARRenderer:
    """AR 렌더링 엔진 - 메모리 최적화"""

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height

        # 프레임 버퍼 재사용을 위한 영구 할당
        self.frame_buffer = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
        self._buffer_initialized = True

        # 성능 추적
        self._render_count = 0
        self._last_cleanup_time = time.time()
        self._cleanup_interval = 300.0  # 5분마다 정리

    async def render_frame(
        self,
        ar_objects: Dict[str, ARObject],
        gaze_state: DriverGazeState,
        adaptive_brightness: bool = True,
    ) -> np.ndarray:
        """AR 프레임 렌더링 - 메모리 최적화"""
        self._render_count += 1

        # 프레임 버퍼 초기화 (재할당 없이 제로화)
        self.frame_buffer.fill(0)

        # 우선순위 순으로 객체 렌더링
        sorted_objects = sorted(ar_objects.values(), key=lambda obj: obj.priority.value)

        for ar_obj in sorted_objects:
            if ar_obj.is_visible:
                await self._render_object(ar_obj, adaptive_brightness)

        # 시선 위치 표시 (디버그용, 실제로는 비활성화)
        if False:  # 디버그 모드에서만
            await self._render_gaze_indicator(gaze_state)

        # 주기적 메모리 정리
        current_time = time.time()
        if current_time - self._last_cleanup_time > self._cleanup_interval:
            self._perform_memory_cleanup()
            self._last_cleanup_time = current_time

        return self.frame_buffer

    def _perform_memory_cleanup(self):
        """주기적 메모리 정리"""
        try:
            # 프레임 버퍼 크기 확인 및 필요시 재할당
            expected_size = (self.screen_height, self.screen_width, 4)
            if self.frame_buffer.shape != expected_size:
                logger.warning(
                    f"프레임 버퍼 크기 불일치 감지: {self.frame_buffer.shape} != {expected_size}"
                )
                self.frame_buffer = np.zeros(expected_size, dtype=np.uint8)
                logger.info("프레임 버퍼 재할당 완료")

            # 가비지 컬렉션 강제 실행 (주의: 성능에 영향)
            import gc

            collected = gc.collect()

            logger.debug(
                f"AR 렌더러 메모리 정리 완료 - 렌더링 횟수: {self._render_count}, "
                f"수집된 객체: {collected}개"
            )

        except Exception as e:
            logger.error(f"AR 렌더러 메모리 정리 중 오류: {e}")

    def resize_buffer(self, new_width: int, new_height: int):
        """프레임 버퍼 크기 변경 - 메모리 효율적"""
        if new_width != self.screen_width or new_height != self.screen_height:
            logger.info(
                f"AR 렌더러 버퍼 크기 변경: {self.screen_width}x{self.screen_height} -> {new_width}x{new_height}"
            )

            self.screen_width = new_width
            self.screen_height = new_height

            # 새로운 크기로 버퍼 재할당
            self.frame_buffer = np.zeros((new_height, new_width, 4), dtype=np.uint8)

            logger.info("AR 렌더러 버퍼 크기 변경 완료")

    async def _render_object(self, ar_obj: ARObject, adaptive_brightness: bool):
        """개별 AR 객체 렌더링"""
        x, y = ar_obj.screen_position
        w, h = ar_obj.size

        # 화면 경계 검사
        if x < 0 or y < 0 or x >= self.screen_width or y >= self.screen_height:
            return  # 화면 밖 객체는 렌더링 생략

        # 깜빡임 처리
        if ar_obj.blink_rate > 0:
            blink_phase = (time.time() * ar_obj.blink_rate) % 1.0
            if blink_phase > 0.5:
                return  # 깜빡임 상태에서는 렌더링 안함

        # 색상 설정
        color = ar_obj.color
        if adaptive_brightness:
            # 환경 밝기에 따른 적응형 색상 조정
            # 실제로는 주변 조도 센서 데이터 활용
            color = self._adjust_color_for_brightness(color)

        # 객체 타입별 렌더링
        if ar_obj.object_type == ARObjectType.WARNING_BOX:
            await self._draw_warning_box(x, y, w, h, color, ar_obj.text)

        elif ar_obj.object_type == ARObjectType.NAVIGATION_ARROW:
            await self._draw_navigation_arrow(x, y, w, h, color, ar_obj.icon)

        elif ar_obj.object_type == ARObjectType.LANE_HIGHLIGHT:
            await self._draw_lane_highlight(x, y, w, h, color)

        elif ar_obj.object_type == ARObjectType.HAZARD_MARKER:
            await self._draw_hazard_marker(x, y, w, h, color, ar_obj.text)

        elif ar_obj.object_type == ARObjectType.BIOMETRIC_OVERLAY:
            await self._draw_text_overlay(x, y, color, ar_obj.text)

    async def _draw_warning_box(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        color: Tuple[int, int, int, int],
        text: str,
    ):
        """경고 박스 그리기"""
        # 박스 테두리
        cv2.rectangle(
            self.frame_buffer,
            (x - w // 2, y - h // 2),
            (x + w // 2, y + h // 2),
            color,
            3,
        )

        # 텍스트
        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = x - text_size[0] // 2
            text_y = y + text_size[1] // 2

            cv2.putText(
                self.frame_buffer,
                text,
                (text_x, text_y),
                font,
                font_scale,
                color[:3],
                thickness,
            )

    async def _draw_navigation_arrow(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        color: Tuple[int, int, int, int],
        icon: str,
    ):
        """네비게이션 화살표 그리기"""
        # 화살표 포인트 계산
        if "right" in icon:
            points = np.array(
                [[x - w // 4, y - h // 4], [x + w // 4, y], [x - w // 4, y + h // 4]],
                np.int32,
            )
        elif "left" in icon:
            points = np.array(
                [[x + w // 4, y - h // 4], [x - w // 4, y], [x + w // 4, y + h // 4]],
                np.int32,
            )
        else:  # 직진
            points = np.array(
                [[x - w // 4, y + h // 4], [x, y - h // 4], [x + w // 4, y + h // 4]],
                np.int32,
            )

        cv2.fillPoly(self.frame_buffer, [points], color[:3])

    async def _draw_lane_highlight(
        self, x: int, y: int, w: int, h: int, color: Tuple[int, int, int, int]
    ):
        """차선 하이라이트 그리기 - 메모리 최적화"""
        # 임시 오버레이를 위한 작은 영역만 복사
        x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
        x2, y2 = min(self.screen_width, x + w // 2), min(self.screen_height, y + h // 2)

        if x2 <= x1 or y2 <= y1:
            return  # 유효하지 않은 영역

        # 작은 영역만 처리하여 메모리 사용량 감소
        region = self.frame_buffer[y1:y2, x1:x2].copy()
        cv2.rectangle(region, (0, 0), (x2 - x1, y2 - y1), color[:3], -1)

        alpha = color[3] / 255.0
        cv2.addWeighted(
            region,
            alpha,
            self.frame_buffer[y1:y2, x1:x2],
            1 - alpha,
            0,
            self.frame_buffer[y1:y2, x1:x2],
        )

    async def _draw_hazard_marker(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        color: Tuple[int, int, int, int],
        text: str,
    ):
        """위험 마커 그리기"""
        # 삼각형 경고 표시
        points = np.array(
            [[x, y - h // 2], [x - w // 2, y + h // 2], [x + w // 2, y + h // 2]],
            np.int32,
        )

        cv2.fillPoly(self.frame_buffer, [points], color[:3])

        # 느낌표
        cv2.putText(
            self.frame_buffer,
            "!",
            (x - 5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
        )

    async def _draw_text_overlay(
        self, x: int, y: int, color: Tuple[int, int, int, int], text: str
    ):
        """텍스트 오버레이 그리기"""
        if text:
            cv2.putText(
                self.frame_buffer,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color[:3],
                1,
            )

    async def _render_gaze_indicator(self, gaze_state: DriverGazeState):
        """시선 위치 표시 (디버그용)"""
        gaze_x = int(gaze_state.gaze_point[0] * self.screen_width)
        gaze_y = int(gaze_state.gaze_point[1] * self.screen_height)

        cv2.circle(self.frame_buffer, (gaze_x, gaze_y), 10, (255, 255, 255, 128), 2)

    def _adjust_color_for_brightness(
        self, color: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """밝기에 따른 색상 조정"""
        # 주간/야간 모드에 따른 색상 적응
        # 실제로는 주변 조도 센서 데이터 활용
        r, g, b, a = color

        # 간단한 야간 모드 시뮬레이션 (시간 기반)
        current_hour = time.localtime().tm_hour
        if 18 <= current_hour or current_hour <= 6:  # 야간
            # 밝기 증가
            r = min(255, int(r * 1.3))
            g = min(255, int(g * 1.3))
            b = min(255, int(b * 1.3))

        return (r, g, b, a)

    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 사용량 통계"""
        buffer_size_mb = self.frame_buffer.nbytes / (1024 * 1024)

        return {
            "frame_buffer_size_mb": buffer_size_mb,
            "buffer_shape": self.frame_buffer.shape,
            "render_count": self._render_count,
            "last_cleanup": self._last_cleanup_time,
            "cleanup_interval": self._cleanup_interval,
        }
