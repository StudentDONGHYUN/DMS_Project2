"""
손 데이터 전문 처리기 (S-Class, 인터페이스 통합 최종 버전)

[파일 설명]
이 클래스는 IHandDataProcessor 인터페이스를 구현하며, 시스템의 다른 모듈과
상호작용하기 위해 core.constants, core.definitions, config.settings의
정의를 사용합니다. 운동역학, FFT, 제스처 시퀀스 분석 등 S-Class급 기술을 통해
운전자의 핸들링 스킬, 주의산만 행동, 피로도를 심층적으로 분석합니다.

[업데이트 내역]
- IHandDataProcessor, IMetricsUpdater 인터페이스 구현
- core.constants의 상수 값 (차량 구역, 수학 상수 등) 적용
- config.settings의 설정 값 (버퍼 크기, 임계값 등) 참조
- core.definitions의 데이터 구조 (AdvancedMetrics, GazeZone 등)에 맞게 출력 조정
- 하드코딩된 값 제거 및 중앙 설정 관리 시스템과 연동
"""

import math
import logging
from collections import deque
from typing import Dict, Any, List, Optional

import numpy as np
from scipy.fft import rfft, rfftfreq

# --- 시스템 핵심 모듈 임포트 ---
from core.interfaces import IHandDataProcessor, IMetricsUpdater
from core.constants import VehicleConstants, MathConstants, AnalysisConstants
from core.definitions import GazeZone # 핸드 프로세서에서는 직접 사용하지 않지만, metrics 구조 이해를 위해 참고
from config.settings import get_config, SystemConfig, HandConfig # HandConfig는 추가가 필요

logger = logging.getLogger(__name__)


class HandDataProcessor(IHandDataProcessor):
    """
    손 데이터 전문 처리기 (S-Class)
    IHandDataProcessor 인터페이스를 구현합니다.
    """

    def __init__(self, metrics_updater: IMetricsUpdater):
        self.metrics_updater = metrics_updater
        self.config: SystemConfig = get_config()

        # --- 상태 추적 변수 ---
        # 이제 HandConfig가 SystemConfig에 정상적으로 통합되었습니다.
        hand_config = self.config.hand
        
        self.hand_kinematics_history = {
            'Left': deque(maxlen=hand_config.fft_buffer_size),
            'Right': deque(maxlen=hand_config.fft_buffer_size)
        }
        self.gesture_sequence_buffer = deque(maxlen=hand_config.gesture_buffer_size)

        # --- 상수 및 설정 적용 ---
        self.vehicle_zones = VehicleConstants.Zones # constants.py에서 차량 구역 정의 사용
        
        # 핸들 위치 및 반경 계산 (하드코딩 제거)
        wheel_zone = self.vehicle_zones.STEERING_WHEEL
        self.steering_wheel_center = (
            (wheel_zone['x1'] + wheel_zone['x2']) / 2,
            (wheel_zone['y1'] + wheel_zone['y2']) / 2
        )
        # 핸들 영역을 원으로 근사화하기 위한 반지름 계산
        self.steering_wheel_radius_sq = ((wheel_zone['x2'] - wheel_zone['x1']) / 2) ** 2

        logger.info("HandDataProcessor (S-Class, 인터페이스 통합) 초기화 완료")

    def get_processor_name(self) -> str:
        """프로세서의 이름을 반환합니다."""
        return "HandDataProcessor_S_Class"

    def get_required_data_types(self) -> List[str]:
        """이 프로세서가 필요로 하는 데이터 타입 목록을 반환합니다."""
        return ["hand_landmarks", "handedness"]

    async def process_data(self, result, timestamp):
        logger.debug(f"[hand_processor_s_class] process_data input: {result}")
        if hasattr(result, 'hand_landmarks'):
            logger.debug(f"[hand_processor_s_class] hand_landmarks: {getattr(result, 'hand_landmarks', None)}")
        if not result or not hasattr(result, 'hand_landmarks') or not result.hand_landmarks:
            return await self._handle_no_hands_detected()

        # 인터페이스의 process_hand_landmarks를 내부적으로 호출하는 구조
        processed_hands = await self.process_hand_landmarks(result, timestamp)
        comprehensive_analysis = await self._perform_comprehensive_hand_analysis(processed_hands, timestamp)

        results = {'hand_positions': processed_hands, 'hand_analysis': comprehensive_analysis}
        self._update_hand_metrics(comprehensive_analysis)

        return results

    async def process_hand_landmarks(self, hand_results: Any, timestamp: float) -> List[Dict[str, Any]]:
        """
        손 랜드마크를 처리하여 상세 정보 리스트를 반환합니다.
        IHandDataProcessor 인터페이스의 추상 메서드를 구현합니다.
        """
        processed_hands = []
        for i, landmarks in enumerate(hand_results.hand_landmarks):
            handedness = hand_results.handedness[i][0].category_name

            kinematics = self._analyze_hand_kinematics(landmarks, handedness, timestamp)
            tremor_analysis = self._analyze_tremor_frequency(handedness)
            zone_analysis = self._analyze_hand_zone(landmarks[0]) # 손목 기준
            grip_analysis = self._analyze_grip_type_and_quality(landmarks)
            gesture = self._infer_hand_gesture(kinematics, zone_analysis, grip_analysis)

            hand_data = {
                'handedness': handedness,
                'confidence': hand_results.handedness[i][0].score,
                'landmarks': landmarks,
                'kinematics': kinematics,
                'tremor_analysis': tremor_analysis,
                'grip_analysis': grip_analysis,
                'zone_analysis': zone_analysis,
                'gesture': gesture,
                'timestamp': timestamp
            }
            processed_hands.append(hand_data)

            self.gesture_sequence_buffer.append({
                'timestamp': timestamp, 'handedness': handedness,
                'gesture': gesture, 'zone': zone_analysis['primary_zone']
            })

        return processed_hands

    async def _perform_comprehensive_hand_analysis(self, hands: List[Dict], timestamp: float) -> Dict[str, Any]:
        """모든 손의 정보를 통합하여 종합적인 분석을 수행합니다."""
        if not hands:
            return await self._get_default_hand_analysis()

        steering_skill = self._analyze_steering_skill(hands)
        distraction = self._detect_distraction_behaviors(hands)
        gesture_patterns = self._analyze_gesture_patterns()
        driving_technique = self._evaluate_driving_technique(steering_skill, distraction)

        return {
            'hands_detected_count': len(hands),
            'steering_skill': steering_skill,
            'distraction_behaviors': distraction,
            'gesture_patterns': gesture_patterns,
            'driving_technique': driving_technique,
            'overall_hand_safety': self._calculate_overall_safety_score(steering_skill, distraction)
        }

    def _analyze_hand_kinematics(self, landmarks: Any, handedness: str, timestamp: float) -> Dict[str, Any]:
        """[S-Class] 손의 운동학적 특성 분석 (속도, 가속도, 저크)"""
        history = self.hand_kinematics_history[handedness]
        wrist_pos = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])

        velocity, acceleration, jerk = np.zeros(3), np.zeros(3), np.zeros(3)

        if len(history) > 2:
            t_prev, p_prev, v_prev, a_prev = history[-1]
            t_pre_prev, p_pre_prev, _, _ = history[-2]

            dt1 = timestamp - t_prev
            dt2 = t_prev - t_pre_prev

            if dt1 > MathConstants.EPSILON:
                velocity = (wrist_pos - p_prev) / dt1
                if dt2 > MathConstants.EPSILON:
                    prev_velocity = (p_prev - p_pre_prev) / dt2
                    acceleration = (velocity - prev_velocity) / dt1
                    jerk = (acceleration - a_prev) / dt1
        
        # 기록 저장: timestamp, 위치, 속도, 가속도
        history.append((timestamp, wrist_pos, velocity, acceleration))
        
        jerk_magnitude = np.linalg.norm(jerk)
        # 이제 HandConfig에서 정의된 저크 한계값을 사용합니다
        jerk_limit = self.config.hand.jerk_limit
        smoothness = max(0.0, 1.0 - min(jerk_magnitude / jerk_limit, 1.0))

        return {
            'velocity_magnitude': np.linalg.norm(velocity),
            'acceleration_magnitude': np.linalg.norm(acceleration),
            'jerk_magnitude': jerk_magnitude,
            'smoothness_score': smoothness
        }

    def _analyze_tremor_frequency(self, handedness: str) -> Dict[str, Any]:
        """[S-Class] FFT 기반 손 떨림 주파수 분석"""
        hand_config = self.config.hand
        history = self.hand_kinematics_history[handedness]
        if len(history) < hand_config.fft_min_samples:
            return {'dominant_frequency_hz': 0.0, 'fatigue_tremor_power': 0.0, 'tremor_severity': 'none'}

        y_positions = np.array([h[1][1] for h in history])
        timestamps = np.array([h[0] for h in history])

        time_delta = timestamps[-1] - timestamps[0]
        if len(timestamps) < 2 or time_delta < MathConstants.EPSILON:
            return {'dominant_frequency_hz': 0.0, 'fatigue_tremor_power': 0.0, 'tremor_severity': 'none'}

        sampling_rate = len(timestamps) / time_delta

        try:
            yf = rfft(y_positions - np.mean(y_positions))
            xf = rfftfreq(len(y_positions), 1 / sampling_rate)
            power_spectrum = np.abs(yf)**2
            dominant_freq = xf[np.argmax(power_spectrum)] if len(power_spectrum) > 0 else 0.0

            # 피로 관련 주파수 대역(8-12Hz)의 파워 계산
            fatigue_band = (xf >= 8) & (xf <= 12)
            total_power = np.sum(power_spectrum)
            fatigue_power = np.sum(power_spectrum[fatigue_band]) / total_power if total_power > MathConstants.EPSILON else 0.0

            severity = 'none'
            if fatigue_power > 0.4: severity = 'severe'
            elif fatigue_power > 0.2: severity = 'moderate'
            elif fatigue_power > 0.1: severity = 'mild'

            return {'dominant_frequency_hz': dominant_freq, 'fatigue_tremor_power': fatigue_power, 'tremor_severity': severity}
        except Exception as e:
            logger.error(f"FFT 떨림 분석 중 오류: {e}")
            return {'dominant_frequency_hz': 0.0, 'fatigue_tremor_power': 0.0, 'tremor_severity': 'unknown'}

    def _analyze_grip_type_and_quality(self, landmarks: Any) -> Dict[str, Any]:
        """[S-Class] 그립 유형 분류 및 품질 분석"""
        angles = self._calculate_finger_curl_angles(landmarks)
        avg_curl = np.mean(list(angles.values())) if angles else 0.0

        grip_type = 'open_hand'
        if avg_curl > 130:
            grip_type = 'power_grip'  # 주먹 쥠 (핸들 그립)
        elif avg_curl > 60:
            thumb_tip, index_tip = landmarks[4], landmarks[8]
            # 2D 좌표로 거리 계산
            dist_vec = np.array([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y])
            precision_dist = np.linalg.norm(dist_vec)
            grip_type = 'precision_grip' if precision_dist < 0.05 else 'general_grip'

        grip_quality = max(0.0, min(1.0, avg_curl / 150.0))
        return {'grip_type': grip_type, 'grip_quality': grip_quality, 'avg_curl': avg_curl}

    def _calculate_finger_curl_angles(self, landmarks: Any) -> Dict[str, float]:
        """손가락별 구부림 각도 계산"""
        try:
            angles = {}
            # 각 손가락의 루트, 중간, 끝 관절 인덱스
            finger_indices = {
                'thumb': [2, 3, 4], 'index': [5, 6, 8], 'middle': [9, 10, 12],
                'ring': [13, 14, 16], 'pinky': [17, 18, 20]
            }
            for name, indices in finger_indices.items():
                p_root, p_mid, p_tip = landmarks[indices[0]], landmarks[indices[1]], landmarks[indices[2]]
                v1 = np.array([p_root.x - p_mid.x, p_root.y - p_mid.y])
                v2 = np.array([p_tip.x - p_mid.x, p_tip.y - p_mid.y])
                angle = self._calculate_angle_between_vectors(v1, v2)
                angles[name] = angle
            return angles
        except Exception as e:
            logger.error(f"손가락 각도 계산 중 오류: {e}")
            return {name: 90.0 for name in ['thumb', 'index', 'middle', 'ring', 'pinky']}

    def _calculate_angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """두 2D 벡터 사이의 각도를 계산합니다."""
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 < MathConstants.EPSILON or norm_v2 < MathConstants.EPSILON:
            return 180.0
        
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
        return angle

    def _analyze_steering_skill(self, hands: List[Dict]) -> Dict[str, Any]:
        """[S-Class] 핸들링 스킬 종합 평가"""
        hands_on_wheel = [h for h in hands if h['zone_analysis']['primary_zone'] == 'STEERING_WHEEL']
        if not hands_on_wheel:
            return {'skill_score': 0.0, 'feedback': 'Hands not on steering wheel', 'components': {}}

        clock_positions = [self._get_hand_clock_position(h['landmarks']) for h in hands_on_wheel]
        position_score = self._evaluate_clock_positions(clock_positions)
        smoothness_score = np.mean([h['kinematics']['smoothness_score'] for h in hands_on_wheel])
        stability_score = np.mean([1.0 - h['tremor_analysis']['fatigue_tremor_power'] for h in hands_on_wheel])
        grip_quality_score = np.mean([h['grip_analysis']['grip_quality'] for h in hands_on_wheel])

        skill_score = (position_score * 0.4 + (smoothness_score + stability_score) / 2 * 0.3) * grip_quality_score + (smoothness_score + stability_score) / 2 * 0.3
        feedback = self._generate_steering_feedback(skill_score, clock_positions, len(hands_on_wheel))
        
        return {'skill_score': skill_score, 'feedback': feedback, 'components': {
            'position_score': position_score, 'movement_smoothness': smoothness_score,
            'hand_stability': stability_score, 'grip_quality': grip_quality_score
        }}

    def _detect_distraction_behaviors(self, hands: List[Dict]) -> Dict[str, Any]:
        """주의산만 행동 감지"""
        risk_score, behaviors, phone_detected = 0.0, [], False
        
        # settings.py의 DistractionConfig에서 휴대폰 위험도 정보 가져오기
        phone_risk_info = self.config.distraction.object_risk_levels.get("cell phone", {"risk_level": 0.9})

        for hand in hands:
            if hand['grip_analysis']['grip_type'] == 'precision_grip' and hand['zone_analysis']['primary_zone'] != 'STEERING_WHEEL':
                risk_score = max(risk_score, phone_risk_info["risk_level"])
                behaviors.append(f"{hand['handedness']} 손, 정밀 그립으로 다른 물체 조작 의심")
                phone_detected = True # 정밀 그립은 휴대폰 사용으로 간주
            
            zone_risk = self._get_zone_risk(hand['zone_analysis']['primary_zone'])
            if zone_risk > AnalysisConstants.Thresholds.RISK_HIGH: # constants.py 사용
                risk_score = max(risk_score, zone_risk)
                behaviors.append(f"{hand['handedness']} 손, 위험 구역({hand['zone_analysis']['primary_zone']})에 위치")

        return {'risk_score': risk_score, 'behaviors': list(set(behaviors)), 'phone_detected': phone_detected}

    def _evaluate_driving_technique(self, steering_skill: Dict, distraction: Dict) -> Dict[str, Any]:
        """운전 기술 종합 평가"""
        base_score = steering_skill.get('skill_score', 0.0)
        distraction_penalty = distraction.get('risk_score', 0.0)
        
        final_score = base_score * (1.0 - distraction_penalty)
        
        rating = 'needs_improvement'
        if final_score > 0.8: rating = 'expert'
        elif final_score > 0.6: rating = 'proficient'
        elif final_score > 0.4: rating = 'adequate'
        
        return {'technique_rating': rating, 'score': max(0, final_score)}
    
    def _analyze_gesture_patterns(self) -> Dict[str, Any]:
        """[S-Class] 제스처 시퀀스 분석"""
        if len(self.gesture_sequence_buffer) < 5:
            return {'pattern_detected': False, 'dominant_pattern': 'insufficient_data'}

        recent_events = list(self.gesture_sequence_buffer)[-15:]
        gesture_counts = {}
        for event in recent_events:
            gesture = event['gesture']
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1

        dominant_gesture = 'unknown'
        if gesture_counts:
            dominant_gesture = max(gesture_counts, key=gesture_counts.get)

        return {'pattern_detected': True, 'dominant_pattern': dominant_gesture, 'gesture_frequency': gesture_counts}

    # --- 헬퍼 및 유틸리티 함수 (상수 및 설정 적용) ---

    def _analyze_hand_zone(self, wrist_landmark: Any) -> Dict[str, Any]:
        """손의 차량 내 위치 분석 (constants.py 기반)"""
        x, y = wrist_landmark.x, wrist_landmark.y
        for zone_name, bounds in self.vehicle_zones.items():
            if bounds["x1"] <= x <= bounds["x2"] and bounds["y1"] <= y <= bounds["y2"]: #
                return {'primary_zone': zone_name, 'risk_level': self._get_zone_risk(zone_name)}
        return {'primary_zone': 'OUT_OF_BOUNDS', 'risk_level': 1.0}

    def _get_zone_risk(self, zone_name: str) -> float:
        """구역별 위험도를 반환합니다."""
        if zone_name == 'STEERING_WHEEL':
            return 0.1
        # 다른 구역들은 주의산만 위험이 높다고 가정
        return 0.8

    def _get_hand_clock_position(self, landmarks: Any) -> Optional[int]:
        """핸들 위 손의 시계 방향 위치 추정 (하드코딩 제거)"""
        center_x, center_y = self.steering_wheel_center
        wrist = landmarks[0]
        dx, dy = wrist.x - center_x, center_y - wrist.y
        
        if dx**2 + dy**2 < self.steering_wheel_radius_sq:
            angle = math.degrees(math.atan2(dy, dx))
            clock_angle = (angle + 90) % 360
            clock_pos = round(clock_angle / 30)
            return 12 if clock_pos == 0 else int(clock_pos)
        return None

    def _evaluate_clock_positions(self, positions: List[Optional[int]]) -> float:
        """핸들 그립 위치 점수 평가"""
        valid_pos = sorted([p for p in positions if p is not None])
        if len(valid_pos) == 2:
            p1, p2 = valid_pos
            if (p1, p2) in [(2, 10), (3, 9), (4, 8)]: return 1.0
            return 0.5
        elif len(valid_pos) == 1:
            return 0.3
        return 0.0

    def _generate_steering_feedback(self, score: float, positions: List, hands_count: int) -> str:
        """핸들링 스킬 피드백 생성"""
        if hands_count == 0: return "핸들을 잡아주세요."
        if hands_count == 1: return "안전을 위해 양손으로 핸들을 잡아주세요."
        if not any(p is not None for p in positions): return "핸들 중앙을 제대로 잡아주세요."
        if score > 0.8: return "매우 안정적인 핸들링입니다."
        return "핸들 그립을 10시-2시 또는 9시-3시 방향으로 조정하여 안정성을 높이세요."

    def _infer_hand_gesture(self, kinematics: Dict, zone: Dict, grip: Dict) -> str:
        """[S-Class] 의도 기반 제스처 인식"""
        velocity = kinematics['velocity_magnitude']
        target_zone = zone['primary_zone']
        
        if target_zone == 'STEERING_WHEEL':
            if grip['grip_type'] == 'power_grip':
                if velocity < 0.02: return 'holding_wheel_steady'
                return 'steering'
            else: return 'touching_wheel'
        if target_zone == 'GEAR_LEVER' and velocity > 0.05: return 'shifting_gear'
        if target_zone == 'CENTER_CONSOLE' and velocity > 0.05: return 'operating_console'
        if velocity > 0.25: return 'rapid_movement'
        return 'resting_hand'

    def _calculate_overall_safety_score(self, steering: Dict, distraction: Dict) -> float:
        """전체 손 안전도 점수 계산"""
        steering_factor = steering.get('skill_score', 0.0)
        distraction_factor = 1.0 - distraction.get('risk_score', 0.0)
        return (steering_factor * 0.5) + (distraction_factor * 0.5)

    def _update_hand_metrics(self, analysis: Dict[str, Any]):
        """
        메트릭 업데이트를 통해 분석 결과를 전송합니다.
        definitions.py의 AdvancedMetrics 구조에 맞춰 데이터를 업데이트합니다.
        """
        try:
            # 안전 구역 내 손 위치 확인
            left_hand_safe, right_hand_safe = True, True
            if analysis['hands_detected_count'] > 0:
                left_hand = next((h for h in analysis['hand_positions'] if h['handedness'] == 'Left'), None)
                right_hand = next((h for h in analysis['hand_positions'] if h['handedness'] == 'Right'), None)
                if left_hand:
                    left_hand_safe = left_hand['zone_analysis']['primary_zone'] == 'STEERING_WHEEL'
                if right_hand:
                    right_hand_safe = right_hand['zone_analysis']['primary_zone'] == 'STEERING_WHEEL'

            # 핸들링 안정성 점수 (움직임 + 떨림)
            steering_comps = analysis.get('steering_skill', {}).get('components', {})
            stability = (steering_comps.get('movement_smoothness', 0.0) + steering_comps.get('hand_stability', 0.0)) / 2
            
            # IMetricsUpdater 인터페이스의 메서드에 맞게 데이터 구성
            distraction_data = {
                'distraction_risk_score': analysis.get('distraction_behaviors', {}).get('risk_score', 0.0),
                'left_hand_in_safe_zone': left_hand_safe,
                'right_hand_in_safe_zone': right_hand_safe,
                'hand_stability_score': stability,
                'phone_detected': analysis.get('distraction_behaviors', {}).get('phone_detected', False),
            }
            self.metrics_updater.update_distraction_metrics(distraction_data)

        except Exception as e:
            logger.error(f"손 메트릭 업데이트 중 오류: {e}")

    async def _handle_no_hands_detected(self) -> Dict[str, Any]:
        """손이 감지되지 않았을 때의 처리"""
        logger.warning("손이 감지되지 않았습니다.")
        analysis = await self._get_default_hand_analysis()
        self._update_hand_metrics(analysis) # 손 미감지 상태도 메트릭으로 전송
        return {'hand_positions': [], 'hand_analysis': analysis}

    async def _get_default_hand_analysis(self) -> Dict[str, Any]:
        """손 미감지 시 반환할 기본 분석 데이터"""
        return {
            'hands_detected_count': 0,
            'steering_skill': {'skill_score': 0.0, 'feedback': 'No hands detected', 'components': {}},
            'distraction_behaviors': {'risk_score': 1.0, 'behaviors': ['손 미감지'], 'phone_detected': False},
            'driving_technique': {'technique_rating': 'unknown', 'score': 0.0},
            'overall_hand_safety': 0.0,
            'hand_positions': [] # 메트릭 업데이트 함수에서 참조할 수 있도록 빈 리스트 추가
        }
