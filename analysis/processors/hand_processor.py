"""
손 데이터 전문 처리기
손의 위치, 제스처, 핸들 그립 상태 등을 전담 분석합니다.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from core.interfaces import IHandDataProcessor, IMetricsUpdater
from core.constants import VehicleConstants, AnalysisConstants, MathConstants
from config.settings import get_config
import logging

logger = logging.getLogger(__name__)


class HandDataProcessor(IHandDataProcessor):
    """
    손 데이터 전문 처리기
    
    이 클래스는 마치 손 동작 전문가처럼 운전자의 손을 관찰합니다.
    - 핸들 그립 상태 분석
    - 손의 안정성과 떨림 감지
    - 위험한 손 위치 탐지 (핸들에서 벗어남)
    - 스마트폰 사용 등 주의산만 행동 감지
    - 손목과 손가락의 자세 분석
    """
    
    def __init__(self, metrics_updater: IMetricsUpdater):
        self.metrics_updater = metrics_updater
        self.config = get_config()
        
        # 손 추적을 위한 이력 관리
        self.hand_position_history = deque(maxlen=90)  # 3초 이력
        self.grip_quality_history = deque(maxlen=60)   # 2초 이력
        self.stability_history = deque(maxlen=30)      # 1초 이력
        
        # 차량 내부 구역 정의
        self.vehicle_zones = VehicleConstants.Zones
        
        logger.info("HandDataProcessor 초기화 완료 - 손 동작 분석 전문가 준비됨")
    
    def get_processor_name(self) -> str:
        return "HandDataProcessor"
    
    def get_required_data_types(self) -> List[str]:
        return ["hand_landmarks", "handedness"]
    
    async def process_data(self, data: Any, timestamp: float) -> Dict[str, Any]:
        """
        손 데이터 통합 처리 메인 메서드
        
        두 손의 모든 정보를 종합적으로 분석하여 운전 안전성을 평가합니다.
        """
        if not data or not data.hand_landmarks:
            return await self._handle_no_hands_detected()
        
        hand_result = data
        hand_positions = await self.process_hand_landmarks(hand_result, timestamp)
        
        # 종합적인 손 분석
        comprehensive_analysis = await self.perform_comprehensive_hand_analysis(
            hand_positions, timestamp
        )
        
        # 결과 통합
        results = {
            'hand_positions': hand_positions,
            'hand_analysis': comprehensive_analysis
        }
        
        # 메트릭 업데이트
        self._update_hand_metrics(comprehensive_analysis)
        
        return results
    
    async def process_hand_landmarks(self, hand_results: Any, timestamp: float) -> List[Dict[str, Any]]:
        """
        손 랜드마크 처리
        
        각 손의 21개 키포인트를 분석하여 상세한 손 상태를 파악합니다.
        """
        hand_positions = []
        
        try:
            for i, hand_landmarks in enumerate(hand_results.hand_landmarks):
                # 손의 좌우 구분
                handedness = hand_results.handedness[i][0].category_name
                confidence = hand_results.handedness[i][0].score
                
                # 손목 위치 (랜드마크 인덱스 0)
                wrist = hand_landmarks[0]
                
                # 손가락 끝 위치들 (엄지부터 새끼까지)
                fingertips = [
                    hand_landmarks[4],   # 엄지
                    hand_landmarks[8],   # 검지
                    hand_landmarks[12],  # 중지
                    hand_landmarks[16],  # 약지
                    hand_landmarks[20]   # 새끼
                ]
                
                # 손 중심점 계산
                hand_center = self._calculate_hand_center(hand_landmarks)
                
                # 손 크기 추정
                hand_size = self._estimate_hand_size(hand_landmarks)
                
                # 손 자세 분석
                hand_pose = await self._analyze_hand_pose(hand_landmarks)
                
                # 손 움직임 분석
                movement_analysis = self._analyze_hand_movement(hand_center, handedness, timestamp)
                
                # 그립 품질 분석
                grip_analysis = self._analyze_grip_quality(hand_landmarks, hand_center)
                
                # 차량 구역 내 위치 확인
                zone_analysis = self._analyze_hand_zones(hand_center)
                
                hand_data = {
                    'handedness': handedness,
                    'confidence': confidence,
                    'wrist_position': {'x': wrist.x, 'y': wrist.y, 'z': getattr(wrist, 'z', 0.0)},
                    'hand_center': hand_center,
                    'hand_size': hand_size,
                    'fingertips': [{'x': tip.x, 'y': tip.y, 'z': getattr(tip, 'z', 0.0)} for tip in fingertips],
                    'landmarks': hand_landmarks,
                    'pose_analysis': hand_pose,
                    'movement_analysis': movement_analysis,
                    'grip_analysis': grip_analysis,
                    'zone_analysis': zone_analysis,
                    'timestamp': timestamp
                }
                
                hand_positions.append(hand_data)
                
        except Exception as e:
            logger.error(f"손 랜드마크 처리 중 오류 발생: {e}")
        
        # 손 위치 이력 업데이트
        self._update_hand_history(hand_positions, timestamp)
        
        return hand_positions
    
    async def perform_comprehensive_hand_analysis(self, hand_positions: List[Dict], timestamp: float) -> Dict[str, Any]:
        """
        종합적인 손 분석
        
        양손의 정보를 통합하여 전반적인 운전 자세와 안전성을 평가합니다.
        """
        try:
            if not hand_positions:
                return await self._get_default_hand_analysis()
            
            # 양손 협력 분석
            bilateral_analysis = self._analyze_bilateral_hand_coordination(hand_positions)
            
            # 핸들 그립 종합 평가
            steering_analysis = self._analyze_steering_grip_overall(hand_positions)
            
            # 주의산만 행동 감지
            distraction_analysis = self._detect_distraction_behaviors(hand_positions)
            
            # 손의 안정성 평가
            stability_analysis = self._analyze_hand_stability(hand_positions)
            
            # 안전 위험도 계산
            safety_assessment = self._calculate_hand_safety_score(
                bilateral_analysis, steering_analysis, distraction_analysis, stability_analysis
            )
            
            # 시간적 추세 분석
            temporal_analysis = self._analyze_temporal_hand_patterns()
            
            return {
                'hands_detected_count': len(hand_positions),
                'bilateral_coordination': bilateral_analysis,
                'steering_grip': steering_analysis,
                'distraction_behaviors': distraction_analysis,
                'stability_assessment': stability_analysis,
                'safety_score': safety_assessment,
                'temporal_patterns': temporal_analysis,
                'overall_hand_health': self._calculate_overall_hand_health(safety_assessment, stability_analysis)
            }
            
        except Exception as e:
            logger.error(f"종합 손 분석 중 오류 발생: {e}")
            return await self._get_default_hand_analysis()
    
    def _calculate_hand_center(self, hand_landmarks: Any) -> Dict[str, float]:
        """손의 중심점 계산"""
        try:
            # 모든 랜드마크의 평균 위치를 중심점으로 사용
            x_coords = [lm.x for lm in hand_landmarks]
            y_coords = [lm.y for lm in hand_landmarks]
            z_coords = [getattr(lm, 'z', 0.0) for lm in hand_landmarks]
            
            return {
                'x': sum(x_coords) / len(x_coords),
                'y': sum(y_coords) / len(y_coords),
                'z': sum(z_coords) / len(z_coords)
            }
        except Exception as e:
            logger.error(f"손 중심점 계산 중 오류: {e}")
            return {'x': 0.5, 'y': 0.5, 'z': 0.0}
    
    def _estimate_hand_size(self, hand_landmarks: Any) -> float:
        """손 크기 추정"""
        try:
            # 손목(0)에서 중지 끝(12)까지의 거리를 손 크기로 사용
            wrist = hand_landmarks[0]
            middle_tip = hand_landmarks[12]
            
            size = math.sqrt(
                (wrist.x - middle_tip.x) ** 2 + 
                (wrist.y - middle_tip.y) ** 2
            )
            return size
        except Exception as e:
            logger.error(f"손 크기 추정 중 오류: {e}")
            return 0.1  # 기본값
    
    async def _analyze_hand_pose(self, hand_landmarks: Any) -> Dict[str, Any]:
        """
        손 자세 분석
        
        손가락의 구부림 정도, 손의 회전 등을 분석하여 
        그립 상태와 제스처를 파악합니다.
        """
        try:
            # 손가락별 구부림 정도 계산
            finger_curl_analysis = self._calculate_finger_curl_degrees(hand_landmarks)
            
            # 손의 회전 각도 계산 (손등의 방향)
            hand_rotation = self._calculate_hand_rotation(hand_landmarks)
            
            # 그립 타입 분류 (주먹, 펼친 손, 그립 등)
            grip_type = self._classify_grip_type(finger_curl_analysis)
            
            # 손가락 간 간격 분석
            finger_spacing = self._analyze_finger_spacing(hand_landmarks)
            
            return {
                'finger_curl': finger_curl_analysis,
                'hand_rotation': hand_rotation,
                'grip_type': grip_type,
                'finger_spacing': finger_spacing,
                'overall_pose_score': self._calculate_pose_naturalness_score(finger_curl_analysis, hand_rotation)
            }
        except Exception as e:
            logger.error(f"손 자세 분석 중 오류: {e}")
            return {'grip_type': 'unknown', 'overall_pose_score': 0.5}
    
    def _analyze_hand_movement(self, hand_center: Dict, handedness: str, timestamp: float) -> Dict[str, Any]:
        """
        손 움직임 분석
        
        이전 프레임들과 비교하여 손의 움직임 패턴을 분석합니다.
        떨림, 급격한 움직임, 정적 상태 등을 감지합니다.
        """
        try:
            # 이전 위치와 비교하여 속도 계산
            velocity = self._calculate_hand_velocity(hand_center, handedness, timestamp)
            
            # 가속도 계산
            acceleration = self._calculate_hand_acceleration(hand_center, handedness, timestamp)
            
            # 떨림 감지 (고주파 작은 움직임)
            tremor_analysis = self._detect_hand_tremor(hand_center, handedness)
            
            # 움직임의 부드러움 평가
            smoothness_score = self._calculate_movement_smoothness(velocity, acceleration)
            
            return {
                'velocity': velocity,
                'acceleration': acceleration,
                'tremor': tremor_analysis,
                'smoothness_score': smoothness_score,
                'movement_type': self._classify_movement_type(velocity, acceleration, tremor_analysis)
            }
        except Exception as e:
            logger.error(f"손 움직임 분석 중 오류: {e}")
            return {'velocity': 0.0, 'movement_type': 'static'}
    
    def _analyze_grip_quality(self, hand_landmarks: Any, hand_center: Dict) -> Dict[str, Any]:
        """
        그립 품질 분석
        
        손이 핸들을 얼마나 잘 잡고 있는지 평가합니다.
        """
        try:
            # 핸들 영역과의 근접성
            steering_proximity = self._calculate_steering_wheel_proximity(hand_center)
            
            # 그립 강도 추정 (손가락 구부림 기반)
            grip_strength = self._estimate_grip_strength(hand_landmarks)
            
            # 그립 안정성 (시간에 따른 일관성)
            grip_consistency = self._calculate_grip_consistency(hand_center)
            
            # 그립 각도 (핸들과의 상대적 각도)
            grip_angle = self._calculate_grip_angle(hand_landmarks)
            
            # 전체 그립 품질 점수
            overall_quality = self._calculate_overall_grip_quality(
                steering_proximity, grip_strength, grip_consistency, grip_angle
            )
            
            return {
                'steering_proximity': steering_proximity,
                'grip_strength': grip_strength,
                'grip_consistency': grip_consistency,
                'grip_angle': grip_angle,
                'overall_quality': overall_quality,
                'is_gripping_steering': overall_quality > 0.6
            }
        except Exception as e:
            logger.error(f"그립 품질 분석 중 오류: {e}")
            return {'overall_quality': 0.0, 'is_gripping_steering': False}
    
    def _analyze_hand_zones(self, hand_center: Dict) -> Dict[str, Any]:
        """
        손의 차량 내 위치 분석
        
        손이 현재 어느 차량 구역에 있는지 파악하고,
        각 구역별 위험도를 평가합니다.
        """
        try:
            x, y = hand_center['x'], hand_center['y']
            current_zones = []
            risk_levels = []
            
            # 각 차량 구역별 체크
            zone_risk_map = {
                'steering_wheel': 0.0,     # 가장 안전
                'dashboard_area': 0.3,     # 약간 위험
                'gear_lever': 0.2,         # 운전 중에는 위험하지만 필요한 행동
                'center_console': 0.4,     # 주의산만 가능성
                'side_mirror_left': 0.8,   # 위험 (운전 중 조작)
                'outside_zones': 0.9       # 가장 위험
            }
            
            for zone_name, zone_bounds in self.vehicle_zones.items():
                if self._is_point_in_zone(x, y, zone_bounds):
                    current_zones.append(zone_name)
                    risk_levels.append(zone_risk_map.get(zone_name, 0.5))
            
            # 어떤 구역에도 속하지 않는 경우
            if not current_zones:
                current_zones = ['outside_zones']
                risk_levels = [zone_risk_map['outside_zones']]
            
            # 가장 높은 위험도 선택
            max_risk = max(risk_levels) if risk_levels else 0.5
            primary_zone = current_zones[risk_levels.index(max_risk)] if current_zones else 'unknown'
            
            return {
                'primary_zone': primary_zone,
                'all_zones': current_zones,
                'risk_level': max_risk,
                'is_in_safe_zone': max_risk < 0.3,
                'zone_description': self._get_zone_description(primary_zone)
            }
        except Exception as e:
            logger.error(f"손 구역 분석 중 오류: {e}")
            return {'primary_zone': 'unknown', 'risk_level': 0.5}
    
    def _analyze_bilateral_hand_coordination(self, hand_positions: List[Dict]) -> Dict[str, Any]:
        """
        양손 협력 분석
        
        양손이 얼마나 잘 협력하여 운전하고 있는지 평가합니다.
        """
        try:
            if len(hand_positions) < 2:
                return {'coordination_possible': False, 'single_hand_detected': len(hand_positions)}
            
            left_hand = next((h for h in hand_positions if h['handedness'] == 'Left'), None)
            right_hand = next((h for h in hand_positions if h['handedness'] == 'Right'), None)
            
            if not left_hand or not right_hand:
                return {'coordination_possible': False, 'reason': 'missing_hand_data'}
            
            # 양손 간 거리
            hand_distance = self._calculate_hand_distance(left_hand['hand_center'], right_hand['hand_center'])
            
            # 양손의 높이 일치성 (y좌표 차이)
            height_alignment = abs(left_hand['hand_center']['y'] - right_hand['hand_center']['y'])
            
            # 양손의 동시 그립 여부
            both_gripping = (left_hand['grip_analysis']['is_gripping_steering'] and 
                           right_hand['grip_analysis']['is_gripping_steering'])
            
            # 움직임 동조성 (양손이 비슷하게 움직이는지)
            movement_sync = self._calculate_movement_synchronization(left_hand, right_hand)
            
            # 전체 협력 점수
            coordination_score = self._calculate_coordination_score(
                hand_distance, height_alignment, both_gripping, movement_sync
            )
            
            return {
                'coordination_possible': True,
                'hand_distance': hand_distance,
                'height_alignment': height_alignment,
                'both_hands_gripping': both_gripping,
                'movement_synchronization': movement_sync,
                'overall_coordination': coordination_score,
                'coordination_quality': self._rate_coordination_quality(coordination_score)
            }
        except Exception as e:
            logger.error(f"양손 협력 분석 중 오류: {e}")
            return {'coordination_possible': False, 'error': str(e)}
    
    def _analyze_steering_grip_overall(self, hand_positions: List[Dict]) -> Dict[str, Any]:
        """핸들 그립 종합 평가"""
        try:
            hands_on_wheel = sum(1 for hand in hand_positions 
                               if hand['grip_analysis']['is_gripping_steering'])
            
            # 그립 품질 평균
            grip_qualities = [hand['grip_analysis']['overall_quality'] for hand in hand_positions]
            avg_grip_quality = sum(grip_qualities) / len(grip_qualities) if grip_qualities else 0.0
            
            # 핸들 그립 안전성 평가
            grip_safety = self._evaluate_grip_safety(hands_on_wheel, avg_grip_quality)
            
            return {
                'hands_on_wheel_count': hands_on_wheel,
                'hands_on_wheel_ratio': hands_on_wheel / 2.0,  # 최대 2개 손
                'average_grip_quality': avg_grip_quality,
                'grip_safety_level': grip_safety,
                'recommendation': self._generate_grip_recommendation(hands_on_wheel, avg_grip_quality)
            }
        except Exception as e:
            logger.error(f"핸들 그립 종합 평가 중 오류: {e}")
            return {'hands_on_wheel_count': 0, 'grip_safety_level': 'unknown'}
    
    def _detect_distraction_behaviors(self, hand_positions: List[Dict]) -> Dict[str, Any]:
        """
        주의산만 행동 감지
        
        스마트폰 사용, 불필요한 조작 등 위험한 손 동작을 감지합니다.
        """
        try:
            distraction_indicators = []
            risk_score = 0.0
            
            for hand in hand_positions:
                zone = hand['zone_analysis']['primary_zone']
                risk = hand['zone_analysis']['risk_level']
                
                # 고위험 구역에서의 활동
                if risk > 0.7:
                    distraction_indicators.append(f"{hand['handedness']} hand in {zone}")
                    risk_score = max(risk_score, risk)
                
                # 비정상적인 손 움직임
                if hand['movement_analysis']['movement_type'] == 'erratic':
                    distraction_indicators.append(f"{hand['handedness']} hand erratic movement")
                    risk_score = max(risk_score, 0.6)
                
                # 핸들에서 벗어난 시간이 긴 경우
                if not hand['grip_analysis']['is_gripping_steering']:
                    distraction_indicators.append(f"{hand['handedness']} hand off steering")
                    risk_score = max(risk_score, 0.5)
            
            # 전체 위험도 평가
            overall_distraction_level = self._calculate_overall_distraction_level(risk_score, distraction_indicators)
            
            return {
                'detected_behaviors': distraction_indicators,
                'risk_score': risk_score,
                'distraction_level': overall_distraction_level,
                'requires_attention': risk_score > 0.6,
                'safety_recommendation': self._generate_distraction_recommendation(distraction_indicators)
            }
        except Exception as e:
            logger.error(f"주의산만 행동 감지 중 오류: {e}")
            return {'detected_behaviors': [], 'risk_score': 0.0}
    
    def _analyze_hand_stability(self, hand_positions: List[Dict]) -> Dict[str, Any]:
        """손의 안정성 분석"""
        try:
            stability_scores = []
            tremor_levels = []
            
            for hand in hand_positions:
                # 개별 손의 안정성
                stability = hand['movement_analysis']['smoothness_score']
                tremor = hand['movement_analysis']['tremor']['severity']
                
                stability_scores.append(stability)
                tremor_levels.append(tremor)
            
            # 전체 안정성 평균
            overall_stability = sum(stability_scores) / len(stability_scores) if stability_scores else 0.5
            max_tremor = max(tremor_levels) if tremor_levels else 0.0
            
            # 시간적 안정성 추세
            temporal_stability = self._analyze_temporal_stability()
            
            return {
                'overall_stability': overall_stability,
                'max_tremor_level': max_tremor,
                'temporal_stability': temporal_stability,
                'stability_rating': self._rate_stability(overall_stability, max_tremor),
                'fatigue_indicators': self._extract_fatigue_from_stability(overall_stability, max_tremor)
            }
        except Exception as e:
            logger.error(f"손 안정성 분석 중 오류: {e}")
            return {'overall_stability': 0.5, 'stability_rating': 'unknown'}
    
    def _calculate_hand_safety_score(self, bilateral: Dict, steering: Dict, distraction: Dict, stability: Dict) -> Dict[str, Any]:
        """손 관련 전체 안전 점수 계산"""
        try:
            # 각 영역별 가중치
            weights = {
                'steering_grip': 0.4,      # 핸들 그립이 가장 중요
                'distraction': 0.3,        # 주의산만도 중요
                'stability': 0.2,          # 안정성
                'coordination': 0.1        # 협력성
            }
            
            # 각 영역 점수 추출
            steering_score = steering.get('hands_on_wheel_ratio', 0.0)
            distraction_score = 1.0 - distraction.get('risk_score', 0.0)  # 위험도를 안전도로 변환
            stability_score = stability.get('overall_stability', 0.5)
            coordination_score = bilateral.get('overall_coordination', 0.5) if bilateral.get('coordination_possible') else 0.5
            
            # 가중 평균으로 전체 점수 계산
            overall_score = (
                steering_score * weights['steering_grip'] +
                distraction_score * weights['distraction'] +
                stability_score * weights['stability'] +
                coordination_score * weights['coordination']
            )
            
            # 안전 등급 결정
            safety_grade = self._determine_safety_grade(overall_score)
            
            return {
                'overall_score': overall_score,
                'safety_grade': safety_grade,
                'component_scores': {
                    'steering': steering_score,
                    'distraction_safety': distraction_score,
                    'stability': stability_score,
                    'coordination': coordination_score
                },
                'is_safe': overall_score > 0.6,
                'improvement_suggestions': self._generate_improvement_suggestions(
                    steering_score, distraction_score, stability_score, coordination_score
                )
            }
        except Exception as e:
            logger.error(f"손 안전 점수 계산 중 오류: {e}")
            return {'overall_score': 0.0, 'safety_grade': 'unknown', 'is_safe': False}
    
    # 헬퍼 메서드들
    def _calculate_finger_curl_degrees(self, landmarks: Any) -> Dict[str, float]:
        """손가락별 구부림 정도 계산"""
        try:
            # 각 손가락의 관절 각도 계산 (간단화된 버전)
            finger_curls = {}
            
            # 엄지손가락 (2, 3, 4번 랜드마크)
            finger_curls['thumb'] = self._calculate_joint_angle(landmarks[2], landmarks[3], landmarks[4])
            
            # 검지손가락 (5, 6, 7, 8번 랜드마크)
            finger_curls['index'] = self._calculate_joint_angle(landmarks[6], landmarks[7], landmarks[8])
            
            # 중지손가락 (9, 10, 11, 12번 랜드마크)
            finger_curls['middle'] = self._calculate_joint_angle(landmarks[10], landmarks[11], landmarks[12])
            
            # 약지손가락 (13, 14, 15, 16번 랜드마크)
            finger_curls['ring'] = self._calculate_joint_angle(landmarks[14], landmarks[15], landmarks[16])
            
            # 새끼손가락 (17, 18, 19, 20번 랜드마크)
            finger_curls['pinky'] = self._calculate_joint_angle(landmarks[18], landmarks[19], landmarks[20])
            
            return finger_curls
        except Exception as e:
            logger.error(f"손가락 구부림 계산 중 오류: {e}")
            return {'thumb': 0, 'index': 0, 'middle': 0, 'ring': 0, 'pinky': 0}
    
    def _calculate_joint_angle(self, p1: Any, p2: Any, p3: Any) -> float:
        """세 점으로 관절 각도 계산"""
        try:
            # 벡터 계산
            v1 = np.array([p1.x - p2.x, p1.y - p2.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            
            # 각도 계산
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + MathConstants.EPSILON)
            angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
            
            return angle
        except Exception as e:
            return 0.0
    
    def _calculate_hand_velocity(self, current_center: Dict, handedness: str, timestamp: float) -> float:
        """손의 속도 계산"""
        try:
            # 이전 위치 찾기
            for prev_data in reversed(self.hand_position_history):
                for hand in prev_data.get('hands', []):
                    if hand.get('handedness') == handedness:
                        prev_center = hand.get('hand_center', {})
                        time_diff = timestamp - prev_data.get('timestamp', timestamp)
                        
                        if time_diff > 0:
                            distance = math.sqrt(
                                (current_center['x'] - prev_center.get('x', current_center['x'])) ** 2 +
                                (current_center['y'] - prev_center.get('y', current_center['y'])) ** 2
                            )
                            return distance / time_diff
            
            return 0.0
        except Exception as e:
            logger.error(f"손 속도 계산 중 오류: {e}")
            return 0.0
    
    def _calculate_hand_acceleration(self, current_center: Dict, handedness: str, timestamp: float) -> float:
        """손의 가속도 계산"""
        # 간단화된 구현 - 실제로는 속도의 변화율을 계산해야 함
        try:
            current_velocity = self._calculate_hand_velocity(current_center, handedness, timestamp)
            # 이전 속도와 비교하여 가속도 계산 (여기서는 간단화)
            return abs(current_velocity)  # 실제 구현에서는 더 정교하게
        except Exception:
            return 0.0
    
    def _detect_hand_tremor(self, hand_center: Dict, handedness: str) -> Dict[str, Any]:
        """손 떨림 감지"""
        try:
            # 최근 이력에서 고주파 작은 움직임 패턴 찾기
            tremor_indicators = 0
            recent_positions = []
            
            for data in list(self.hand_position_history)[-10:]:  # 최근 10프레임
                for hand in data.get('hands', []):
                    if hand.get('handedness') == handedness:
                        recent_positions.append(hand.get('hand_center', {}))
            
            if len(recent_positions) >= 5:
                # 위치 변화의 표준편차로 떨림 정도 측정
                x_positions = [pos.get('x', 0) for pos in recent_positions]
                y_positions = [pos.get('y', 0) for pos in recent_positions]
                
                x_std = np.std(x_positions)
                y_std = np.std(y_positions)
                
                tremor_severity = (x_std + y_std) / 2.0
                
                return {
                    'detected': tremor_severity > 0.01,  # 임계값
                    'severity': tremor_severity,
                    'confidence': min(1.0, tremor_severity * 100)
                }
            
            return {'detected': False, 'severity': 0.0, 'confidence': 0.0}
        except Exception as e:
            logger.error(f"손 떨림 감지 중 오류: {e}")
            return {'detected': False, 'severity': 0.0}
    
    def _update_hand_history(self, hand_positions: List[Dict], timestamp: float):
        """손 위치 이력 업데이트"""
        try:
            self.hand_position_history.append({
                'timestamp': timestamp,
                'hands': hand_positions
            })
        except Exception as e:
            logger.error(f"손 이력 업데이트 중 오류: {e}")
    
    def _is_point_in_zone(self, x: float, y: float, zone: Dict) -> bool:
        """점이 구역 내에 있는지 확인"""
        return zone["x1"] <= x <= zone["x2"] and zone["y1"] <= y <= zone["y2"]
    
    def _update_hand_metrics(self, hand_analysis: Dict[str, Any]):
        """손 관련 메트릭 업데이트"""
        try:
            metrics_data = {
                'hands_on_wheel_confidence': hand_analysis.get('steering_grip', {}).get('hands_on_wheel_ratio', 0.0),
                'hand_stability_score': hand_analysis.get('stability_assessment', {}).get('overall_stability', 0.5),
                'hand_distraction_risk': hand_analysis.get('distraction_behaviors', {}).get('risk_score', 0.0)
            }
            
            # 메트릭 업데이터를 통해 업데이트
            if hasattr(self.metrics_updater, 'update_hand_metrics'):
                self.metrics_updater.update_hand_metrics(metrics_data)
                
        except Exception as e:
            logger.error(f"손 메트릭 업데이트 중 오류: {e}")
    
    async def _handle_no_hands_detected(self) -> Dict[str, Any]:
        """손이 감지되지 않은 상황 처리"""
        logger.warning("손이 감지되지 않음 - 백업 시스템 필요")
        
        return {
            'hands_detected': False,
            'hand_positions': [],
            'hand_analysis': await self._get_default_hand_analysis()
        }
    
    async def _get_default_hand_analysis(self) -> Dict[str, Any]:
        """기본 손 분석 데이터"""
        return {
            'hands_detected_count': 0,
            'bilateral_coordination': {'coordination_possible': False},
            'steering_grip': {'hands_on_wheel_count': 0, 'hands_on_wheel_ratio': 0.0},
            'distraction_behaviors': {'detected_behaviors': [], 'risk_score': 0.5},
            'stability_assessment': {'overall_stability': 0.0},
            'safety_score': {'overall_score': 0.0, 'is_safe': False},
            'overall_hand_health': 'poor'
        }
    
    # 추가 헬퍼 메서드들 (간단화된 구현)
    def _calculate_hand_rotation(self, landmarks: Any) -> float:
        """손 회전 각도 계산"""
        return 0.0  # 간단화된 구현
    
    def _classify_grip_type(self, finger_curls: Dict) -> str:
        """그립 타입 분류"""
        avg_curl = sum(finger_curls.values()) / len(finger_curls)
        if avg_curl > 120:
            return 'closed_fist'
        elif avg_curl > 60:
            return 'grip'
        else:
            return 'open_hand'
    
    def _analyze_finger_spacing(self, landmarks: Any) -> Dict[str, float]:
        """손가락 간 간격 분석"""
        return {'average_spacing': 0.1}  # 간단화된 구현
    
    def _calculate_pose_naturalness_score(self, finger_curls: Dict, rotation: float) -> float:
        """자연스러운 자세 점수"""
        return 0.7  # 간단화된 구현
    
    def _calculate_movement_smoothness(self, velocity: float, acceleration: float) -> float:
        """움직임 부드러움 계산"""
        return max(0.0, 1.0 - min(velocity * 10, acceleration * 10))
    
    def _classify_movement_type(self, velocity: float, acceleration: float, tremor: Dict) -> str:
        """움직임 타입 분류"""
        if tremor.get('detected'):
            return 'tremor'
        elif velocity > 0.1:
            return 'active'
        elif acceleration > 0.1:
            return 'erratic'
        else:
            return 'static'
    
    def _calculate_steering_wheel_proximity(self, hand_center: Dict) -> float:
        """핸들과의 근접성 계산"""
        wheel_zone = self.vehicle_zones.STEERING_WHEEL
        wheel_center_x = (wheel_zone['x1'] + wheel_zone['x2']) / 2
        wheel_center_y = (wheel_zone['y1'] + wheel_zone['y2']) / 2
        
        distance = math.sqrt(
            (hand_center['x'] - wheel_center_x) ** 2 + 
            (hand_center['y'] - wheel_center_y) ** 2
        )
        
        return max(0.0, 1.0 - distance * 2)  # 거리 기반 근접성
    
    def _estimate_grip_strength(self, landmarks: Any) -> float:
        """그립 강도 추정"""
        return 0.7  # 간단화된 구현
    
    def _calculate_grip_consistency(self, hand_center: Dict) -> float:
        """그립 일관성 계산"""
        return 0.8  # 간단화된 구현
    
    def _calculate_grip_angle(self, landmarks: Any) -> float:
        """그립 각도 계산"""
        return 0.0  # 간단화된 구현
    
    def _calculate_overall_grip_quality(self, proximity: float, strength: float, consistency: float, angle: float) -> float:
        """전체 그립 품질 계산"""
        return (proximity * 0.4 + strength * 0.3 + consistency * 0.2 + (1.0 - abs(angle) / 180.0) * 0.1)
    
    def _get_zone_description(self, zone: str) -> str:
        """구역 설명 반환"""
        descriptions = {
            'steering_wheel': '핸들',
            'dashboard_area': '대시보드',
            'gear_lever': '기어 레버',
            'center_console': '중앙 콘솔',
            'outside_zones': '차량 외부'
        }
        return descriptions.get(zone, '알 수 없음')
    
    def _calculate_hand_distance(self, left_center: Dict, right_center: Dict) -> float:
        """양손 간 거리 계산"""
        return math.sqrt(
            (left_center['x'] - right_center['x']) ** 2 + 
            (left_center['y'] - right_center['y']) ** 2
        )
    
    def _calculate_movement_synchronization(self, left_hand: Dict, right_hand: Dict) -> float:
        """양손 움직임 동조성 계산"""
        return 0.8  # 간단화된 구현
    
    def _calculate_coordination_score(self, distance: float, alignment: float, both_gripping: bool, sync: float) -> float:
        """협력 점수 계산"""
        distance_score = max(0.0, 1.0 - distance * 2)  # 가까울수록 좋음
        alignment_score = max(0.0, 1.0 - alignment * 5)  # 높이가 비슷할수록 좋음
        grip_score = 1.0 if both_gripping else 0.5
        
        return (distance_score * 0.3 + alignment_score * 0.2 + grip_score * 0.3 + sync * 0.2)
    
    def _rate_coordination_quality(self, score: float) -> str:
        """협력 품질 등급화"""
        if score > 0.8:
            return 'excellent'
        elif score > 0.6:
            return 'good'
        elif score > 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _evaluate_grip_safety(self, hands_count: int, quality: float) -> str:
        """그립 안전성 평가"""
        if hands_count == 2 and quality > 0.7:
            return 'very_safe'
        elif hands_count >= 1 and quality > 0.5:
            return 'safe'
        elif hands_count >= 1:
            return 'caution'
        else:
            return 'unsafe'
    
    def _generate_grip_recommendation(self, hands_count: int, quality: float) -> str:
        """그립 권장사항 생성"""
        if hands_count == 0:
            return "양손을 핸들에 올려주세요"
        elif hands_count == 1:
            return "두 번째 손도 핸들에 올려주세요"
        elif quality < 0.5:
            return "핸들을 더 확실하게 잡아주세요"
        else:
            return "좋은 그립을 유지하고 있습니다"
    
    def _calculate_overall_distraction_level(self, risk_score: float, indicators: List[str]) -> str:
        """전체 주의산만 수준 계산"""
        if risk_score > 0.8:
            return 'high'
        elif risk_score > 0.5:
            return 'medium'
        elif risk_score > 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _generate_distraction_recommendation(self, indicators: List[str]) -> str:
        """주의산만 개선 권장사항"""
        if not indicators:
            return "양손 사용 패턴이 안전합니다"
        else:
            return f"주의 필요: {', '.join(indicators[:2])}"  # 최대 2개까지만 표시
    
    def _analyze_temporal_stability(self) -> Dict[str, Any]:
        """시간적 안정성 분석"""
        return {'trend': 'stable', 'confidence': 0.8}  # 간단화된 구현
    
    def _rate_stability(self, stability: float, tremor: float) -> str:
        """안정성 등급화"""
        if stability > 0.8 and tremor < 0.1:
            return 'excellent'
        elif stability > 0.6 and tremor < 0.3:
            return 'good'
        elif stability > 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _extract_fatigue_from_stability(self, stability: float, tremor: float) -> Dict[str, Any]:
        """안정성에서 피로도 지표 추출"""
        fatigue_score = (1.0 - stability) * 0.7 + tremor * 0.3
        return {
            'fatigue_from_hands': fatigue_score,
            'tremor_related_fatigue': tremor > 0.2,
            'stability_related_fatigue': stability < 0.5
        }
    
    def _determine_safety_grade(self, score: float) -> str:
        """안전 등급 결정"""
        if score > 0.9:
            return 'A+'
        elif score > 0.8:
            return 'A'
        elif score > 0.7:
            return 'B'
        elif score > 0.6:
            return 'C'
        elif score > 0.5:
            return 'D'
        else:
            return 'F'
    
    def _generate_improvement_suggestions(self, steering: float, distraction: float, stability: float, coordination: float) -> List[str]:
        """개선 제안 생성"""
        suggestions = []
        
        if steering < 0.5:
            suggestions.append("양손을 핸들에 올리세요")
        if distraction < 0.5:
            suggestions.append("핸들에서 손을 떼지 마세요")
        if stability < 0.5:
            suggestions.append("손의 떨림을 줄이고 안정된 그립을 유지하세요")
        if coordination < 0.5:
            suggestions.append("양손의 협력을 개선하세요")
        
        return suggestions if suggestions else ["현재 손 사용 패턴이 양호합니다"]
    
    def _analyze_temporal_hand_patterns(self) -> Dict[str, Any]:
        """손 패턴의 시간적 분석"""
        return {'pattern_detected': False, 'trend': 'stable'}  # 간단화된 구현
    
    def _calculate_overall_hand_health(self, safety: Dict, stability: Dict) -> str:
        """전체 손 건강도 계산"""
        safety_score = safety.get('overall_score', 0.0)
        stability_score = stability.get('overall_stability', 0.0)
        
        combined_score = (safety_score + stability_score) / 2.0
        
        if combined_score > 0.8:
            return 'excellent'
        elif combined_score > 0.6:
            return 'good'
        elif combined_score > 0.4:
            return 'fair'
        else:
            return 'poor'