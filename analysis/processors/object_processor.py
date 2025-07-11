"""
객체 데이터 전문 처리기
주의산만을 유발할 수 있는 객체들을 실시간으로 감지하고 분석합니다.
"""

import math
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from core.interfaces import IObjectDataProcessor, IMetricsUpdater
from core.constants import AnalysisConstants
from config.settings import get_config
import logging

logger = logging.getLogger(__name__)


class ObjectDataProcessor(IObjectDataProcessor):
    """
    객체 데이터 전문 처리기
    
    이 클래스는 마치 보안 요원처럼 운전자 주변의 모든 물체를 감시합니다.
    - 주의산만 유발 객체 감지 (휴대폰, 음식, 책 등)
    - 객체와 손의 상호작용 분석
    - 위험도 평가 및 지속적 위험 행동 감지
    - 객체별 맞춤형 위험도 산정
    - 시간적 패턴 분석을 통한 습관적 행동 감지
    """
    
    def __init__(self, metrics_updater: IMetricsUpdater):
        self.metrics_updater = metrics_updater
        self.config = get_config()
        
        # 주의산만 객체 정의 (설정에서 가져오기)
        self.distraction_objects = self.config.distraction.object_risk_levels
        
        # 감지 이력 관리
        self.detection_history = deque(maxlen=self.config.distraction.detection_history_size)
        self.interaction_patterns = deque(maxlen=300)  # 10초간 상호작용 패턴
        self.risk_assessment_cache = {}
        
        # 객체별 추적 상태
        self.tracked_objects = {}
        self.object_lifecycle = {}
        
        logger.info("ObjectDataProcessor 초기화 완료 - 객체 감시 전문가 준비됨")
    
    def get_processor_name(self) -> str:
        return "ObjectDataProcessor"
    
    def get_required_data_types(self) -> List[str]:
        return ["object_detections", "hand_positions"]
    
    async def process_data(self, data: Any, timestamp: float) -> Dict[str, Any]:
        """
        객체 데이터 통합 처리 메인 메서드
        
        감지된 모든 객체를 분석하여 운전 안전성에 미치는 영향을 종합 평가합니다.
        """
        if not data or not data.detections:
            return await self._handle_no_objects_detected(timestamp)
        
        # 손 위치 정보는 별도로 전달받음 (다른 프로세서에서)
        hand_positions = getattr(data, 'hand_positions', [])
        
        # 객체 감지 결과 처리
        object_analysis = await self.process_object_detections(
            data, hand_positions, timestamp
        )
        
        # 종합적인 위험도 분석
        comprehensive_analysis = await self.perform_comprehensive_risk_analysis(
            object_analysis, timestamp
        )
        
        # 결과 통합
        results = {
            'object_detections': object_analysis,
            'risk_analysis': comprehensive_analysis
        }
        
        # 메트릭 업데이트
        self._update_object_metrics(comprehensive_analysis)
        
        return results
    
    async def process_object_detections(self, detections: Any, hand_positions: List[Dict], timestamp: float) -> Dict[str, Any]:
        """
        객체 감지 결과 처리
        
        MediaPipe Object Detection의 결과를 분석하여
        각 객체의 위험도와 손과의 상호작용을 평가합니다.
        """
        try:
            detected_objects = []
            immediate_risk_score = 0.0
            interaction_events = []
            
            # 각 감지된 객체 분석
            for detection in detections.detections:
                category = detection.categories[0].category_name
                confidence = detection.categories[0].score
                bbox = detection.bounding_box
                
                # 관심 객체인지 확인
                if category in self.distraction_objects:
                    object_analysis = await self._analyze_single_object(
                        category, confidence, bbox, hand_positions, timestamp
                    )
                    
                    if object_analysis:
                        detected_objects.append(object_analysis)
                        immediate_risk_score = max(immediate_risk_score, object_analysis['risk_level'])
                        
                        # 상호작용 이벤트 기록
                        if object_analysis['hand_interaction']['is_interacting']:
                            interaction_events.append({
                                'object': category,
                                'interaction_type': object_analysis['hand_interaction']['interaction_type'],
                                'risk_level': object_analysis['risk_level'],
                                'timestamp': timestamp
                            })
            
            # 감지 이력 업데이트
            self._update_detection_history(detected_objects, immediate_risk_score, timestamp)
            
            # 지속적 위험 행동 감지
            persistent_risk = self._detect_persistent_risk_behaviors()
            
            # 객체 추적 업데이트
            self._update_object_tracking(detected_objects, timestamp)
            
            return {
                'detected_objects': detected_objects,
                'object_count': len(detected_objects),
                'immediate_risk_score': immediate_risk_score,
                'persistent_risk_score': persistent_risk,
                'interaction_events': interaction_events,
                'tracking_info': self._get_tracking_summary()
            }
            
        except Exception as e:
            logger.error(f"객체 감지 처리 중 오류 발생: {e}")
            return await self._get_default_object_analysis()
    
    async def _analyze_single_object(
        self, category: str, confidence: float, bbox: Any, 
        hand_positions: List[Dict], timestamp: float
    ) -> Optional[Dict[str, Any]]:
        """
        단일 객체 상세 분석
        
        하나의 객체에 대한 모든 측면을 분석합니다:
        - 기본 위험도 평가
        - 손과의 근접성 및 상호작용
        - 위치 기반 위험도 조정
        - 시간적 컨텍스트 고려
        """
        try:
            obj_info = self.distraction_objects[category]
            
            # 기본 객체 정보
            object_data = {
                'category': category,
                'description': obj_info['description'],
                'detection_confidence': confidence,
                'base_risk_level': obj_info['risk_level'],
                'bbox': self._normalize_bounding_box(bbox),
                'timestamp': timestamp
            }
            
            # 손과의 상호작용 분석
            hand_interaction = await self._analyze_hand_object_interaction(
                bbox, hand_positions, category
            )
            object_data['hand_interaction'] = hand_interaction
            
            # 위치 기반 위험도 조정
            position_risk = self._analyze_object_position_risk(bbox, category)
            object_data['position_analysis'] = position_risk
            
            # 최종 위험도 계산
            final_risk = self._calculate_final_object_risk(
                obj_info['risk_level'], confidence, 
                hand_interaction['proximity_score'], position_risk['risk_multiplier']
            )
            object_data['risk_level'] = final_risk
            
            # 상황별 위험도 조정
            contextual_risk = await self._apply_contextual_risk_adjustments(
                object_data, timestamp
            )
            object_data.update(contextual_risk)
            
            # 객체 생명주기 분석
            lifecycle_info = self._analyze_object_lifecycle(category, bbox, timestamp)
            object_data['lifecycle'] = lifecycle_info
            
            return object_data
            
        except Exception as e:
            logger.error(f"단일 객체 분석 중 오류 발생: {e}")
            return None
    
    async def _analyze_hand_object_interaction(
        self, bbox: Any, hand_positions: List[Dict], category: str
    ) -> Dict[str, Any]:
        """
        손과 객체의 상호작용 분석
        
        마치 행동 분석가처럼 손과 객체 사이의 미묘한 상호작용을 관찰합니다.
        """
        try:
            if not hand_positions:
                return {
                    'proximity_score': 0.0,
                    'is_interacting': False,
                    'interaction_type': 'no_hands_detected',
                    'closest_hand': None,
                    'interaction_confidence': 0.0
                }
            
            # 객체 중심점 계산
            bbox_center_x = bbox.origin_x + bbox.width / 2
            bbox_center_y = bbox.origin_y + bbox.height / 2
            
            min_distance = float('inf')
            closest_hand = None
            interaction_details = []
            
            # 각 손과의 거리 계산
            for hand in hand_positions:
                hand_x = hand.get('x', 0.5)
                hand_y = hand.get('y', 0.5)
                
                # 유클리드 거리 계산
                distance = math.sqrt(
                    (bbox_center_x - hand_x) ** 2 + 
                    (bbox_center_y - hand_y) ** 2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    closest_hand = hand
                
                # 상호작용 세부 분석
                interaction_detail = self._analyze_detailed_interaction(
                    hand, bbox, category, distance
                )
                interaction_details.append(interaction_detail)
            
            # 근접성 점수 계산 (0-1 범위)
            proximity_threshold = self.config.distraction.proximity_threshold
            proximity_score = max(0.0, 1.0 - min_distance / proximity_threshold)
            
            # 상호작용 여부 판단
            is_interacting = proximity_score > 0.3  # 30% 이상 근접시 상호작용으로 판단
            
            # 상호작용 타입 결정
            interaction_type = self._determine_interaction_type(
                proximity_score, closest_hand, category, interaction_details
            )
            
            # 상호작용 신뢰도 계산
            interaction_confidence = self._calculate_interaction_confidence(
                proximity_score, interaction_details
            )
            
            return {
                'proximity_score': proximity_score,
                'min_distance': min_distance,
                'is_interacting': is_interacting,
                'interaction_type': interaction_type,
                'closest_hand': closest_hand['handedness'] if closest_hand else None,
                'interaction_confidence': interaction_confidence,
                'all_hand_interactions': interaction_details
            }
            
        except Exception as e:
            logger.error(f"손-객체 상호작용 분석 중 오류: {e}")
            return {'proximity_score': 0.0, 'is_interacting': False}
    
    def _analyze_detailed_interaction(
        self, hand: Dict, bbox: Any, category: str, distance: float
    ) -> Dict[str, Any]:
        """상호작용 세부 분석"""
        try:
            hand_center = {'x': hand.get('x', 0.5), 'y': hand.get('y', 0.5)}
            
            # 손의 자세 정보가 있다면 활용
            grip_type = hand.get('pose_analysis', {}).get('grip_type', 'unknown')
            
            # 객체 영역 내 손의 위치
            is_hand_over_object = self._is_hand_over_object(hand_center, bbox)
            
            # 상호작용의 방향성 (손이 객체 쪽으로 움직이는지)
            movement_toward_object = self._analyze_movement_direction(hand, bbox)
            
            # 카테고리별 특수 분석
            category_specific = self._analyze_category_specific_interaction(
                category, hand, bbox, distance
            )
            
            return {
                'handedness': hand.get('handedness', 'unknown'),
                'distance': distance,
                'grip_type': grip_type,
                'is_over_object': is_hand_over_object,
                'movement_toward': movement_toward_object,
                'category_specific': category_specific
            }
        except Exception as e:
            logger.error(f"상호작용 세부 분석 중 오류: {e}")
            return {}
    
    def _analyze_object_position_risk(self, bbox: Any, category: str) -> Dict[str, Any]:
        """
        객체 위치 기반 위험도 분석
        
        같은 객체라도 위치에 따라 위험도가 달라집니다.
        예: 대시보드 위의 휴대폰 vs 조수석의 휴대폰
        """
        try:
            # 객체 중심점
            center_x = bbox.origin_x + bbox.width / 2
            center_y = bbox.origin_y + bbox.height / 2
            
            # 위험 구역 정의
            high_risk_zones = [
                {'name': 'driver_immediate', 'x1': 0.3, 'y1': 0.3, 'x2': 0.7, 'y2': 0.9, 'multiplier': 1.5},
                {'name': 'dashboard_center', 'x1': 0.4, 'y1': 0.0, 'x2': 0.6, 'y2': 0.3, 'multiplier': 1.3},
            ]
            
            medium_risk_zones = [
                {'name': 'center_console', 'x1': 0.45, 'y1': 0.5, 'x2': 0.55, 'y2': 0.8, 'multiplier': 1.1},
                {'name': 'passenger_side', 'x1': 0.7, 'y1': 0.3, 'x2': 1.0, 'y2': 0.9, 'multiplier': 0.8},
            ]
            
            # 해당하는 구역 찾기
            risk_multiplier = 1.0
            position_zone = 'neutral'
            
            for zone in high_risk_zones:
                if self._is_point_in_zone(center_x, center_y, zone):
                    risk_multiplier = zone['multiplier']
                    position_zone = zone['name']
                    break
            
            if risk_multiplier == 1.0:  # 고위험 구역에 없다면 중위험 구역 체크
                for zone in medium_risk_zones:
                    if self._is_point_in_zone(center_x, center_y, zone):
                        risk_multiplier = zone['multiplier']
                        position_zone = zone['name']
                        break
            
            # 카테고리별 위치 특수성
            category_position_adjustment = self._get_category_position_adjustment(category, position_zone)
            final_multiplier = risk_multiplier * category_position_adjustment
            
            return {
                'position_zone': position_zone,
                'base_risk_multiplier': risk_multiplier,
                'category_adjustment': category_position_adjustment,
                'risk_multiplier': final_multiplier,
                'position_description': self._get_position_description(position_zone)
            }
            
        except Exception as e:
            logger.error(f"객체 위치 위험도 분석 중 오류: {e}")
            return {'risk_multiplier': 1.0, 'position_zone': 'unknown'}
    
    def _calculate_final_object_risk(
        self, base_risk: float, confidence: float, 
        proximity: float, position_multiplier: float
    ) -> float:
        """
        최종 객체 위험도 계산
        
        여러 요소를 종합하여 객체의 최종 위험도를 산정합니다.
        """
        try:
            # 기본 위험도에 신뢰도 적용
            confidence_adjusted_risk = base_risk * confidence
            
            # 근접성 보너스 (손과 가까울수록 위험)
            proximity_bonus = proximity * 0.3
            
            # 위치 기반 조정
            position_adjusted_risk = confidence_adjusted_risk * position_multiplier
            
            # 최종 위험도 계산
            final_risk = min(1.0, position_adjusted_risk + proximity_bonus)
            
            return final_risk
            
        except Exception as e:
            logger.error(f"최종 위험도 계산 중 오류: {e}")
            return base_risk
    
    async def _apply_contextual_risk_adjustments(
        self, object_data: Dict, timestamp: float
    ) -> Dict[str, Any]:
        """
        상황별 위험도 조정
        
        시간, 빈도, 패턴 등을 고려한 맥락적 위험도 조정을 수행합니다.
        """
        try:
            category = object_data['category']
            base_risk = object_data['risk_level']
            
            # 시간적 지속성 고려
            duration_factor = self._calculate_object_duration_factor(category, timestamp)
            
            # 빈도 기반 조정 (자주 나타나는 객체는 습관적 행동일 가능성)
            frequency_factor = self._calculate_object_frequency_factor(category)
            
            # 동시 다발적 객체 출현에 대한 가중치
            multi_object_penalty = self._calculate_multi_object_penalty(timestamp)
            
            # 운전 상황 고려 (속도, 교통 상황 등 - 현재는 간단화)
            driving_context_factor = 1.0  # 추후 확장 가능
            
            # 조정된 위험도 계산
            adjusted_risk = base_risk * duration_factor * frequency_factor * multi_object_penalty * driving_context_factor
            adjusted_risk = min(1.0, adjusted_risk)
            
            return {
                'adjusted_risk_level': adjusted_risk,
                'risk_adjustments': {
                    'duration_factor': duration_factor,
                    'frequency_factor': frequency_factor,
                    'multi_object_penalty': multi_object_penalty,
                    'driving_context_factor': driving_context_factor
                },
                'risk_explanation': self._generate_risk_explanation(
                    category, base_risk, adjusted_risk, duration_factor, frequency_factor
                )
            }
            
        except Exception as e:
            logger.error(f"상황별 위험도 조정 중 오류: {e}")
            return {'adjusted_risk_level': object_data['risk_level']}
    
    def _analyze_object_lifecycle(self, category: str, bbox: Any, timestamp: float) -> Dict[str, Any]:
        """
        객체 생명주기 분석
        
        객체의 출현, 지속, 소멸 패턴을 추적하여 
        운전자의 행동 패턴을 이해합니다.
        """
        try:
            object_key = f"{category}_{int(bbox.origin_x * 100)}_{int(bbox.origin_y * 100)}"
            
            if object_key not in self.object_lifecycle:
                # 새로운 객체 등장
                self.object_lifecycle[object_key] = {
                    'first_detected': timestamp,
                    'last_seen': timestamp,
                    'total_duration': 0.0,
                    'appearance_count': 1,
                    'position_changes': 0,
                    'interaction_count': 0
                }
                lifecycle_status = 'newly_appeared'
            else:
                # 기존 객체 업데이트
                obj_lifecycle = self.object_lifecycle[object_key]
                obj_lifecycle['last_seen'] = timestamp
                obj_lifecycle['total_duration'] = timestamp - obj_lifecycle['first_detected']
                
                # 위치 변화 감지
                if self._has_object_moved(object_key, bbox):
                    obj_lifecycle['position_changes'] += 1
                
                lifecycle_status = 'continuing'
            
            current_lifecycle = self.object_lifecycle[object_key]
            
            return {
                'object_key': object_key,
                'lifecycle_status': lifecycle_status,
                'first_detected': current_lifecycle['first_detected'],
                'duration': current_lifecycle['total_duration'],
                'appearance_count': current_lifecycle['appearance_count'],
                'position_stability': self._calculate_position_stability(current_lifecycle),
                'interaction_frequency': self._calculate_interaction_frequency(current_lifecycle),
                'behavioral_pattern': self._infer_behavioral_pattern(current_lifecycle, category)
            }
            
        except Exception as e:
            logger.error(f"객체 생명주기 분석 중 오류: {e}")
            return {'lifecycle_status': 'unknown'}
    
    async def perform_comprehensive_risk_analysis(
        self, object_analysis: Dict, timestamp: float
    ) -> Dict[str, Any]:
        """
        종합적인 위험도 분석
        
        모든 객체 정보를 통합하여 전반적인 주의산만 위험도를 평가합니다.
        """
        try:
            detected_objects = object_analysis.get('detected_objects', [])
            
            if not detected_objects:
                return await self._get_default_risk_analysis()
            
            # 전체 위험도 계산
            overall_risk = self._calculate_overall_risk_score(detected_objects)
            
            # 위험 카테고리 분류
            risk_categories = self._classify_risk_categories(detected_objects)
            
            # 시간적 패턴 분석
            temporal_patterns = self._analyze_temporal_risk_patterns()
            
            # 행동 예측
            behavior_prediction = self._predict_future_behavior(detected_objects, temporal_patterns)
            
            # 우선순위 위험 객체 식별
            priority_objects = self._identify_priority_risk_objects(detected_objects)
            
            # 안전 권장사항 생성
            safety_recommendations = self._generate_safety_recommendations(
                detected_objects, overall_risk, risk_categories
            )
            
            # 트렌드 분석
            risk_trend = self._analyze_risk_trend(timestamp)
            
            return {
                'overall_risk_score': overall_risk,
                'risk_level_category': self._categorize_risk_level(overall_risk),
                'risk_categories': risk_categories,
                'temporal_patterns': temporal_patterns,
                'behavior_prediction': behavior_prediction,
                'priority_risk_objects': priority_objects,
                'safety_recommendations': safety_recommendations,
                'risk_trend': risk_trend,
                'summary': self._generate_risk_summary(overall_risk, priority_objects)
            }
            
        except Exception as e:
            logger.error(f"종합 위험도 분석 중 오류: {e}")
            return await self._get_default_risk_analysis()
    
    # 헬퍼 메서드들
    def _normalize_bounding_box(self, bbox: Any) -> Dict[str, float]:
        """바운딩 박스 정규화"""
        return {
            'x': bbox.origin_x,
            'y': bbox.origin_y,
            'width': bbox.width,
            'height': bbox.height,
            'center_x': bbox.origin_x + bbox.width / 2,
            'center_y': bbox.origin_y + bbox.height / 2,
            'area': bbox.width * bbox.height
        }
    
    def _is_point_in_zone(self, x: float, y: float, zone: Dict) -> bool:
        """점이 구역 내에 있는지 확인"""
        return zone['x1'] <= x <= zone['x2'] and zone['y1'] <= y <= zone['y2']
    
    def _is_hand_over_object(self, hand_center: Dict, bbox: Any) -> bool:
        """손이 객체 위에 있는지 확인"""
        return (bbox.origin_x <= hand_center['x'] <= bbox.origin_x + bbox.width and
                bbox.origin_y <= hand_center['y'] <= bbox.origin_y + bbox.height)
    
    def _analyze_movement_direction(self, hand: Dict, bbox: Any) -> Dict[str, Any]:
        """손의 움직임 방향 분석"""
        # 간단화된 구현 - 실제로는 이전 프레임과 비교 필요
        return {'toward_object': False, 'confidence': 0.0}
    
    def _analyze_category_specific_interaction(
        self, category: str, hand: Dict, bbox: Any, distance: float
    ) -> Dict[str, Any]:
        """카테고리별 특수 상호작용 분석"""
        specific_analysis = {}
        
        if category == 'cell phone':
            specific_analysis = {
                'likely_usage': distance < 0.3 and hand.get('pose_analysis', {}).get('grip_type') == 'grip',
                'screen_orientation': 'unknown',  # 추후 확장 가능
                'calling_gesture': False  # 귀 근처 위치 감지로 확장 가능
            }
        elif category in ['cup', 'bottle']:
            specific_analysis = {
                'drinking_posture': distance < 0.2,
                'holding_style': hand.get('pose_analysis', {}).get('grip_type', 'unknown')
            }
        elif category in ['sandwich', 'food']:
            specific_analysis = {
                'eating_behavior': distance < 0.15,
                'bite_preparation': False  # 입 근처 위치로 확장 가능
            }
        
        return specific_analysis
    
    def _determine_interaction_type(
        self, proximity: float, closest_hand: Dict, category: str, details: List[Dict]
    ) -> str:
        """상호작용 타입 결정"""
        if proximity < 0.2:
            return 'no_interaction'
        elif proximity < 0.5:
            return 'approaching'
        elif proximity < 0.8:
            return 'near_interaction'
        else:
            return 'direct_interaction'
    
    def _calculate_interaction_confidence(self, proximity: float, details: List[Dict]) -> float:
        """상호작용 신뢰도 계산"""
        base_confidence = proximity
        
        # 상세 정보 기반 신뢰도 보정
        detail_factors = [detail.get('is_over_object', False) for detail in details]
        detail_bonus = sum(detail_factors) * 0.1
        
        return min(1.0, base_confidence + detail_bonus)
    
    def _get_category_position_adjustment(self, category: str, position_zone: str) -> float:
        """카테고리별 위치 조정 계수"""
        adjustments = {
            ('cell phone', 'driver_immediate'): 1.5,
            ('cell phone', 'dashboard_center'): 1.3,
            ('cell phone', 'passenger_side'): 0.7,
            ('cup', 'center_console'): 0.8,
            ('book', 'passenger_side'): 0.6,
        }
        
        return adjustments.get((category, position_zone), 1.0)
    
    def _get_position_description(self, zone: str) -> str:
        """위치 설명 반환"""
        descriptions = {
            'driver_immediate': '운전자 즉시 접근 구역',
            'dashboard_center': '대시보드 중앙',
            'center_console': '중앙 콘솔',
            'passenger_side': '조수석 측',
            'neutral': '중립 구역'
        }
        return descriptions.get(zone, '알 수 없는 위치')
    
    def _calculate_object_duration_factor(self, category: str, timestamp: float) -> float:
        """객체 지속시간 기반 가중치"""
        # 최근 이력에서 같은 카테고리 객체의 지속시간 계산
        duration_count = 0
        for record in list(self.detection_history)[-30:]:  # 최근 1초
            if any(obj['category'] == category for obj in record.get('objects', [])):
                duration_count += 1
        
        # 지속시간이 길수록 위험도 증가 (습관적 사용)
        duration_factor = 1.0 + (duration_count / 30.0) * 0.5
        return min(2.0, duration_factor)
    
    def _calculate_object_frequency_factor(self, category: str) -> float:
        """객체 출현 빈도 기반 가중치"""
        # 전체 이력에서 해당 카테고리의 출현 빈도
        total_appearances = sum(
            1 for record in self.detection_history
            if any(obj['category'] == category for obj in record.get('objects', []))
        )
        
        frequency_ratio = total_appearances / len(self.detection_history) if self.detection_history else 0
        
        # 빈도가 높을수록 가중치 증가
        return 1.0 + frequency_ratio * 0.3
    
    def _calculate_multi_object_penalty(self, timestamp: float) -> float:
        """다중 객체 출현 페널티"""
        # 현재 프레임에서 여러 객체가 동시에 나타나면 위험도 증가
        current_object_count = len(self.detection_history[-1].get('objects', [])) if self.detection_history else 0
        
        if current_object_count > 1:
            return 1.0 + (current_object_count - 1) * 0.2
        return 1.0
    
    def _generate_risk_explanation(
        self, category: str, base_risk: float, adjusted_risk: float, 
        duration_factor: float, frequency_factor: float
    ) -> str:
        """위험도 설명 생성"""
        if adjusted_risk > base_risk * 1.3:
            return f"{category} 장시간 사용으로 위험도 증가"
        elif frequency_factor > 1.2:
            return f"{category} 빈번한 사용 패턴 감지"
        elif duration_factor > 1.3:
            return f"{category} 지속적인 주의산만 행동"
        else:
            return f"{category} 일반적인 위험 수준"
    
    def _has_object_moved(self, object_key: str, current_bbox: Any) -> bool:
        """객체 이동 감지"""
        # 간단화된 구현 - 실제로는 이전 위치와 비교
        return False
    
    def _calculate_position_stability(self, lifecycle: Dict) -> float:
        """위치 안정성 계산"""
        if lifecycle['total_duration'] == 0:
            return 1.0
        
        position_changes = lifecycle.get('position_changes', 0)
        stability = max(0.0, 1.0 - position_changes / 10.0)  # 10번 이상 이동시 불안정
        return stability
    
    def _calculate_interaction_frequency(self, lifecycle: Dict) -> float:
        """상호작용 빈도 계산"""
        if lifecycle['total_duration'] == 0:
            return 0.0
        
        interactions = lifecycle.get('interaction_count', 0)
        return interactions / max(1.0, lifecycle['total_duration'])
    
    def _infer_behavioral_pattern(self, lifecycle: Dict, category: str) -> str:
        """행동 패턴 추론"""
        duration = lifecycle.get('total_duration', 0)
        interactions = lifecycle.get('interaction_count', 0)
        
        if duration > 10 and interactions > 5:
            return 'active_engagement'
        elif duration > 5 and interactions < 2:
            return 'passive_presence'
        elif interactions > 3:
            return 'intermittent_use'
        else:
            return 'casual_presence'
    
    def _detect_persistent_risk_behaviors(self) -> float:
        """지속적 위험 행동 감지"""
        if len(self.detection_history) < self.config.distraction.persistent_risk_frames:
            return 0.0
        
        recent_detections = list(self.detection_history)[-self.config.distraction.persistent_risk_frames:]
        risk_frames = sum(
            1 for detection in recent_detections 
            if detection.get('risk_score', 0) > self.config.distraction.persistent_risk_threshold
        )
        
        persistence_ratio = risk_frames / len(recent_detections)
        return persistence_ratio
    
    def _update_detection_history(self, objects: List[Dict], risk_score: float, timestamp: float):
        """감지 이력 업데이트"""
        self.detection_history.append({
            'timestamp': timestamp,
            'objects': objects,
            'risk_score': risk_score,
            'object_count': len(objects)
        })
    
    def _update_object_tracking(self, objects: List[Dict], timestamp: float):
        """객체 추적 정보 업데이트"""
        current_objects = {obj['category']: obj for obj in objects}
        
        # 기존 추적 객체 업데이트
        for category in list(self.tracked_objects.keys()):
            if category in current_objects:
                self.tracked_objects[category]['last_seen'] = timestamp
                self.tracked_objects[category]['frames_tracked'] += 1
            else:
                # 객체가 사라진 경우
                self.tracked_objects[category]['disappeared_at'] = timestamp
        
        # 새로운 객체 추가
        for category, obj_data in current_objects.items():
            if category not in self.tracked_objects:
                self.tracked_objects[category] = {
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'frames_tracked': 1,
                    'disappeared_at': None
                }
    
    def _get_tracking_summary(self) -> Dict[str, Any]:
        """추적 요약 정보"""
        active_objects = {
            k: v for k, v in self.tracked_objects.items() 
            if v.get('disappeared_at') is None
        }
        
        return {
            'total_tracked': len(self.tracked_objects),
            'currently_active': len(active_objects),
            'active_objects': list(active_objects.keys()),
            'longest_tracked': max(
                [v.get('frames_tracked', 0) for v in active_objects.values()], 
                default=0
            )
        }
    
    def _calculate_overall_risk_score(self, objects: List[Dict]) -> float:
        """전체 위험도 점수 계산"""
        if not objects:
            return 0.0
        
        # 가장 높은 위험도와 평균 위험도의 가중 평균
        max_risk = max(obj.get('adjusted_risk_level', obj.get('risk_level', 0)) for obj in objects)
        avg_risk = sum(obj.get('adjusted_risk_level', obj.get('risk_level', 0)) for obj in objects) / len(objects)
        
        # 객체 수에 따른 가중치
        count_multiplier = min(2.0, 1.0 + (len(objects) - 1) * 0.2)
        
        overall_risk = (max_risk * 0.7 + avg_risk * 0.3) * count_multiplier
        return min(1.0, overall_risk)
    
    def _classify_risk_categories(self, objects: List[Dict]) -> Dict[str, List[str]]:
        """위험 카테고리 분류"""
        categories = {
            'high_risk': [],
            'medium_risk': [],
            'low_risk': []
        }
        
        for obj in objects:
            risk = obj.get('adjusted_risk_level', obj.get('risk_level', 0))
            category = obj['category']
            
            if risk > 0.7:
                categories['high_risk'].append(category)
            elif risk > 0.4:
                categories['medium_risk'].append(category)
            else:
                categories['low_risk'].append(category)
        
        return categories
    
    def _analyze_temporal_risk_patterns(self) -> Dict[str, Any]:
        """시간적 위험 패턴 분석"""
        if len(self.detection_history) < 10:
            return {'pattern_detected': False}
        
        recent_risks = [record.get('risk_score', 0) for record in list(self.detection_history)[-10:]]
        
        # 위험도 증가/감소 트렌드
        trend = 'stable'
        if len(recent_risks) >= 5:
            early_avg = sum(recent_risks[:5]) / 5
            late_avg = sum(recent_risks[5:]) / len(recent_risks[5:])
            
            if late_avg > early_avg * 1.2:
                trend = 'increasing'
            elif late_avg < early_avg * 0.8:
                trend = 'decreasing'
        
        return {
            'pattern_detected': True,
            'trend': trend,
            'average_risk': sum(recent_risks) / len(recent_risks),
            'risk_volatility': np.std(recent_risks) if len(recent_risks) > 1 else 0.0
        }
    
    def _predict_future_behavior(self, objects: List[Dict], patterns: Dict) -> Dict[str, Any]:
        """미래 행동 예측"""
        if not objects or not patterns.get('pattern_detected'):
            return {'prediction_available': False}
        
        trend = patterns.get('trend', 'stable')
        current_risk = patterns.get('average_risk', 0)
        
        if trend == 'increasing':
            predicted_action = 'escalating_distraction'
            confidence = 0.7
        elif trend == 'decreasing':
            predicted_action = 'returning_to_focus'
            confidence = 0.6
        else:
            predicted_action = 'maintaining_current_state'
            confidence = 0.8
        
        return {
            'prediction_available': True,
            'predicted_action': predicted_action,
            'confidence': confidence,
            'time_horizon': '10-30 seconds'
        }
    
    def _identify_priority_risk_objects(self, objects: List[Dict]) -> List[Dict]:
        """우선순위 위험 객체 식별"""
        if not objects:
            return []
        
        # 위험도와 상호작용 정도를 기준으로 정렬
        prioritized = sorted(
            objects,
            key=lambda obj: (
                obj.get('adjusted_risk_level', obj.get('risk_level', 0)),
                obj.get('hand_interaction', {}).get('proximity_score', 0)
            ),
            reverse=True
        )
        
        return prioritized[:3]  # 상위 3개만 반환
    
    def _generate_safety_recommendations(self, objects: List[Dict], overall_risk: float, categories: Dict) -> List[str]:
        """안전 권장사항 생성"""
        recommendations = []
        
        if overall_risk > 0.8:
            recommendations.append("즉시 모든 주의산만 객체를 치워주세요")
        elif overall_risk > 0.6:
            recommendations.append("운전에 집중하고 불필요한 물건을 손에서 떼주세요")
        
        if categories['high_risk']:
            high_risk_items = ', '.join(categories['high_risk'])
            recommendations.append(f"고위험 객체 주의: {high_risk_items}")
        
        if len(objects) > 2:
            recommendations.append("한 번에 너무 많은 물건을 다루지 마세요")
        
        return recommendations if recommendations else ["현재 상태가 안전합니다"]
    
    def _categorize_risk_level(self, overall_risk: float) -> str:
        """위험도 레벨 카테고리화"""
        if overall_risk > 0.8:
            return 'critical'
        elif overall_risk > 0.6:
            return 'high'
        elif overall_risk > 0.4:
            return 'medium'
        elif overall_risk > 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _analyze_risk_trend(self, timestamp: float) -> Dict[str, Any]:
        """위험도 트렌드 분석"""
        if len(self.detection_history) < 5:
            return {'trend': 'insufficient_data'}
        
        recent_risks = [record.get('risk_score', 0) for record in list(self.detection_history)[-5:]]
        
        # 선형 회귀로 트렌드 계산 (간단화)
        x = list(range(len(recent_risks)))
        slope = (len(recent_risks) * sum(i * r for i, r in enumerate(recent_risks)) - 
                sum(x) * sum(recent_risks)) / (len(recent_risks) * sum(i*i for i in x) - sum(x)**2)
        
        if slope > 0.1:
            trend = 'increasing'
        elif slope < -0.1:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': slope,
            'recent_average': sum(recent_risks) / len(recent_risks)
        }
    
    def _generate_risk_summary(self, overall_risk: float, priority_objects: List[Dict]) -> str:
        """위험도 요약 생성"""
        risk_level = self._categorize_risk_level(overall_risk)
        
        if risk_level == 'critical':
            summary = "즉각적인 주의 필요"
        elif risk_level == 'high':
            summary = "주의산만 위험 상태"
        elif risk_level == 'medium':
            summary = "경미한 주의산만"
        else:
            summary = "안전한 상태"
        
        if priority_objects:
            top_object = priority_objects[0]['description']
            summary += f" (주요 위험: {top_object})"
        
        return summary
    
    def _update_object_metrics(self, risk_analysis: Dict[str, Any]):
        """객체 관련 메트릭 업데이트"""
        try:
            metrics_data = {
                'distraction_objects_detected': len(risk_analysis.get('priority_risk_objects', [])),
                'overall_object_risk': risk_analysis.get('overall_risk_score', 0.0),
                'object_interaction_level': self._calculate_interaction_level(risk_analysis)
            }
            
            if hasattr(self.metrics_updater, 'update_distraction_metrics'):
                self.metrics_updater.update_distraction_metrics(metrics_data)
                
        except Exception as e:
            logger.error(f"객체 메트릭 업데이트 중 오류: {e}")
    
    def _calculate_interaction_level(self, risk_analysis: Dict) -> float:
        """상호작용 수준 계산"""
        priority_objects = risk_analysis.get('priority_risk_objects', [])
        if not priority_objects:
            return 0.0
        
        interaction_scores = [
            obj.get('hand_interaction', {}).get('proximity_score', 0)
            for obj in priority_objects
        ]
        
        return max(interaction_scores) if interaction_scores else 0.0
    
    async def _handle_no_objects_detected(self, timestamp: float) -> Dict[str, Any]:
        """객체가 감지되지 않은 상황 처리"""
        # 이력에 빈 감지 결과 추가
        self.detection_history.append({
            'timestamp': timestamp,
            'objects': [],
            'risk_score': 0.0,
            'object_count': 0
        })
        
        return {
            'object_detections': await self._get_default_object_analysis(),
            'risk_analysis': await self._get_default_risk_analysis()
        }
    
    async def _get_default_object_analysis(self) -> Dict[str, Any]:
        """기본 객체 분석 데이터"""
        return {
            'detected_objects': [],
            'object_count': 0,
            'immediate_risk_score': 0.0,
            'persistent_risk_score': 0.0,
            'interaction_events': [],
            'tracking_info': {'currently_active': 0}
        }
    
    async def _get_default_risk_analysis(self) -> Dict[str, Any]:
        """기본 위험도 분석 데이터"""
        return {
            'overall_risk_score': 0.0,
            'risk_level_category': 'minimal',
            'risk_categories': {'high_risk': [], 'medium_risk': [], 'low_risk': []},
            'priority_risk_objects': [],
            'safety_recommendations': ["운전에 집중하세요"],
            'summary': "주의산만 객체가 감지되지 않음"
        }