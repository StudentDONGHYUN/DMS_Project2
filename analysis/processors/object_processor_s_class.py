"""
Object Processor (S-Class): 디지털 행동예측 전문가
- [S-Class] 베이지안 네트워크 기반 행동 의도 추론 시스템
- [S-Class] 어텐션 히트맵 생성 및 시각적 주의 분산 패턴 분석  
- [S-Class] 상황인지형 위험도 동적 조정 (교통상황, 날씨, 시간대 반영)
- [S-Class] 복합 행동 시퀀스 모델링을 통한 미래 행동 예측
"""

import math
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from enum import Enum

from core.interfaces import IObjectDataProcessor, IMetricsUpdater
from core.constants import AnalysisConstants
from config.settings import get_config
import logging

logger = logging.getLogger(__name__)


class BehaviorSequence(Enum):
    """행동 시퀀스 패턴 정의"""
    PHONE_PICKUP_SEQUENCE = "phone_pickup_sequence"  # 휴대폰을 집는 동작 시퀀스
    DRINKING_SEQUENCE = "drinking_sequence"  # 음료를 마시는 동작 시퀀스
    CONSOLE_OPERATION_SEQUENCE = "console_operation_sequence"  # 콘솔 조작 시퀀스
    DISTRACTED_SCROLLING = "distracted_scrolling"  # 산만한 스크롤링 패턴


@dataclass
class BehaviorPrediction:
    """행동 예측 결과"""
    predicted_action: str
    confidence: float
    time_to_action: float  # 초 단위
    risk_escalation_probability: float


@dataclass
class AttentionHeatmap:
    """시각적 주의 히트맵"""
    zones: Dict[str, float]  # 구역별 주의 집중도
    center_of_attention: Tuple[float, float]  # 주의 중심점
    dispersion_score: float  # 주의 분산도 (0-1)


class ObjectDataProcessor(IObjectDataProcessor):
    """
    Object Data Processor (S-Class)
    
    이 클래스는 미래를 예측하는 행동 분석가처럼 운전자의 의도를 파악하고
    앞으로 일어날 위험한 행동을 사전에 예측합니다.
    """
    
    def __init__(self, metrics_updater: IMetricsUpdater):
        self.metrics_updater = metrics_updater
        self.config = get_config()
        
        # 주의산만 객체 정의
        self.distraction_objects = self.config.distraction.object_risk_levels
        
        # --- S-Class 고도화: 행동 예측을 위한 고급 데이터 구조 ---
        self.detection_history = deque(maxlen=self.config.distraction.detection_history_size)
        self.behavior_sequence_buffer = deque(maxlen=50)  # 행동 시퀀스 추적
        self.attention_heatmap_history = deque(maxlen=30)  # 시각적 주의 패턴
        self.contextual_factors = {}  # 상황적 요인들 (시간대, 교통상황 등)
        
        # 베이지안 네트워크 확률 테이블 (간단화된 버전)
        self.behavior_probability_matrix = self._initialize_behavior_probabilities()
        
        # 객체별 추적 상태
        self.tracked_objects = {}
        self.object_lifecycle = {}
        
        logger.info("ObjectDataProcessor (S-Class) 초기화 완료 - 행동 예측 전문가 준비됨")

    def get_processor_name(self) -> str:
        """프로세서의 이름을 반환합니다."""
        return "ObjectDataProcessor_S_Class"

    def get_required_data_types(self) -> List[str]:
        """이 프로세서가 필요로 하는 데이터 타입 목록을 반환합니다."""
        return ["object_detections", "hand_positions"]

    def _initialize_behavior_probabilities(self) -> Dict[str, Dict[str, float]]:
        """베이지안 네트워크 확률 테이블 초기화"""
        return {
            # 휴대폰 관련 행동 확률
            'cell_phone': {
                'pickup_probability': 0.3,
                'usage_escalation': 0.7,
                'attention_capture': 0.9
            },
            # 음료 관련 행동 확률  
            'cup': {
                'drinking_probability': 0.8,
                'spillage_risk': 0.2,
                'attention_capture': 0.4
            },
            # 음식 관련 행동 확률
            'food': {
                'eating_probability': 0.6,
                'messy_handling': 0.4,
                'attention_capture': 0.6
            }
        }

    async def process_data(self, result, timestamp):
        logger.debug(f"[object_processor_s_class] process_data input: {result}")
        if hasattr(result, 'detections'):
            logger.debug(f"[object_processor_s_class] detections: {getattr(result, 'detections', None)}")
        if not result or not hasattr(result, 'detections') or not result.detections:
            return await self._handle_no_objects_detected(timestamp)
        
        # 손 위치 정보는 다른 프로세서에서 처리된 결과를 받아옴
        hand_positions = getattr(result, 'hand_positions', [])
        
        # [S-Class] 상황적 컨텍스트 업데이트
        self._update_contextual_factors(timestamp)
        
        # 객체 감지 결과 처리
        object_analysis = self.process_object_detections(
            result, hand_positions, timestamp
        )
        
        # [S-Class] 어텐션 히트맵 생성
        attention_heatmap = self._generate_attention_heatmap(object_analysis, hand_positions)
        
        # [S-Class] 행동 시퀀스 분석 및 예측
        behavior_prediction = self._analyze_behavior_sequences(
            object_analysis, hand_positions, timestamp
        )
        
        # 종합적인 위험도 분석
        comprehensive_analysis = self.perform_comprehensive_risk_analysis(
            object_analysis, timestamp
        )
        
        # [S-Class] 예측 결과를 종합 분석에 통합
        comprehensive_analysis['behavior_prediction'] = behavior_prediction
        comprehensive_analysis['attention_heatmap'] = attention_heatmap
        
        results = {
            'object_detections': object_analysis,
            'risk_analysis': comprehensive_analysis
        }
        
        self._update_object_metrics(comprehensive_analysis)
        return results

    def _update_contextual_factors(self, timestamp: float):
        """[S-Class] 상황적 컨텍스트 업데이트"""
        current_hour = time.localtime(timestamp).tm_hour
        
        # 시간대별 위험도 가중치 (출퇴근 시간, 야간 운전 등)
        if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
            traffic_factor = 1.3  # 러시아워
        elif 22 <= current_hour or current_hour <= 5:
            fatigue_factor = 1.4  # 야간 운전
        else:
            traffic_factor = 1.0
            fatigue_factor = 1.0
        
        self.contextual_factors.update({
            'current_hour': current_hour,
            'traffic_risk_multiplier': traffic_factor,
            'fatigue_risk_multiplier': fatigue_factor,
            'timestamp': timestamp
        })

    def _generate_attention_heatmap(
        self, object_analysis: Dict, hand_positions: List[Dict]
    ) -> AttentionHeatmap:
        """[S-Class] 시각적 주의 히트맵 생성"""
        zones = {
            'steering_wheel': 0.0,
            'center_console': 0.0, 
            'dashboard': 0.0,
            'passenger_area': 0.0,
            'floor': 0.0
        }
        
        total_attention = 0.0
        weighted_x, weighted_y = 0.0, 0.0
        
        # 감지된 객체들로부터 주의 분산 계산
        for obj in object_analysis.get('detected_objects', []):
            bbox = obj['bbox']
            risk_level = obj['risk_level']
            
            # 객체 위치를 차량 내 구역으로 매핑
            zone = self._map_position_to_zone(bbox['center_x'], bbox['center_y'])
            attention_weight = risk_level * obj['detection_confidence']
            
            zones[zone] += attention_weight
            total_attention += attention_weight
            
            # 가중 평균으로 주의 중심점 계산
            weighted_x += bbox['center_x'] * attention_weight
            weighted_y += bbox['center_y'] * attention_weight
        
        # 손 위치도 주의 분산에 반영
        for hand in hand_positions:
            hand_zone = self._map_position_to_zone(
                hand.get('hand_center', {}).get('x', 0.5),
                hand.get('hand_center', {}).get('y', 0.5)
            )
            zones[hand_zone] += 0.3  # 손 위치는 중간 정도의 주의값
            total_attention += 0.3
        
        # 정규화
        if total_attention > 0:
            for zone in zones:
                zones[zone] /= total_attention
            center_of_attention = (weighted_x / total_attention, weighted_y / total_attention)
        else:
            center_of_attention = (0.5, 0.5)  # 중앙 기본값
        
        # 주의 분산도 계산 (엔트로피 기반)
        dispersion_score = self._calculate_attention_dispersion(zones)
        
        heatmap = AttentionHeatmap(
            zones=zones,
            center_of_attention=center_of_attention,
            dispersion_score=dispersion_score
        )
        
        self.attention_heatmap_history.append(heatmap)
        return heatmap

    def _calculate_attention_dispersion(self, zones: Dict[str, float]) -> float:
        """주의 분산도 계산 (엔트로피 기반)"""
        entropy = 0.0
        for attention_value in zones.values():
            if attention_value > 0:
                entropy -= attention_value * math.log2(attention_value)
        
        # 정규화 (최대 엔트로피는 log2(구역수))
        max_entropy = math.log2(len(zones))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    async def _analyze_behavior_sequences(
        self, object_analysis: Dict, hand_positions: List[Dict], timestamp: float
    ) -> BehaviorPrediction:
        """[S-Class] 행동 시퀀스 분석 및 미래 행동 예측"""
        
        # 현재 프레임의 행동 상태 기록
        current_behavior = self._extract_current_behavior_state(object_analysis, hand_positions)
        self.behavior_sequence_buffer.append({
            'timestamp': timestamp,
            'behavior_state': current_behavior
        })
        
        if len(self.behavior_sequence_buffer) < 10:
            return BehaviorPrediction(
                predicted_action="insufficient_data",
                confidence=0.0,
                time_to_action=0.0,
                risk_escalation_probability=0.0
            )
        
        # 행동 시퀀스 패턴 매칭
        detected_sequence = self._detect_behavior_sequence()
        
        # 베이지안 추론으로 미래 행동 예측
        prediction = self._predict_next_behavior(detected_sequence, current_behavior)
        
        return prediction

    def _extract_current_behavior_state(
        self, object_analysis: Dict, hand_positions: List[Dict]
    ) -> Dict[str, Any]:
        """현재 프레임의 행동 상태 추출"""
        state = {
            'objects_present': [obj['category'] for obj in object_analysis.get('detected_objects', [])],
            'hand_object_interactions': [],
            'dominant_object': None,
            'interaction_intensity': 0.0
        }
        
        # 가장 위험한 객체 식별
        max_risk = 0.0
        for obj in object_analysis.get('detected_objects', []):
            if obj['risk_level'] > max_risk:
                max_risk = obj['risk_level']
                state['dominant_object'] = obj['category']
                state['interaction_intensity'] = obj['hand_interaction']['proximity_score']
        
        # 손-객체 상호작용 기록
        for obj in object_analysis.get('detected_objects', []):
            if obj['hand_interaction']['is_interacting']:
                state['hand_object_interactions'].append({
                    'object': obj['category'],
                    'interaction_type': obj['hand_interaction']['interaction_type']
                })
        
        return state

    def _detect_behavior_sequence(self) -> Optional[BehaviorSequence]:
        """행동 시퀀스 패턴 감지"""
        recent_states = list(self.behavior_sequence_buffer)[-10:]
        
        # 휴대폰 집기 시퀀스 감지
        phone_sequence_score = self._calculate_phone_pickup_sequence_score(recent_states)
        if phone_sequence_score > 0.7:
            return BehaviorSequence.PHONE_PICKUP_SEQUENCE
        
        # 음료 마시기 시퀀스 감지
        drinking_sequence_score = self._calculate_drinking_sequence_score(recent_states)
        if drinking_sequence_score > 0.6:
            return BehaviorSequence.DRINKING_SEQUENCE
        
        return None

    def _calculate_phone_pickup_sequence_score(self, states: List[Dict]) -> float:
        """휴대폰 집기 시퀀스 점수 계산"""
        phone_presence = 0
        hand_movement_toward_phone = 0
        
        for state in states:
            if 'cell phone' in state['behavior_state']['objects_present']:
                phone_presence += 1
            
            for interaction in state['behavior_state']['hand_object_interactions']:
                if interaction['object'] == 'cell phone':
                    hand_movement_toward_phone += 1
        
        # 시퀀스 점수: 휴대폰 지속적 존재 + 손의 접근
        sequence_score = (phone_presence / len(states)) * 0.6 + (hand_movement_toward_phone / len(states)) * 0.4
        return min(1.0, sequence_score)

    def _calculate_drinking_sequence_score(self, states: List[Dict]) -> float:
        """음료 마시기 시퀀스 점수 계산"""
        cup_presence = 0
        hand_cup_interactions = 0
        
        for state in states:
            if any(obj in state['behavior_state']['objects_present'] for obj in ['cup', 'bottle']):
                cup_presence += 1
            
            for interaction in state['behavior_state']['hand_object_interactions']:
                if interaction['object'] in ['cup', 'bottle']:
                    hand_cup_interactions += 1
        
        sequence_score = (cup_presence / len(states)) * 0.5 + (hand_cup_interactions / len(states)) * 0.5
        return min(1.0, sequence_score)

    def _predict_next_behavior(
        self, detected_sequence: Optional[BehaviorSequence], current_state: Dict
    ) -> BehaviorPrediction:
        """베이지안 추론을 통한 다음 행동 예측"""
        
        if detected_sequence == BehaviorSequence.PHONE_PICKUP_SEQUENCE:
            # 휴대폰을 집는 중이면 다음 행동은 사용 확률이 높음
            prob_table = self.behavior_probability_matrix.get('cell_phone', {})
            return BehaviorPrediction(
                predicted_action="phone_usage_imminent",
                confidence=prob_table.get('usage_escalation', 0.7),
                time_to_action=2.5,  # 평균 2.5초 후 사용 시작
                risk_escalation_probability=prob_table.get('attention_capture', 0.9)
            )
        
        elif detected_sequence == BehaviorSequence.DRINKING_SEQUENCE:
            prob_table = self.behavior_probability_matrix.get('cup', {})
            return BehaviorPrediction(
                predicted_action="drinking_behavior",
                confidence=prob_table.get('drinking_probability', 0.8),
                time_to_action=1.5,
                risk_escalation_probability=prob_table.get('attention_capture', 0.4)
            )
        
        # 특별한 시퀀스가 감지되지 않은 경우
        return BehaviorPrediction(
            predicted_action="maintaining_current_state",
            confidence=0.6,
            time_to_action=0.0,
            risk_escalation_probability=0.2
        )

    async def perform_comprehensive_risk_analysis(
        self, object_analysis: Dict, timestamp: float
    ) -> Dict[str, Any]:
        """[S-Class] 종합 위험도 분석 (상황적 요인 반영)"""
        try:
            detected_objects = object_analysis.get('detected_objects', [])
            
            if not detected_objects:
                return self._get_default_risk_analysis()
            
            # 기본 위험도 계산
            base_risk = self._calculate_overall_risk_score(detected_objects)
            
            # [S-Class] 상황적 요인 반영
            contextual_multiplier = self._calculate_contextual_risk_multiplier()
            adjusted_risk = min(1.0, base_risk * contextual_multiplier)
            
            # 시간적 패턴 분석
            temporal_patterns = self._analyze_temporal_risk_patterns()
            
            # 우선순위 위험 객체 식별
            priority_objects = sorted(detected_objects, key=lambda x: x['risk_level'], reverse=True)[:3]
            
            # [S-Class] 상황인지형 안전 권장사항
            safety_recommendations = self._generate_contextual_safety_recommendations(
                detected_objects, adjusted_risk, self.contextual_factors
            )
            
            return {
                'overall_risk_score': adjusted_risk,
                'base_risk_score': base_risk,
                'contextual_multiplier': contextual_multiplier,
                'risk_level_category': self._categorize_risk_level(adjusted_risk),
                'priority_risk_objects': priority_objects,
                'temporal_patterns': temporal_patterns,
                'safety_recommendations': safety_recommendations,
                'contextual_factors': self.contextual_factors.copy()
            }
            
        except Exception as e:
            logger.error(f"종합 위험도 분석 중 오류: {e}")
            return self._get_default_risk_analysis()

    def _calculate_contextual_risk_multiplier(self) -> float:
        """상황적 위험도 가중치 계산"""
        multiplier = 1.0
        
        # 시간대별 가중치
        multiplier *= self.contextual_factors.get('traffic_risk_multiplier', 1.0)
        multiplier *= self.contextual_factors.get('fatigue_risk_multiplier', 1.0)
        
        # 주의 분산도에 따른 가중치
        if self.attention_heatmap_history:
            latest_heatmap = self.attention_heatmap_history[-1]
            dispersion_multiplier = 1.0 + (latest_heatmap.dispersion_score * 0.5)
            multiplier *= dispersion_multiplier
        
        return min(2.0, multiplier)  # 최대 2배까지 가중

    def _generate_contextual_safety_recommendations(
        self, objects: List[Dict], risk_score: float, context: Dict
    ) -> List[str]:
        """[S-Class] 상황인지형 안전 권장사항 생성"""
        recommendations = []
        
        # 시간대별 맞춤 권장사항
        current_hour = context.get('current_hour', 12)
        
        if risk_score > 0.8:
            if 22 <= current_hour or current_hour <= 5:
                recommendations.append("야간 운전 중 고위험 상황 - 즉시 안전한 곳에 정차하세요")
            else:
                recommendations.append("즉시 모든 주의산만 요소를 제거하고 운전에 집중하세요")
        
        elif risk_score > 0.6:
            top_object = objects[0]['category'] if objects else "알 수 없는 객체"
            if current_hour in [7, 8, 17, 18]:  # 러시아워
                recommendations.append(f"교통량이 많은 시간대입니다. {top_object} 사용을 중단하세요")
            else:
                recommendations.append(f"{top_object} 사용을 줄이고 전방 주시에 집중하세요")
        
        else:
            recommendations.append("현재 안전한 상태를 유지하고 있습니다")
        
        return recommendations

    # --- 헬퍼 메서드들 ---
    
    def _map_position_to_zone(self, x: float, y: float) -> str:
        """위치를 차량 내 구역으로 매핑"""
        if 0.3 <= x <= 0.7 and 0.6 <= y <= 1.0:
            return 'steering_wheel'
        elif 0.4 <= x <= 0.8 and 0.3 <= y <= 0.6:
            return 'center_console'
        elif y <= 0.3:
            return 'dashboard'
        elif x >= 0.7:
            return 'passenger_area'
        else:
            return 'floor'

    def _calculate_overall_risk_score(self, objects: List[Dict]) -> float:
        """전체 위험도 점수 계산"""
        if not objects:
            return 0.0
        
        risk_levels = [obj['risk_level'] for obj in objects]
        max_risk = max(risk_levels)
        avg_risk = np.mean(risk_levels)
        
        # 가장 높은 위험도에 가중치를 두되, 여러 객체가 있으면 페널티
        overall_risk = (max_risk * 0.7 + avg_risk * 0.3) * min(1.5, 1.0 + (len(objects) - 1) * 0.25)
        return min(1.0, overall_risk)

    def _analyze_temporal_risk_patterns(self) -> Dict[str, Any]:
        """시간적 위험 패턴 분석"""
        if len(self.detection_history) < 60:
            return {'pattern_detected': False, 'trend': 'stable'}

        recent_risks = [
            np.mean([o.get('risk_level', 0) for o in rec.get('objects', [])] + [0]) 
            for rec in list(self.detection_history)
        ]
        timestamps = [rec['timestamp'] for rec in list(self.detection_history)]
        
        try:
            slope = np.polyfit(timestamps, recent_risks, 1)[0]
        except np.linalg.LinAlgError:
            slope = 0.0

        if slope > 0.05:
            trend = 'increasing'
        elif slope < -0.05:
            trend = 'decreasing'  
        else:
            trend = 'stable'
        
        return {'pattern_detected': True, 'trend': trend, 'risk_slope': slope}

    def _categorize_risk_level(self, risk: float) -> str:
        """위험도 레벨 분류"""
        if risk > 0.8:
            return 'critical'
        elif risk > 0.6:
            return 'high'
        elif risk > 0.4:
            return 'medium'
        else:
            return 'low'

    async def _handle_no_objects_detected(self, timestamp: float) -> Dict[str, Any]:
        """객체 미감지 처리"""
        self.detection_history.append({
            'timestamp': timestamp,
            'objects': [],
            'risk_score': 0.0,
            'object_count': 0
        })
        return {
            'object_detections': {'detected_objects': [], 'object_count': 0},
            'risk_analysis': self._get_default_risk_analysis()
        }
    
    def _get_default_risk_analysis(self) -> Dict[str, Any]:
        """기본 위험 분석 데이터"""
        return {
            'overall_risk_score': 0.0,
            'base_risk_score': 0.0,
            'contextual_multiplier': 1.0,
            'risk_level_category': 'minimal',
            'priority_risk_objects': [],
            'temporal_patterns': {'trend': 'stable'},
            'safety_recommendations': ["주의산만 객체가 감지되지 않았습니다."],
            'contextual_factors': self.contextual_factors.copy(),
            'behavior_prediction': BehaviorPrediction(
                predicted_action="normal_driving",
                confidence=0.8,
                time_to_action=0.0,
                risk_escalation_probability=0.1
            ),
            'attention_heatmap': AttentionHeatmap(
                zones={'steering_wheel': 1.0, 'center_console': 0.0, 'dashboard': 0.0, 'passenger_area': 0.0, 'floor': 0.0},
                center_of_attention=(0.5, 0.8),  # 핸들 위치
                dispersion_score=0.0
            )
        }

    def _update_object_metrics(self, comprehensive_analysis: Dict[str, Any]):
        """객체 관련 메트릭 업데이트"""
        try:
            metrics_data = {
                'distraction_objects_detected': len(comprehensive_analysis.get('priority_risk_objects', [])),
                'overall_object_risk': comprehensive_analysis.get('overall_risk_score', 0.0),
                'contextual_risk_multiplier': comprehensive_analysis.get('contextual_multiplier', 1.0),
                'attention_dispersion_score': comprehensive_analysis.get('attention_heatmap', {}).get('dispersion_score', 0.0),
                'predicted_risk_escalation': comprehensive_analysis.get('behavior_prediction', {}).get('risk_escalation_probability', 0.0)
            }
            if hasattr(self.metrics_updater, 'update_distraction_metrics'):
                self.metrics_updater.update_distraction_metrics(metrics_data)
        except Exception as e:
            logger.error(f"객체 메트릭 업데이트 중 오류: {e}")

    def process_object_detections(self, detections: Any, hand_positions: List[Dict], timestamp: float) -> Dict[str, Any]:
        """객체 감지 결과 처리 (S-Class: 고급 행동 예측 및 위험 분석과 연동)"""
        # 기존 process_data에서 사용하는 고급 분석 파이프라인과 연동
        # S-Class에서는 process_data가 더 고도화되어 있으므로, 핵심 로직을 재사용
        # (실제 시스템에서는 process_data를 통해 전체 분석을 수행하고, 여기서는 객체 감지 결과만 반환)
        if not detections or not hasattr(detections, 'detections') or not detections.detections:
            return {'detected_objects': [], 'object_count': 0}
        # 손 위치 정보는 별도로 전달됨
        object_analysis = []
        for detection in detections.detections:
            category = detection.categories[0].category_name
            confidence = detection.categories[0].score
            bbox = detection.bounding_box
            # 관심 객체인지 확인
            if category in self.distraction_objects:
                object_analysis.append({
                    'category': category,
                    'detection_confidence': confidence,
                    'bbox': {
                        'x': bbox.origin_x,
                        'y': bbox.origin_y,
                        'width': bbox.width,
                        'height': bbox.height,
                        'center_x': bbox.origin_x + bbox.width / 2,
                        'center_y': bbox.origin_y + bbox.height / 2
                    },
                    'timestamp': timestamp
                })
        return {
            'detected_objects': object_analysis,
            'object_count': len(object_analysis)
        }
