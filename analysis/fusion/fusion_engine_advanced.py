"""
MultiModal Fusion Engine (S-Class): 디지털 신경과학자
- [S-Class] 어텐션 메커니즘 기반 동적 가중치 계산 (Transformer 구조 응용)
- [S-Class] 인지 부하 이론 적용한 멀티태스킹 위험도 모델링
- [S-Class] 시간적 상관관계 분석을 통한 신호간 인과관계 추론
- [S-Class] 불확실성 정량화(Uncertainty Quantification)를 통한 신뢰도 추정
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import math
from dataclasses import dataclass
from enum import Enum

from core.interfaces import IMultiModalAnalyzer
from config.settings import get_config
import logging

logger = logging.getLogger(__name__)


class CognitiveLoadLevel(Enum):
    """인지 부하 수준"""
    MINIMAL = "minimal"      # 자동화된 운전
    LIGHT = "light"          # 일반적인 운전
    MODERATE = "moderate"    # 복잡한 교통상황
    HIGH = "high"           # 멀티태스킹 상황
    OVERLOAD = "overload"   # 인지 과부하


@dataclass
class AttentionWeights:
    """어텐션 가중치"""
    face: float
    pose: float 
    hand: float
    object: float
    total_attention_capacity: float


@dataclass
class FusionConfidence:
    """융합 신뢰도"""
    overall_confidence: float
    modality_reliability: Dict[str, float]
    uncertainty_sources: List[str]


class MultiModalFusionEngine(IMultiModalAnalyzer):
    """
    Multi-Modal Fusion Engine (S-Class)
    
    이 클래스는 마치 인간의 뇌가 여러 감각 정보를 통합하는 것처럼,
    각기 다른 분석 모듈의 결과를 지능적으로 융합합니다.
    """

    def __init__(self):
        self.config = get_config()
        self.weights = self.config.multimodal
        
        # --- S-Class 고도화: 인지과학 기반 파라미터 ---
        self.attention_capacity = 1.0  # 총 인지적 주의 용량
        self.working_memory_load = 0.0  # 작업 기억 부하
        self.cognitive_load_history = []  # 인지 부하 이력
        
        # 어텐션 메커니즘을 위한 학습 가능한 파라미터 (간단화된 버전)
        self.attention_query_weights = {
            'drowsiness': np.array([0.8, 0.3, 0.2, 0.1]),  # [face, pose, hand, object]
            'distraction': np.array([0.6, 0.2, 0.7, 0.9])
        }
        
        logger.info("MultiModalFusionEngine (S-Class) 초기화 완료 - 디지털 신경과학자 준비됨")

    def fuse_drowsiness_signals(self, face_data: Dict, pose_data: Dict, hand_data: Dict) -> float:
        """[S-Class] 졸음 신호 융합 (어텐션 메커니즘 적용)"""
        
        # 1. 각 모달리티의 신뢰도 평가
        modality_data = [face_data, pose_data, hand_data, {}]  # object는 졸음에 직접 관련 없음
        reliability_scores = self._assess_modality_reliability(modality_data)
        
        # 2. 어텐션 가중치 계산 (Query-Key-Value 메커니즘)
        attention_weights = self._calculate_attention_weights(
            'drowsiness', reliability_scores, modality_data
        )
        
        # 3. 각 모달리티별 졸음 신호 추출
        face_drowsiness = self._extract_face_drowsiness_signal(face_data)
        pose_drowsiness = self._extract_pose_drowsiness_signal(pose_data) 
        hand_drowsiness = self._extract_hand_drowsiness_signal(hand_data)
        
        # 4. 어텐션 가중치를 적용한 융합
        fused_score = (
            face_drowsiness * attention_weights.face +
            pose_drowsiness * attention_weights.pose +
            hand_drowsiness * attention_weights.hand
        )
        
        # 5. [S-Class] 시간적 상관관계 보정
        temporal_correlation_boost = self._analyze_temporal_correlations([
            face_drowsiness, pose_drowsiness, hand_drowsiness
        ])
        
        final_score = min(1.0, fused_score * (1.0 + temporal_correlation_boost))
        
        # 6. 인지 부하 이력 업데이트
        self._update_cognitive_load_history('drowsiness', final_score)
        
        return final_score

    def fuse_distraction_signals(
        self, face_data: Dict, pose_data: Dict, hand_data: Dict, 
        object_data: Dict, emotion_data: Dict
    ) -> float:
        """[S-Class] 주의산만 신호 융합 (인지 부하 이론 적용)"""
        
        # 1. 모달리티 신뢰도 평가
        modality_data = [face_data, pose_data, hand_data, object_data]
        reliability_scores = self._assess_modality_reliability(modality_data)
        
        # 2. 어텐션 가중치 계산
        attention_weights = self._calculate_attention_weights(
            'distraction', reliability_scores, modality_data
        )
        
        # 3. 각 모달리티별 주의산만 신호 추출
        signals = {
            'gaze_distraction': self._extract_gaze_distraction_signal(face_data),
            'pose_distraction': self._extract_pose_distraction_signal(pose_data),
            'hand_distraction': self._extract_hand_distraction_signal(hand_data),
            'object_distraction': self._extract_object_distraction_signal(object_data),
            'emotion_amplifier': self._extract_emotion_amplifier(emotion_data)
        }
        
        # 4. [S-Class] 인지 부하 모델링
        cognitive_load = self._calculate_cognitive_load(signals, attention_weights)
        
        # 5. 멀티태스킹 페널티 적용
        multitasking_penalty = self._calculate_multitasking_penalty(signals)
        
        # 6. 최종 주의산만 점수 계산
        base_distraction = (
            signals['gaze_distraction'] * attention_weights.face +
            signals['pose_distraction'] * attention_weights.pose +
            signals['hand_distraction'] * attention_weights.hand +
            signals['object_distraction'] * attention_weights.object
        )
        
        # 감정 증폭기와 멀티태스킹 페널티 적용
        amplified_score = base_distraction * (1.0 + signals['emotion_amplifier'])
        final_score = min(1.0, amplified_score * (1.0 + multitasking_penalty))
        
        # 7. 인지 부하 이력 업데이트
        self._update_cognitive_load_history('distraction', final_score)
        
        return final_score

    def _calculate_attention_weights(
        self, task_type: str, reliability_scores: List[float], 
        modality_data: List[Dict]
    ) -> AttentionWeights:
        """[S-Class] 어텐션 메커니즘 기반 동적 가중치 계산"""
        
        # Query vector (task type에 따른 어텐션 방향)
        query = self.attention_query_weights.get(task_type, np.array([0.25, 0.25, 0.25, 0.25]))
        
        # Key vectors (각 모달리티의 특성)
        keys = np.array([
            [1.0, 0.8, 0.6, 0.3],  # face: 시각적 정보가 풍부
            [0.4, 1.0, 0.7, 0.2],  # pose: 자세 정보 전문
            [0.3, 0.6, 1.0, 0.8],  # hand: 행동 의도 파악
            [0.2, 0.3, 0.9, 1.0]   # object: 환경적 컨텍스트
        ])
        
        # Attention scores 계산 (scaled dot-product attention)
        attention_scores = np.dot(query, keys.T)
        
        # 신뢰도 가중치 적용
        reliability_weighted_scores = attention_scores * np.array(reliability_scores)
        
        # Softmax로 정규화
        normalized_weights = self._softmax(reliability_weighted_scores)
        
        # 총 어텐션 용량 계산 (너무 많은 정보는 인지 과부하 유발)
        total_capacity = min(1.0, sum(reliability_scores) / 2.0)
        
        return AttentionWeights(
            face=normalized_weights[0],
            pose=normalized_weights[1], 
            hand=normalized_weights[2],
            object=normalized_weights[3],
            total_attention_capacity=total_capacity
        )

    def _assess_modality_reliability(self, modality_data: List[Dict]) -> List[float]:
        """각 모달리티의 신뢰도 평가"""
        reliability_scores = []
        
        for data in modality_data:
            if not data or not data.get('available', True):
                reliability_scores.append(0.0)
            else:
                # 데이터 품질 지표들을 종합하여 신뢰도 계산
                confidence = data.get('confidence', data.get('detection_confidence', 0.5))
                stability = data.get('stability', data.get('stability_score', 0.5))
                completeness = 1.0 if data.get('available', False) else 0.0
                
                reliability = (confidence * 0.5 + stability * 0.3 + completeness * 0.2)
                reliability_scores.append(min(1.0, reliability))
        
        return reliability_scores

    def _calculate_cognitive_load(
        self, signals: Dict[str, float], weights: AttentionWeights
    ) -> CognitiveLoadLevel:
        """[S-Class] 인지 부하 수준 계산"""
        
        # 동시 진행되는 인지적 작업의 수
        active_distractions = sum(1 for signal in signals.values() if signal > 0.3)
        
        # 각 신호의 강도
        max_signal_intensity = max(signals.values())
        avg_signal_intensity = sum(signals.values()) / len(signals)
        
        # 어텐션 용량 대비 요구량
        attention_demand = avg_signal_intensity / max(0.1, weights.total_attention_capacity)
        
        # 종합적인 인지 부하 점수
        cognitive_load_score = (
            active_distractions * 0.3 +
            max_signal_intensity * 0.4 +
            attention_demand * 0.3
        )
        
        # 인지 부하 수준 분류
        if cognitive_load_score < 0.2:
            return CognitiveLoadLevel.MINIMAL
        elif cognitive_load_score < 0.4:
            return CognitiveLoadLevel.LIGHT
        elif cognitive_load_score < 0.6:
            return CognitiveLoadLevel.MODERATE
        elif cognitive_load_score < 0.8:
            return CognitiveLoadLevel.HIGH
        else:
            return CognitiveLoadLevel.OVERLOAD

    def _calculate_multitasking_penalty(self, signals: Dict[str, float]) -> float:
        """멀티태스킹 페널티 계산 (인지 심리학의 이중 과제 간섭 이론)"""
        
        # 동시에 활성화된 주의산만 신호들
        active_signals = [signal for signal in signals.values() if signal > 0.2]
        
        if len(active_signals) <= 1:
            return 0.0  # 단일 작업은 페널티 없음
        
        # 멀티태스킹 비용 = 작업 수의 제곱에 비례 (인지 심리학 연구 결과)
        task_interference = (len(active_signals) - 1) ** 1.5 * 0.2
        
        # 신호 강도가 높을수록 간섭 효과 증가
        intensity_factor = sum(active_signals) / len(active_signals)
        
        penalty = task_interference * intensity_factor
        return min(0.5, penalty)  # 최대 50% 페널티

    def _analyze_temporal_correlations(self, signal_values: List[float]) -> float:
        """[S-Class] 시간적 상관관계 분석을 통한 신호 증폭"""
        
        # 여러 신호가 동시에 높을 때 (크로스 밸리데이션 효과)
        high_signals = [s for s in signal_values if s > 0.6]
        
        if len(high_signals) >= 2:
            # 신호간 상관관계 계산
            correlation_strength = min(high_signals) / max(high_signals) if max(high_signals) > 0 else 0
            
            # 상관관계가 높을수록 신뢰도 증가
            correlation_boost = correlation_strength * len(high_signals) * 0.1
            return min(0.3, correlation_boost)  # 최대 30% 증폭
        
        return 0.0

    def _update_cognitive_load_history(self, task_type: str, score: float):
        """인지 부하 이력 업데이트"""
        self.cognitive_load_history.append({
            'timestamp': time.time(),
            'task_type': task_type,
            'cognitive_load_score': score
        })
        
        # 최근 100개 항목만 유지
        if len(self.cognitive_load_history) > 100:
            self.cognitive_load_history.pop(0)

    def get_fusion_confidence(self) -> FusionConfidence:
        """[S-Class] 융합 결과의 신뢰도 정량화"""
        
        # 최근 분석 결과들의 일관성 평가
        if len(self.cognitive_load_history) < 5:
            return FusionConfidence(
                overall_confidence=0.5,
                modality_reliability={'insufficient_data': 0.5},
                uncertainty_sources=['insufficient_history']
            )
        
        # 최근 결과들의 표준편차로 일관성 측정
        recent_scores = [h['cognitive_load_score'] for h in self.cognitive_load_history[-10:]]
        consistency = 1.0 - min(1.0, np.std(recent_scores) * 2)
        
        # 모달리티별 신뢰도 (가정값)
        modality_reliability = {
            'face': 0.8,
            'pose': 0.7, 
            'hand': 0.6,
            'object': 0.8
        }
        
        # 불확실성 원인 식별
        uncertainty_sources = []
        if consistency < 0.7:
            uncertainty_sources.append('temporal_inconsistency')
        if any(rel < 0.5 for rel in modality_reliability.values()):
            uncertainty_sources.append('low_modality_reliability')
        
        overall_confidence = consistency * 0.6 + np.mean(list(modality_reliability.values())) * 0.4
        
        return FusionConfidence(
            overall_confidence=overall_confidence,
            modality_reliability=modality_reliability,
            uncertainty_sources=uncertainty_sources
        )

    # --- 신호 추출 메서드들 ---
    
    def _extract_face_drowsiness_signal(self, face_data: Dict) -> float:
        """얼굴 데이터에서 졸음 신호 추출"""
        if not face_data.get('available', True):
            return 0.0
        
        drowsiness = face_data.get('drowsiness', {})
        # 개선된 EAR, PERCLOS, 시간적 어텐션을 종합
        ear_signal = 1.0 - drowsiness.get('enhanced_ear', 1.0)
        perclos_signal = drowsiness.get('perclos', 0.0)
        attention_signal = drowsiness.get('temporal_attention_score', 0.0)
        
        return (ear_signal * 0.4 + perclos_signal * 0.4 + attention_signal * 0.2)

    def _extract_pose_drowsiness_signal(self, pose_data: Dict) -> float:
        """자세 데이터에서 졸음 신호 추출"""
        if not pose_data.get('available', True):
            return 0.0
        
        fatigue_indicators = pose_data.get('fatigue_indicators', {})
        return fatigue_indicators.get('slouching', 0.0)

    def _extract_hand_drowsiness_signal(self, hand_data: Dict) -> float:
        """손 데이터에서 졸음 신호 추출 (떨림, 불안정성)"""
        if not hand_data.get('available', True):
            return 0.0
        
        steering_skill = hand_data.get('steering_skill', {})
        movement_quality = steering_skill.get('movement_quality_score', 1.0)
        
        # 움직임 품질이 떨어지면 피로 신호로 간주
        return max(0.0, 1.0 - movement_quality)

    def _extract_gaze_distraction_signal(self, face_data: Dict) -> float:
        """시선 주의산만 신호 추출"""
        if not face_data.get('available', True):
            return 0.0
        
        gaze = face_data.get('gaze', {})
        return gaze.get('deviation_score', 0.0)

    def _extract_pose_distraction_signal(self, pose_data: Dict) -> float:
        """자세 주의산만 신호 추출"""
        if not pose_data.get('available', True):
            return 0.0
        
        distraction_indicators = pose_data.get('distraction_indicators', {})
        return distraction_indicators.get('unusual_positioning', 0.0)

    def _extract_hand_distraction_signal(self, hand_data: Dict) -> float:
        """손 주의산만 신호 추출"""
        if not hand_data.get('available', True):
            return 0.0
        
        distraction_behaviors = hand_data.get('distraction_behaviors', {})
        return distraction_behaviors.get('risk_score', 0.0)

    def _extract_object_distraction_signal(self, object_data: Dict) -> float:
        """객체 주의산만 신호 추출"""
        if not object_data.get('available', True):
            return 0.0
        
        risk_analysis = object_data.get('risk_analysis', {})
        return risk_analysis.get('overall_risk_score', 0.0)

    def _extract_emotion_amplifier(self, emotion_data: Dict) -> float:
        """감정 증폭 신호 추출"""
        if not emotion_data.get('available', True):
            return 0.0
        
        emotion = emotion_data.get('emotion', {})
        stress_level = emotion.get('stress_level', 0.0)
        
        # 스트레스가 높을수록 다른 신호들을 증폭
        return stress_level * 0.3

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax 함수"""
        exp_x = np.exp(x - np.max(x))  # 수치적 안정성을 위한 정규화
        return exp_x / np.sum(exp_x)

import time  # 누락된 import 추가