"""
Pose Processor (S-Class): 디지털 생체역학 전문가
- [S-Class] 3D 랜드마크 기반 척추 정렬 및 목 각도(거북목) 분석
- [S-Class] 몸통 중심의 미세 흔들림(Postural Sway) 측정을 통한 피로도 분석
- [S-Class] 생체역학적 지표를 통합한 종합 운전 자세 건강 점수
- [S-Class] 시간의 흐름에 따른 자세 변화 추세 분석 기능 추가
"""

import numpy as np
import math
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from core.interfaces import IPoseDataProcessor, IMetricsUpdater
from core.constants import MediaPipeConstants, AnalysisConstants, MathConstants
from config.settings import get_config
import logging

logger = logging.getLogger(__name__)


class PoseDataProcessor(IPoseDataProcessor):
    """
    자세 데이터 전문 처리기 (S-Class)
    
    이 클래스는 마치 물리치료사나 자세 교정 전문가처럼 운전자의 몸 자세를 분석합니다.
    - 어깨와 몸통의 방향성 분석
    - 자세의 복잡도와 안정성 평가
    - 구부정한 자세(slouch) 감지 + 척추 건강도 평가
    - 몸의 균형과 대칭성 분석
    - [S-Class] 3D 기반 척추 정렬 및 거북목 분석
    - [S-Class] 자세 불안정성(Postural Sway) 정량 측정
    - [S-Class] 자세 변화 추세 예측 및 건강 권고
    """
    
    def __init__(self, metrics_updater: IMetricsUpdater):
        self.metrics_updater = metrics_updater
        self.config = get_config()
        
        # --- 고도화: 상태 추적 변수 추가 ---
        self.posture_history = deque(maxlen=300) # 10초(30fps*10) 이력
        self.torso_center_history = deque(maxlen=150) # 5초간 몸통 중심 추적

        logger.info("PoseDataProcessor (S-Class) 초기화 완료 - 자세 분석 전문가 준비됨")
    
    def get_processor_name(self) -> str:
        return "PoseDataProcessor"
    
    def get_required_data_types(self) -> List[str]:
        return ["pose_landmarks", "pose_world_landmarks"]
    
    async def process_data(self, result, timestamp):
        logger.debug(f"[pose_processor_s_class] process_data input: {result}")
        if hasattr(result, 'pose_landmarks'):
            logger.debug(f"[pose_processor_s_class] pose_landmarks: {getattr(result, 'pose_landmarks', None)}")
        if not result or not result.pose_landmarks:
            return await self._handle_no_pose_detected()
        
        pose_result = result
        results = {}
        
        if pose_result.pose_landmarks:
            results['pose_2d'] = await self.process_pose_landmarks(pose_result.pose_landmarks[0], timestamp)
        
        if pose_result.pose_world_landmarks:
            results['pose_3d'] = await self.process_world_landmarks(pose_result.pose_world_landmarks[0], timestamp)
        
        # [고도화] 2D와 3D 분석을 종합한 최종 분석
        comprehensive_analysis = await self.perform_comprehensive_pose_analysis(results, timestamp)
        results['pose_analysis'] = comprehensive_analysis
        
        # 메트릭 업데이트
        self._update_pose_metrics(comprehensive_analysis)
        
        return results
    
    async def process_pose_landmarks(self, landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """ 2D 자세 랜드마크 처리 """
        try:
            key_points = self._extract_key_pose_points(landmarks)
            shoulder_analysis = self._analyze_shoulder_orientation(key_points)
            symmetry_analysis = self._analyze_body_symmetry(key_points)
            
            return {
                'available': True,
                'shoulder_analysis': shoulder_analysis,
                'symmetry_analysis': symmetry_analysis,
            }
        except Exception as e:
            logger.error(f"2D 자세 분석 중 오류 발생: {e}")
            return {'available': False}

    async def process_world_landmarks(self, world_landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """ 3D 월드 랜드마크 처리 """
        try:
            # [S-Class] 새로운 분석 추가
            spinal_analysis = self._analyze_spinal_alignment(world_landmarks)
            postural_sway = self._analyze_postural_sway(world_landmarks, timestamp)
            
            slouch_analysis = self._detect_slouching_posture(world_landmarks)
            balance_analysis = self._analyze_3d_body_balance(world_landmarks)
            complexity_analysis = self._calculate_pose_complexity(world_landmarks)
            
            return {
                'available': True,
                'spinal_analysis': spinal_analysis,
                'postural_sway': postural_sway,
                'slouch_detection': slouch_analysis,
                'balance': balance_analysis,
                'complexity': complexity_analysis,
            }
        except Exception as e:
            logger.error(f"3D 자세 분석 중 오류 발생: {e}")
            return {'available': False}

    async def perform_comprehensive_pose_analysis(self, results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """ [고도화] 종합 자세 분석 """
        pose_2d = results.get('pose_2d', {'available': False})
        pose_3d = results.get('pose_3d', {'available': False})
        
        # 1. 운전 자세 적합성 평가
        suitability, issues = self._evaluate_driving_posture_suitability(pose_2d, pose_3d)
        
        # 2. 피로 및 주의산만 지표 추출
        fatigue_indicators = self._extract_fatigue_indicators_from_pose(pose_3d)
        distraction_indicators = self._extract_distraction_indicators_from_pose(pose_3d)

        # 3. 분석 결과를 이력에 저장
        current_pose_summary = {
            'timestamp': timestamp,
            'suitability': suitability,
            'slouch_factor': pose_3d.get('slouch_detection', {}).get('slouch_factor', 0),
            'complexity': pose_3d.get('complexity', {}).get('overall_complexity', 0),
            'spinal_health': pose_3d.get('spinal_analysis', {}).get('spine_health_score', 0.5),
            'neck_health': pose_3d.get('spinal_analysis', {}).get('neck_health_score', 0.5),
        }
        self.posture_history.append(current_pose_summary)

        # 4. 자세 변화 추세 분석
        trend_analysis = self._analyze_posture_trends()

        # 5. [S-Class] 생체역학적 건강 종합 점수
        biomechanical_health = self._calculate_biomechanical_health_score(pose_3d)

        return {
            'driving_suitability': suitability,
            'identified_issues': issues,
            'recommendation': self._generate_posture_recommendation(suitability, issues, biomechanical_health),
            'fatigue_indicators': fatigue_indicators,
            'distraction_indicators': distraction_indicators,
            'trend_analysis': trend_analysis,
            'biomechanical_health': biomechanical_health,
            'data_quality': {'pose_2d': pose_2d['available'], 'pose_3d': pose_3d['available']}
        }

    def _analyze_spinal_alignment(self, world_landmarks: Any) -> Dict[str, Any]:
        """ [S-Class] 척추 정렬 및 목 각도 분석 """
        try:
            lm = MediaPipeConstants.PoseLandmarks
            
            # 주요 3D 포인트 추출
            nose = np.array([world_landmarks[lm.NOSE].x, world_landmarks[lm.NOSE].y, world_landmarks[lm.NOSE].z])
            ls, rs = world_landmarks[lm.LEFT_SHOULDER], world_landmarks[lm.RIGHT_SHOULDER]
            lh, rh = world_landmarks[lm.LEFT_HIP], world_landmarks[lm.RIGHT_HIP]
            
            neck_base = np.mean([[p.x, p.y, p.z] for p in [ls, rs]], axis=0)
            torso_base = np.mean([[p.x, p.y, p.z] for p in [lh, rh]], axis=0)
            
            # 목 벡터와 몸통 벡터 계산
            neck_vector = nose - neck_base
            torso_vector = neck_base - torso_base
            
            # 거북목 각도 (Forward Head Posture)
            # z축(전후방)과 y축(상하)만 사용하여 측면에서 본 각도 계산
            neck_vector_2d = np.array([neck_vector[2], neck_vector[1]])
            vertical_vector_2d = np.array([0, 1])
            
            if np.linalg.norm(neck_vector_2d) > MathConstants.VECTOR_NORM_MIN:
                fhp_angle = math.degrees(math.acos(np.clip(
                    np.dot(neck_vector_2d, vertical_vector_2d) / np.linalg.norm(neck_vector_2d), 
                    -1.0, 1.0
                )))
            else:
                fhp_angle = 0.0
            
            # 척추 각도
            if np.linalg.norm(torso_vector) > MathConstants.VECTOR_NORM_MIN and np.linalg.norm(neck_vector) > MathConstants.VECTOR_NORM_MIN:
                spine_angle = math.degrees(math.acos(np.clip(
                    np.dot(torso_vector, neck_vector) / (np.linalg.norm(torso_vector) * np.linalg.norm(neck_vector)), 
                    -1.0, 1.0
                )))
            else:
                spine_angle = 180.0  # 기본값
            
            # 건강 점수 계산
            neck_health_score = max(0, 1 - (max(0, fhp_angle - 15) / 30))  # 15도 이상 기울면 점수 감소
            spine_health_score = max(0, (spine_angle - 150) / 30)  # 180에 가까울수록 좋음
            
            return {
                'forward_head_posture_angle': fhp_angle, # 20도 이상이면 거북목 위험
                'spine_curvature_angle': spine_angle, # 180에 가까울수록 곧은 자세
                'neck_health_score': neck_health_score,
                'spine_health_score': spine_health_score,
                'cervical_risk_level': 'high' if fhp_angle > 25 else 'medium' if fhp_angle > 15 else 'low'
            }
            
        except Exception as e:
            logger.error(f"척추 정렬 분석 중 오류: {e}")
            return {
                'forward_head_posture_angle': 0.0,
                'spine_curvature_angle': 180.0,
                'neck_health_score': 0.5,
                'spine_health_score': 0.5,
                'cervical_risk_level': 'unknown'
            }

    def _analyze_postural_sway(self, world_landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """ [S-Class] 자세 불안정성(Postural Sway) 측정 """
        try:
            lm = MediaPipeConstants.PoseLandmarks
            torso_center = np.mean([[p.x, p.y, p.z] for p in [
                world_landmarks[lm.LEFT_SHOULDER], world_landmarks[lm.RIGHT_SHOULDER],
                world_landmarks[lm.LEFT_HIP], world_landmarks[lm.RIGHT_HIP]
            ]], axis=0)
            
            # 이력에 현재 몸통 중심점 추가
            self.torso_center_history.append({
                'timestamp': timestamp,
                'position': torso_center
            })
            
            if len(self.torso_center_history) < 30:
                return {'sway_area_cm2': 0.0, 'sway_velocity_cm_s': 0.0, 'stability_score': 1.0}

            # 95% 신뢰구간 타원 면적으로 Sway Area 계산
            recent_positions = np.array([h['position'] for h in list(self.torso_center_history)])
            
            # X-Z 평면에서의 흔들림 (좌우, 전후)
            x_positions = recent_positions[:, 0]
            z_positions = recent_positions[:, 2]
            
            # 공분산 행렬 계산
            cov = np.cov(x_positions, z_positions)
            eigenvalues, _ = np.linalg.eig(cov)
            
            # 실제 거리(cm)로 변환하기 위한 스케일 팩터(가정: 1 unit = 100cm)
            scale_factor = 100 
            sway_area = math.pi * 5.991 * np.sqrt(np.prod(np.abs(eigenvalues))) * (scale_factor**2)
            
            # Sway Velocity 계산 (연속된 포인트 간 거리의 평균)
            if len(self.torso_center_history) > 1:
                velocities = []
                for i in range(1, len(self.torso_center_history)):
                    dt = self.torso_center_history[i]['timestamp'] - self.torso_center_history[i-1]['timestamp']
                    if dt > 0:
                        dp = np.linalg.norm(self.torso_center_history[i]['position'] - self.torso_center_history[i-1]['position'])
                        velocities.append(dp / dt * scale_factor)  # cm/s
                
                sway_velocity = np.mean(velocities) if velocities else 0.0
            else:
                sway_velocity = 0.0
            
            # 안정성 점수 (낮은 sway가 좋음)
            stability_score = max(0, 1 - (sway_area / 25.0))  # 25cm² 이상이면 불안정

            return {
                'sway_area_cm2': sway_area,
                'sway_velocity_cm_s': sway_velocity, 
                'stability_score': stability_score,
                'sway_pattern': self._classify_sway_pattern(x_positions, z_positions)
            }
            
        except Exception as e:
            logger.error(f"자세 불안정성 분석 중 오류: {e}")
            return {'sway_area_cm2': 0.0, 'sway_velocity_cm_s': 0.0, 'stability_score': 1.0}

    def _classify_sway_pattern(self, x_positions: np.ndarray, z_positions: np.ndarray) -> str:
        """ 흔들림 패턴 분류 """
        try:
            x_var = np.var(x_positions)
            z_var = np.var(z_positions)
            
            if x_var > z_var * 2:
                return 'lateral_dominant'  # 좌우 흔들림 우세
            elif z_var > x_var * 2:
                return 'anterior_posterior_dominant'  # 전후 흔들림 우세
            else:
                return 'circular_pattern'  # 원형 패턴
                
        except Exception:
            return 'unknown'

    def _calculate_biomechanical_health_score(self, pose_3d: Dict) -> Dict[str, Any]:
        """ [S-Class] 생체역학적 건강 종합 점수 """
        if not pose_3d.get('available'):
            return {'overall_score': 0.5, 'risk_factors': [], 'recommendations': []}
        
        scores = []
        risk_factors = []
        recommendations = []
        
        # 척추 건강도
        spinal_score = pose_3d.get('spinal_analysis', {}).get('spine_health_score', 0.5)
        neck_score = pose_3d.get('spinal_analysis', {}).get('neck_health_score', 0.5)
        scores.extend([spinal_score, neck_score])
        
        if neck_score < 0.6:
            risk_factors.append('거북목 위험')
            recommendations.append('목을 곧게 펴고 어깨를 뒤로 당기세요')
        
        # 자세 안정성
        stability_score = pose_3d.get('postural_sway', {}).get('stability_score', 1.0)
        scores.append(stability_score)
        
        if stability_score < 0.7:
            risk_factors.append('자세 불안정')
            recommendations.append('코어 근육 강화 운동을 권장합니다')
        
        # 구부정한 정도
        slouch_factor = pose_3d.get('slouch_detection', {}).get('slouch_factor', 0)
        slouch_score = 1.0 - slouch_factor
        scores.append(slouch_score)
        
        if slouch_factor > 0.6:
            risk_factors.append('구부정한 자세')
            recommendations.append('등받이에 기대어 앉으세요')
        
        overall_score = np.mean(scores) if scores else 0.5
        
        return {
            'overall_score': overall_score,
            'spinal_health': spinal_score,
            'neck_health': neck_score,
            'stability': stability_score,
            'posture_quality': slouch_score,
            'risk_factors': risk_factors,
            'recommendations': recommendations
        }

    def _detect_slouching_posture(self, world_landmarks: Any) -> Dict[str, Any]:
        """ [고도화] 3D 기반 구부정한 자세(slouch) 감지 """
        try:
            lm = MediaPipeConstants.PoseLandmarks
            shoulder_center = np.mean([
                [p.x, p.y, p.z] for p in [world_landmarks[lm.LEFT_SHOULDER], world_landmarks[lm.RIGHT_SHOULDER]]
            ], axis=0)
            hip_center = np.mean([
                [p.x, p.y, p.z] for p in [world_landmarks[lm.LEFT_HIP], world_landmarks[lm.RIGHT_HIP]]
            ], axis=0)
            
            torso_vector = shoulder_center - hip_center
            vertical_vector = np.array([0, 1, 0]) # y축을 수직 기준으로 사용
            
            # 수직 벡터와의 각도 계산
            if np.linalg.norm(torso_vector) > MathConstants.VECTOR_NORM_MIN:
                dot_product = np.dot(torso_vector, vertical_vector) / np.linalg.norm(torso_vector)
                slouch_angle = math.degrees(math.acos(np.clip(dot_product, -1.0, 1.0)))
            else:
                slouch_angle = 0.0
            
            slouch_factor = max(0.0, min(1.0, (slouch_angle - 20) / 70.0)) # 20도 이상부터 slouch로 간주
            
            return {
                'slouch_angle': slouch_angle,
                'slouch_factor': slouch_factor,
                'is_slouching': slouch_factor > 0.6,
                'severity': 'severe' if slouch_factor > 0.8 else 'moderate' if slouch_factor > 0.6 else 'mild'
            }
        except Exception as e:
            logger.error(f"Slouch 감지 중 오류: {e}")
            return {'slouch_factor': 0.0, 'is_slouching': False, 'severity': 'unknown'}

    def _evaluate_driving_posture_suitability(self, pose_2d: Dict, pose_3d: Dict) -> Tuple[float, List[str]]:
        """ [고도화] 운전 자세 적합성 평가 """
        suitability_factors = []
        issues = []
        
        if pose_2d.get('available'):
            tilt_severity = pose_2d.get('shoulder_analysis', {}).get('tilt_severity', 0)
            if tilt_severity > 0.5: issues.append('과도한 어깨 기울기')
            suitability_factors.append(1.0 - tilt_severity)
            
        if pose_3d.get('available'):
            slouch_factor = pose_3d.get('slouch_detection', {}).get('slouch_factor', 0)
            if slouch_factor > 0.6: issues.append('구부정한 자세')
            suitability_factors.append(1.0 - slouch_factor)
            
            balance_score = pose_3d.get('balance', {}).get('balance_score', 0.5)
            if balance_score < 0.4: issues.append('몸의 균형 불량')
            suitability_factors.append(balance_score)
            
            # [S-Class] 척추 건강도 추가
            spine_health = pose_3d.get('spinal_analysis', {}).get('spine_health_score', 0.5)
            if spine_health < 0.6: issues.append('척추 정렬 불량')
            suitability_factors.append(spine_health)
            
        return np.mean(suitability_factors) if suitability_factors else 0.5, issues

    def _generate_posture_recommendation(self, suitability: float, issues: List[str], biomech_health: Dict) -> str:
        """ [고도화] 자세 개선 권장사항 생성 """
        if suitability > 0.8 and biomech_health['overall_score'] > 0.8:
            return "우수한 운전 자세를 유지하고 있습니다."
        
        if biomech_health['risk_factors']:
            primary_risk = biomech_health['risk_factors'][0]
            primary_recommendation = biomech_health['recommendations'][0] if biomech_health['recommendations'] else "자세를 교정하세요"
            return f"{primary_risk} 감지: {primary_recommendation}"
        
        if issues:
            return f"다음 사항 개선 필요: {', '.join(issues[:2])}"  # 최대 2개만 표시
        
        return "자세 교정이 필요합니다. 바른 자세로 앉아주세요."
        
    def _analyze_posture_trends(self) -> Dict[str, Any]:
        """ [고도화] 자세 변화 추세 분석 """
        if len(self.posture_history) < 60: # 최소 2초 데이터 필요
            return {'trend_available': False, 'stability_trend': 'stable', 'deterioration_rate': 0.0}

        recent_history = list(self.posture_history)
        timestamps = np.array([h['timestamp'] for h in recent_history])
        suitability_scores = np.array([h['suitability'] for h in recent_history])
        
        # 선형 회귀로 추세 기울기 계산
        try:
            coeffs = np.polyfit(timestamps, suitability_scores, 1)
            slope = coeffs[0]
        except np.linalg.LinAlgError:
            slope = 0.0

        if slope < -0.01: trend = 'deteriorating' # 시간당 1% 이상 하락
        elif slope > 0.01: trend = 'improving'
        else: trend = 'stable'
        
        # 건강 점수 추세도 분석
        health_scores = np.array([h.get('spinal_health', 0.5) + h.get('neck_health', 0.5) for h in recent_history]) / 2
        try:
            health_slope = np.polyfit(timestamps, health_scores, 1)[0]
        except np.linalg.LinAlgError:
            health_slope = 0.0
        
        return {
            'trend_available': True,
            'stability_trend': trend,
            'deterioration_rate': -slope if slope < 0 else 0.0,
            'health_trend': 'improving' if health_slope > 0.005 else 'declining' if health_slope < -0.005 else 'stable',
            'prediction': self._predict_posture_future(slope, health_slope)
        }

    def _predict_posture_future(self, suitability_slope: float, health_slope: float) -> str:
        """ 자세 변화 예측 """
        if suitability_slope < -0.02 or health_slope < -0.01:
            return "지속적인 자세 악화 예상 - 즉시 휴식 권장"
        elif suitability_slope > 0.01 and health_slope > 0.005:
            return "자세 개선 중 - 현재 상태 유지"
        else:
            return "안정적인 자세 유지 중"

    def _extract_key_pose_points(self, landmarks: Any) -> Dict[str, Any]:
        """주요 자세 포인트 추출"""
        constants = MediaPipeConstants.PoseLandmarks
        return {
            'left_shoulder': landmarks[constants.LEFT_SHOULDER],
            'right_shoulder': landmarks[constants.RIGHT_SHOULDER],
            'left_hip': landmarks[constants.LEFT_HIP],
            'right_hip': landmarks[constants.RIGHT_HIP],
        }

    def _analyze_shoulder_orientation(self, key_points: Dict) -> Dict[str, Any]:
        """어깨 방향 분석"""
        left_shoulder = key_points['left_shoulder']
        right_shoulder = key_points['right_shoulder']
        shoulder_angle = math.degrees(math.atan2(left_shoulder.y - right_shoulder.y, left_shoulder.x - right_shoulder.x))
        return { 'angle_degrees': shoulder_angle, 'tilt_severity': min(1.0, abs(shoulder_angle) / 45.0) }

    def _analyze_body_symmetry(self, key_points: Dict) -> Dict[str, Any]:
        """몸의 좌우 대칭성 분석"""
        shoulder_height_diff = abs(key_points['left_shoulder'].y - key_points['right_shoulder'].y)
        hip_height_diff = abs(key_points['left_hip'].y - key_points['right_hip'].y)
        return { 'symmetry_score': 1.0 - min(1.0, (shoulder_height_diff + hip_height_diff) * 2) }

    def _analyze_3d_body_balance(self, world_landmarks: Any) -> Dict[str, Any]:
        """3D 공간에서의 몸의 균형 분석"""
        lm = MediaPipeConstants.PoseLandmarks
        left_center = np.mean([[p.x, p.y, p.z] for p in [world_landmarks[lm.LEFT_SHOULDER], world_landmarks[lm.LEFT_HIP]]], axis=0)
        right_center = np.mean([[p.x, p.y, p.z] for p in [world_landmarks[lm.RIGHT_SHOULDER], world_landmarks[lm.RIGHT_HIP]]], axis=0)
        balance_distance = np.linalg.norm(left_center - right_center)
        return {'balance_score': max(0.0, 1.0 - balance_distance * 5) }

    def _calculate_pose_complexity(self, world_landmarks: Any) -> Dict[str, Any]:
        """자세 복잡도 계산"""
        torso_points = [world_landmarks[i] for i in [11, 12, 23, 24]]  # 어깨와 엉덩이 포인트들
        positions = np.array([[lm.x, lm.y, lm.z] for lm in torso_points])
        return {'overall_complexity': min(1.0, np.var(positions) * 10) }

    def _extract_fatigue_indicators_from_pose(self, pose_3d: Dict) -> Dict[str, float]:
        """자세에서 피로도 지표 추출"""
        if not pose_3d.get('available'): return {'slouching': 0.0, 'instability': 0.0}
        
        slouching = pose_3d.get('slouch_detection', {}).get('slouch_factor', 0.0)
        instability = 1.0 - pose_3d.get('postural_sway', {}).get('stability_score', 1.0)
        
        return {'slouching': slouching, 'instability': instability}

    def _extract_distraction_indicators_from_pose(self, pose_3d: Dict) -> Dict[str, float]:
        """자세에서 주의산만 지표 추출"""
        if not pose_3d.get('available'): return {'unusual_positioning': 0.0}
        return {'unusual_positioning': pose_3d.get('complexity', {}).get('overall_complexity', 0.0)}

    def _update_pose_metrics(self, pose_analysis: Dict[str, Any]):
        """ 자세 관련 메트릭 업데이트 """
        metrics_data = {
            'pose_complexity_score': pose_analysis.get('distraction_indicators', {}).get('unusual_positioning', 0.0),
            'slouch_factor': pose_analysis.get('fatigue_indicators', {}).get('slouching', 0.0),
            'posture_suitability': pose_analysis.get('driving_suitability', 0.5),
            'spinal_health_score': pose_analysis.get('biomechanical_health', {}).get('spinal_health', 0.5),
            'postural_stability': pose_analysis.get('biomechanical_health', {}).get('stability', 1.0)
        }
        if hasattr(self.metrics_updater, 'update_pose_metrics'):
            self.metrics_updater.update_pose_metrics(metrics_data)

    async def _handle_no_pose_detected(self) -> Dict[str, Any]:
        """자세 미감지 처리"""
        logger.warning("자세가 감지되지 않음 - 백업 모드 또는 센서 재보정 필요")
        return { 
            'pose_detected': False, 
            'pose_analysis': {
                'driving_suitability': 0.0,
                'biomechanical_health': {'overall_score': 0.0, 'risk_factors': ['자세 감지 불가'], 'recommendations': ['카메라 위치를 조정하세요']}
            }
        }