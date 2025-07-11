"""
자세 데이터 전문 처리기
33개 키포인트를 활용한 고도화된 자세 분석을 담당합니다.
"""

import numpy as np
import math
from typing import Dict, Any, List, Optional
from core.interfaces import IPoseDataProcessor, IMetricsUpdater
from core.constants import MediaPipeConstants, AnalysisConstants, MathConstants
from config.settings import get_config
import logging

logger = logging.getLogger(__name__)


class PoseDataProcessor(IPoseDataProcessor):
    """
    자세 데이터 전문 처리기
    
    이 클래스는 마치 물리치료사나 자세 교정 전문가처럼 운전자의 몸 자세를 분석합니다.
    33개의 신체 키포인트를 활용하여:
    - 어깨와 몸통의 방향성 분석
    - 자세의 복잡도와 안정성 평가
    - 구부정한 자세(slouch) 감지
    - 몸의 균형과 대칭성 분석
    """
    
    def __init__(self, metrics_updater: IMetricsUpdater):
        self.metrics_updater = metrics_updater
        self.config = get_config()
        
        # 자세 분석을 위한 상태 추적 변수들
        self.pose_stability_history = []
        self.shoulder_alignment_history = []
        
        logger.info("PoseDataProcessor 초기화 완료 - 자세 분석 전문가 준비됨")
    
    def get_processor_name(self) -> str:
        return "PoseDataProcessor"
    
    def get_required_data_types(self) -> List[str]:
        return ["pose_landmarks", "pose_world_landmarks"]
    
    async def process_data(self, data: Any, timestamp: float) -> Dict[str, Any]:
        """
        자세 데이터 통합 처리 메인 메서드
        
        자세 데이터의 모든 측면을 종합적으로 분석합니다.
        마치 종합 건강 검진처럼 다양한 각도에서 자세를 평가합니다.
        """
        if not data or not data.pose_landmarks:
            return await self._handle_no_pose_detected()
        
        pose_result = data
        results = {}
        
        # 1. 2D 랜드마크 기반 분석
        if pose_result.pose_landmarks:
            landmarks_2d_data = await self.process_pose_landmarks(
                pose_result.pose_landmarks[0], timestamp
            )
            results.update(landmarks_2d_data)
        
        # 2. 3D 월드 랜드마크 기반 고급 분석
        if pose_result.pose_world_landmarks:
            world_landmarks_data = await self.process_world_landmarks(
                pose_result.pose_world_landmarks[0], timestamp
            )
            results.update(world_landmarks_data)
        
        # 3. 종합 자세 평가
        comprehensive_analysis = await self.perform_comprehensive_pose_analysis(results)
        results.update(comprehensive_analysis)
        
        # 메트릭 업데이트
        if 'pose_analysis' in results:
            self._update_pose_metrics(results['pose_analysis'])
        
        return results
    
    async def process_pose_landmarks(self, landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """
        2D 자세 랜드마크 처리
        
        화면상의 2차원 좌표를 기반으로 기본적인 자세 분석을 수행합니다.
        주로 어깨의 기울기와 기본적인 몸통 방향을 파악합니다.
        """
        try:
            if not landmarks or len(landmarks) < MediaPipeConstants.PoseLandmarks.RIGHT_HIP + 1:
                return {'pose_2d': {'available': False, 'reason': 'insufficient_landmarks'}}
            
            # 주요 포인트 추출
            key_points = self._extract_key_pose_points(landmarks)
            
            # 어깨 방향 분석
            shoulder_analysis = self._analyze_shoulder_orientation(key_points)
            
            # 기본 몸통 자세 분석
            torso_analysis = self._analyze_basic_torso_posture(key_points)
            
            # 대칭성 분석
            symmetry_analysis = self._analyze_body_symmetry(key_points)
            
            return {
                'pose_2d': {
                    'available': True,
                    'shoulder_analysis': shoulder_analysis,
                    'torso_analysis': torso_analysis,
                    'symmetry_analysis': symmetry_analysis,
                    'timestamp': timestamp
                }
            }
        except Exception as e:
            logger.error(f"2D 자세 분석 중 오류 발생: {e}")
            return {'pose_2d': {'available': False, 'reason': 'processing_error'}}
    
    async def process_world_landmarks(self, world_landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """
        3D 월드 랜드마크 처리
        
        실제 3차원 공간에서의 좌표를 활용한 고급 자세 분석입니다.
        이는 마치 3D 스캐너로 몸을 분석하는 것과 같습니다.
        """
        try:
            if not world_landmarks or len(world_landmarks) < 33:
                return {'pose_3d': {'available': False, 'reason': 'insufficient_world_landmarks'}}
            
            # 3D 공간에서의 몸통 분석
            torso_3d_analysis = self._analyze_3d_torso_geometry(world_landmarks)
            
            # 자세 복잡도 계산
            complexity_analysis = self._calculate_pose_complexity(world_landmarks)
            
            # 몸의 기울기와 균형 분석
            balance_analysis = self._analyze_3d_body_balance(world_landmarks)
            
            # 어깨 라인 3D 분석
            shoulder_3d_analysis = self._analyze_3d_shoulder_alignment(world_landmarks)
            
            # 구부정한 자세(slouch) 감지
            slouch_analysis = self._detect_slouching_posture(world_landmarks)
            
            return {
                'pose_3d': {
                    'available': True,
                    'torso_geometry': torso_3d_analysis,
                    'complexity': complexity_analysis,
                    'balance': balance_analysis,
                    'shoulder_3d': shoulder_3d_analysis,
                    'slouch_detection': slouch_analysis,
                    'timestamp': timestamp
                }
            }
        except Exception as e:
            logger.error(f"3D 자세 분석 중 오류 발생: {e}")
            return {'pose_3d': {'available': False, 'reason': 'processing_error'}}
    
    async def perform_comprehensive_pose_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        종합 자세 분석
        
        2D와 3D 분석 결과를 종합하여 최종적인 자세 평가를 수행합니다.
        마치 여러 검사 결과를 종합하여 최종 진단을 내리는 것과 같습니다.
        """
        try:
            pose_2d = results.get('pose_2d', {})
            pose_3d = results.get('pose_3d', {})
            
            # 전반적인 자세 건강도 점수 (0-1)
            overall_posture_score = self._calculate_overall_posture_score(pose_2d, pose_3d)
            
            # 운전 자세 적합성 평가
            driving_suitability = self._evaluate_driving_posture_suitability(pose_2d, pose_3d)
            
            # 피로도 관련 자세 지표
            fatigue_indicators = self._extract_fatigue_indicators_from_pose(pose_2d, pose_3d)
            
            # 주의산만 관련 자세 지표
            distraction_indicators = self._extract_distraction_indicators_from_pose(pose_2d, pose_3d)
            
            # 자세 변화 추세 분석
            trend_analysis = self._analyze_posture_trends()
            
            return {
                'pose_analysis': {
                    'overall_score': overall_posture_score,
                    'driving_suitability': driving_suitability,
                    'fatigue_indicators': fatigue_indicators,
                    'distraction_indicators': distraction_indicators,
                    'trend_analysis': trend_analysis,
                    'data_quality': {
                        'pose_2d_available': pose_2d.get('available', False),
                        'pose_3d_available': pose_3d.get('available', False)
                    }
                }
            }
        except Exception as e:
            logger.error(f"종합 자세 분석 중 오류 발생: {e}")
            return {'pose_analysis': {'overall_score': 0.5, 'error': str(e)}}
    
    def _extract_key_pose_points(self, landmarks: Any) -> Dict[str, Any]:
        """주요 자세 포인트 추출"""
        constants = MediaPipeConstants.PoseLandmarks
        
        return {
            'left_shoulder': landmarks[constants.LEFT_SHOULDER],
            'right_shoulder': landmarks[constants.RIGHT_SHOULDER],
            'left_hip': landmarks[constants.LEFT_HIP],
            'right_hip': landmarks[constants.RIGHT_HIP],
            'left_wrist': landmarks[constants.LEFT_WRIST],
            'right_wrist': landmarks[constants.RIGHT_WRIST],
            'nose': landmarks[constants.NOSE],
            'left_ear': landmarks[constants.LEFT_EAR],
            'right_ear': landmarks[constants.RIGHT_EAR]
        }
    
    def _analyze_shoulder_orientation(self, key_points: Dict) -> Dict[str, Any]:
        """
        어깨 방향 분석
        
        어깨 라인의 기울기를 분석하여 운전자가 몸을 기울이고 있는지,
        또는 한쪽으로 치우쳐 앉아있는지 파악합니다.
        """
        try:
            left_shoulder = key_points['left_shoulder']
            right_shoulder = key_points['right_shoulder']
            
            # 어깨 라인의 각도 계산
            shoulder_angle = math.degrees(math.atan2(
                left_shoulder.y - right_shoulder.y,
                left_shoulder.x - right_shoulder.x
            ))
            
            # 어깨 너비 계산
            shoulder_width = math.sqrt(
                (left_shoulder.x - right_shoulder.x) ** 2 +
                (left_shoulder.y - right_shoulder.y) ** 2
            )
            
            # 기울기 정도 평가
            tilt_severity = min(1.0, abs(shoulder_angle) / 45.0)  # 45도를 최대값으로 정규화
            
            return {
                'angle_degrees': shoulder_angle,
                'width_normalized': shoulder_width,
                'tilt_severity': tilt_severity,
                'is_tilted': abs(shoulder_angle) > 15.0,  # 15도 이상이면 기울어진 것으로 판단
                'tilt_direction': 'left' if shoulder_angle > 0 else 'right' if shoulder_angle < 0 else 'center'
            }
        except Exception as e:
            logger.error(f"어깨 방향 분석 중 오류: {e}")
            return {'angle_degrees': 0.0, 'tilt_severity': 0.0}
    
    def _analyze_basic_torso_posture(self, key_points: Dict) -> Dict[str, Any]:
        """기본 몸통 자세 분석"""
        try:
            # 몸통 중심점들 계산
            shoulder_center = {
                'x': (key_points['left_shoulder'].x + key_points['right_shoulder'].x) / 2,
                'y': (key_points['left_shoulder'].y + key_points['right_shoulder'].y) / 2
            }
            
            hip_center = {
                'x': (key_points['left_hip'].x + key_points['right_hip'].x) / 2,
                'y': (key_points['left_hip'].y + key_points['right_hip'].y) / 2
            }
            
            # 몸통 각도 계산
            torso_angle = math.degrees(math.atan2(
                shoulder_center['y'] - hip_center['y'],
                abs(shoulder_center['x'] - hip_center['x']) + MathConstants.EPSILON
            ))
            
            # 몸통 길이 (정규화됨)
            torso_length = math.sqrt(
                (shoulder_center['x'] - hip_center['x']) ** 2 +
                (shoulder_center['y'] - hip_center['y']) ** 2
            )
            
            return {
                'torso_angle': torso_angle,
                'torso_length': torso_length,
                'is_upright': 70.0 <= abs(torso_angle) <= 110.0,  # 직립 자세 범위
                'forward_lean': torso_angle < 70.0,
                'backward_lean': torso_angle > 110.0
            }
        except Exception as e:
            logger.error(f"몸통 자세 분석 중 오류: {e}")
            return {'torso_angle': 90.0, 'is_upright': True}
    
    def _analyze_body_symmetry(self, key_points: Dict) -> Dict[str, Any]:
        """몸의 좌우 대칭성 분석"""
        try:
            # 중심선 계산 (코와 어깨/엉덩이 중심점을 연결)
            nose = key_points['nose']
            shoulder_center_x = (key_points['left_shoulder'].x + key_points['right_shoulder'].x) / 2
            hip_center_x = (key_points['left_hip'].x + key_points['right_hip'].x) / 2
            
            # 좌우 어깨 높이 차이
            shoulder_height_diff = abs(key_points['left_shoulder'].y - key_points['right_shoulder'].y)
            
            # 좌우 엉덩이 높이 차이
            hip_height_diff = abs(key_points['left_hip'].y - key_points['right_hip'].y)
            
            # 전체 대칭성 점수 (0: 완전 비대칭, 1: 완전 대칭)
            symmetry_score = 1.0 - min(1.0, (shoulder_height_diff + hip_height_diff) * 2)
            
            # 몸이 한쪽으로 치우쳐 있는지 확인
            body_shift = abs(nose.x - (shoulder_center_x + hip_center_x) / 2)
            
            return {
                'symmetry_score': symmetry_score,
                'shoulder_height_diff': shoulder_height_diff,
                'hip_height_diff': hip_height_diff,
                'body_shift': body_shift,
                'is_symmetric': symmetry_score > 0.7,
                'shift_direction': 'left' if nose.x < shoulder_center_x else 'right' if nose.x > shoulder_center_x else 'center'
            }
        except Exception as e:
            logger.error(f"대칭성 분석 중 오류: {e}")
            return {'symmetry_score': 0.5, 'is_symmetric': True}
    
    def _analyze_3d_torso_geometry(self, world_landmarks: Any) -> Dict[str, Any]:
        """3D 공간에서의 몸통 기하학 분석"""
        try:
            # 주요 3D 포인트들 추출
            landmarks = MediaPipeConstants.PoseLandmarks
            ls = world_landmarks[landmarks.LEFT_SHOULDER]
            rs = world_landmarks[landmarks.RIGHT_SHOULDER]
            lh = world_landmarks[landmarks.LEFT_HIP]
            rh = world_landmarks[landmarks.RIGHT_HIP]
            
            # 어깨-엉덩이 벡터 계산
            shoulder_vector = np.array([ls.x - rs.x, ls.y - rs.y, ls.z - rs.z])
            hip_vector = np.array([lh.x - rh.x, lh.y - rh.y, lh.z - rh.z])
            
            # 몸통 회전 각도 (어깨와 엉덩이의 회전 차이)
            if np.linalg.norm(shoulder_vector) > 0 and np.linalg.norm(hip_vector) > 0:
                torso_twist = math.degrees(math.acos(np.clip(
                    np.dot(shoulder_vector, hip_vector) / 
                    (np.linalg.norm(shoulder_vector) * np.linalg.norm(hip_vector)),
                    -1.0, 1.0
                )))
            else:
                torso_twist = 0.0
            
            # 3D 자세 안정성 점수
            stability_score = max(0.0, 1.0 - torso_twist / 90.0)
            
            return {
                'torso_twist_degrees': torso_twist,
                'stability_score': stability_score,
                'shoulder_vector_magnitude': np.linalg.norm(shoulder_vector),
                'hip_vector_magnitude': np.linalg.norm(hip_vector),
                'is_stable': torso_twist < 30.0
            }
        except Exception as e:
            logger.error(f"3D 몸통 기하학 분석 중 오류: {e}")
            return {'torso_twist_degrees': 0.0, 'stability_score': 1.0}
    
    def _calculate_pose_complexity(self, world_landmarks: Any) -> Dict[str, Any]:
        """
        자세 복잡도 계산
        
        여러 신체 부위의 위치 변화를 종합하여 자세가 얼마나 복잡한지 측정합니다.
        복잡한 자세는 주의산만이나 불편함을 나타낼 수 있습니다.
        """
        try:
            # 몸통 주요 포인트들 (어깨와 엉덩이)
            torso_points = [
                world_landmarks[i] for i in MediaPipeConstants.PoseLandmarks.TORSO_POINTS
            ]
            
            # 각 포인트의 3D 좌표 배열 생성
            positions = np.array([[lm.x, lm.y, lm.z] for lm in torso_points])
            
            # 위치의 분산을 복잡도로 사용
            complexity_variance = np.var(positions)
            
            # 0-1 범위로 정규화 (경험적 최대값 10 사용)
            complexity_score = min(1.0, complexity_variance * 10)
            
            # 각 축별 분산도 계산
            x_variance = np.var(positions[:, 0])
            y_variance = np.var(positions[:, 1])
            z_variance = np.var(positions[:, 2])
            
            return {
                'overall_complexity': complexity_score,
                'x_axis_variance': x_variance,
                'y_axis_variance': y_variance,
                'z_axis_variance': z_variance,
                'is_complex': complexity_score > 0.7,
                'dominant_axis': 'x' if x_variance > max(y_variance, z_variance) else 
                                'y' if y_variance > z_variance else 'z'
            }
        except Exception as e:
            logger.error(f"자세 복잡도 계산 중 오류: {e}")
            return {'overall_complexity': 0.0, 'is_complex': False}
    
    def _analyze_3d_body_balance(self, world_landmarks: Any) -> Dict[str, Any]:
        """3D 공간에서의 몸의 균형 분석"""
        try:
            landmarks = MediaPipeConstants.PoseLandmarks
            
            # 몸의 중심점들 계산
            left_points = [
                world_landmarks[landmarks.LEFT_SHOULDER],
                world_landmarks[landmarks.LEFT_HIP]
            ]
            right_points = [
                world_landmarks[landmarks.RIGHT_SHOULDER], 
                world_landmarks[landmarks.RIGHT_HIP]
            ]
            
            # 좌우 무게중심 계산
            left_center = np.mean([[p.x, p.y, p.z] for p in left_points], axis=0)
            right_center = np.mean([[p.x, p.y, p.z] for p in right_points], axis=0)
            
            # 균형 점수 계산 (좌우 중심점 간의 거리 기반)
            balance_distance = np.linalg.norm(left_center - right_center)
            balance_score = max(0.0, 1.0 - balance_distance * 5)  # 정규화
            
            # 전후 균형 (어깨와 엉덩이의 z축 차이)
            shoulder_z = (world_landmarks[landmarks.LEFT_SHOULDER].z + 
                         world_landmarks[landmarks.RIGHT_SHOULDER].z) / 2
            hip_z = (world_landmarks[landmarks.LEFT_HIP].z + 
                    world_landmarks[landmarks.RIGHT_HIP].z) / 2
            
            forward_lean_amount = shoulder_z - hip_z
            
            return {
                'balance_score': balance_score,
                'balance_distance': balance_distance,
                'forward_lean_amount': forward_lean_amount,
                'is_balanced': balance_score > 0.6,
                'lean_direction': 'forward' if forward_lean_amount > 0.1 else 
                                'backward' if forward_lean_amount < -0.1 else 'neutral'
            }
        except Exception as e:
            logger.error(f"3D 균형 분석 중 오류: {e}")
            return {'balance_score': 0.5, 'is_balanced': True}
    
    def _analyze_3d_shoulder_alignment(self, world_landmarks: Any) -> Dict[str, Any]:
        """3D 공간에서의 어깨 정렬 분석"""
        try:
            landmarks = MediaPipeConstants.PoseLandmarks
            ls = world_landmarks[landmarks.LEFT_SHOULDER]
            rs = world_landmarks[landmarks.RIGHT_SHOULDER]
            
            # 어깨 라인 벡터
            shoulder_vector = np.array([rs.x - ls.x, rs.y - ls.y, rs.z - ls.z])
            
            # 수평선과의 각도 계산
            horizontal_vector = np.array([1, 0, 0])  # x축을 수평 기준으로 사용
            
            if np.linalg.norm(shoulder_vector) > 0:
                alignment_angle = math.degrees(math.acos(np.clip(
                    np.dot(shoulder_vector, horizontal_vector) / np.linalg.norm(shoulder_vector),
                    -1.0, 1.0
                )))
            else:
                alignment_angle = 0.0
            
            # 정렬 점수 (0: 완전 비정렬, 1: 완전 정렬)
            alignment_score = max(0.0, 1.0 - abs(alignment_angle - 90.0) / 90.0)
            
            return {
                'alignment_angle': alignment_angle,
                'alignment_score': alignment_score,
                'shoulder_vector_3d': shoulder_vector.tolist(),
                'is_aligned': alignment_score > 0.7,
                'tilt_severity': abs(alignment_angle - 90.0) / 90.0
            }
        except Exception as e:
            logger.error(f"3D 어깨 정렬 분석 중 오류: {e}")
            return {'alignment_score': 0.5, 'is_aligned': True}
    
    def _detect_slouching_posture(self, world_landmarks: Any) -> Dict[str, Any]:
        """
        구부정한 자세(slouch) 감지
        
        운전자가 등받이에 기대어 구부정하게 앉아있는지 감지합니다.
        이는 졸음이나 피로의 중요한 지표가 될 수 있습니다.
        """
        try:
            landmarks = MediaPipeConstants.PoseLandmarks
            
            # 어깨와 엉덩이의 중심점 계산
            shoulder_center = np.array([
                (world_landmarks[landmarks.LEFT_SHOULDER].x + world_landmarks[landmarks.RIGHT_SHOULDER].x) / 2,
                (world_landmarks[landmarks.LEFT_SHOULDER].y + world_landmarks[landmarks.RIGHT_SHOULDER].y) / 2,
                (world_landmarks[landmarks.LEFT_SHOULDER].z + world_landmarks[landmarks.RIGHT_SHOULDER].z) / 2
            ])
            
            hip_center = np.array([
                (world_landmarks[landmarks.LEFT_HIP].x + world_landmarks[landmarks.RIGHT_HIP].x) / 2,
                (world_landmarks[landmarks.LEFT_HIP].y + world_landmarks[landmarks.RIGHT_HIP].y) / 2,
                (world_landmarks[landmarks.LEFT_HIP].z + world_landmarks[landmarks.RIGHT_HIP].z) / 2
            ])
            
            # 몸통 벡터 계산
            torso_vector = shoulder_center - hip_center
            
            # 수직선과의 각도 계산 (slouch 정도)
            vertical_vector = np.array([0, 1, 0])  # y축을 수직 기준으로 사용
            
            if np.linalg.norm(torso_vector) > 0:
                slouch_angle = math.degrees(math.acos(np.clip(
                    np.dot(torso_vector, vertical_vector) / np.linalg.norm(torso_vector),
                    -1.0, 1.0
                )))
            else:
                slouch_angle = 0.0
            
            # slouch factor 계산 (0: 바른 자세, 1: 완전히 구부정함)
            slouch_factor = max(0.0, min(1.0, (90 - abs(slouch_angle)) / 90))
            
            return {
                'slouch_angle': slouch_angle,
                'slouch_factor': slouch_factor,
                'is_slouching': slouch_factor > 0.5,
                'slouch_severity': 'mild' if slouch_factor < 0.3 else 
                                 'moderate' if slouch_factor < 0.7 else 'severe',
                'torso_vector_3d': torso_vector.tolist()
            }
        except Exception as e:
            logger.error(f"slouch 감지 중 오류: {e}")
            return {'slouch_factor': 0.0, 'is_slouching': False}
    
    def _calculate_overall_posture_score(self, pose_2d: Dict, pose_3d: Dict) -> float:
        """전반적인 자세 건강도 점수 계산"""
        try:
            scores = []
            
            # 2D 분석 결과에서 점수 추출
            if pose_2d.get('available'):
                if 'shoulder_analysis' in pose_2d:
                    scores.append(1.0 - pose_2d['shoulder_analysis'].get('tilt_severity', 0))
                if 'symmetry_analysis' in pose_2d:
                    scores.append(pose_2d['symmetry_analysis'].get('symmetry_score', 0.5))
            
            # 3D 분석 결과에서 점수 추출
            if pose_3d.get('available'):
                if 'torso_geometry' in pose_3d:
                    scores.append(pose_3d['torso_geometry'].get('stability_score', 0.5))
                if 'balance' in pose_3d:
                    scores.append(pose_3d['balance'].get('balance_score', 0.5))
                if 'slouch_detection' in pose_3d:
                    scores.append(1.0 - pose_3d['slouch_detection'].get('slouch_factor', 0))
            
            # 평균 점수 계산
            if scores:
                return sum(scores) / len(scores)
            else:
                return 0.5  # 기본값
                
        except Exception as e:
            logger.error(f"전체 자세 점수 계산 중 오류: {e}")
            return 0.5
    
    def _evaluate_driving_posture_suitability(self, pose_2d: Dict, pose_3d: Dict) -> Dict[str, Any]:
        """운전 자세 적합성 평가"""
        try:
            suitability_factors = []
            issues = []
            
            # 어깨 기울기 확인
            if pose_2d.get('available') and 'shoulder_analysis' in pose_2d:
                shoulder_tilt = pose_2d['shoulder_analysis'].get('tilt_severity', 0)
                if shoulder_tilt > 0.5:
                    issues.append('과도한 어깨 기울기')
                suitability_factors.append(1.0 - shoulder_tilt)
            
            # slouch 확인
            if pose_3d.get('available') and 'slouch_detection' in pose_3d:
                slouch_factor = pose_3d['slouch_detection'].get('slouch_factor', 0)
                if slouch_factor > 0.6:
                    issues.append('구부정한 자세')
                suitability_factors.append(1.0 - slouch_factor)
            
            # 균형 확인
            if pose_3d.get('available') and 'balance' in pose_3d:
                balance_score = pose_3d['balance'].get('balance_score', 0.5)
                if balance_score < 0.4:
                    issues.append('몸의 균형 불량')
                suitability_factors.append(balance_score)
            
            # 전체 적합성 점수
            if suitability_factors:
                overall_suitability = sum(suitability_factors) / len(suitability_factors)
            else:
                overall_suitability = 0.5
            
            return {
                'overall_suitability': overall_suitability,
                'is_suitable': overall_suitability > 0.6,
                'identified_issues': issues,
                'recommendation': self._generate_posture_recommendation(overall_suitability, issues)
            }
            
        except Exception as e:
            logger.error(f"운전 자세 적합성 평가 중 오류: {e}")
            return {'overall_suitability': 0.5, 'is_suitable': True}
    
    def _extract_fatigue_indicators_from_pose(self, pose_2d: Dict, pose_3d: Dict) -> Dict[str, Any]:
        """자세에서 피로도 지표 추출"""
        fatigue_indicators = {
            'slouching': 0.0,
            'asymmetry': 0.0,
            'instability': 0.0,
            'overall_fatigue_from_pose': 0.0
        }
        
        try:
            # slouch는 피로의 대표적 지표
            if pose_3d.get('available') and 'slouch_detection' in pose_3d:
                fatigue_indicators['slouching'] = pose_3d['slouch_detection'].get('slouch_factor', 0)
            
            # 비대칭성도 피로 지표
            if pose_2d.get('available') and 'symmetry_analysis' in pose_2d:
                fatigue_indicators['asymmetry'] = 1.0 - pose_2d['symmetry_analysis'].get('symmetry_score', 1.0)
            
            # 자세 불안정성
            if pose_3d.get('available') and 'torso_geometry' in pose_3d:
                fatigue_indicators['instability'] = 1.0 - pose_3d['torso_geometry'].get('stability_score', 1.0)
            
            # 전체 피로도 (가중 평균)
            weights = [0.5, 0.3, 0.2]  # slouch, asymmetry, instability
            values = [fatigue_indicators['slouching'], fatigue_indicators['asymmetry'], fatigue_indicators['instability']]
            fatigue_indicators['overall_fatigue_from_pose'] = sum(w * v for w, v in zip(weights, values))
            
        except Exception as e:
            logger.error(f"자세 피로도 지표 추출 중 오류: {e}")
        
        return fatigue_indicators
    
    def _extract_distraction_indicators_from_pose(self, pose_2d: Dict, pose_3d: Dict) -> Dict[str, Any]:
        """자세에서 주의산만 지표 추출"""
        distraction_indicators = {
            'unusual_positioning': 0.0,
            'excessive_movement': 0.0,
            'asymmetric_posture': 0.0,
            'overall_distraction_from_pose': 0.0
        }
        
        try:
            # 비정상적인 위치 (복잡도로 측정)
            if pose_3d.get('available') and 'complexity' in pose_3d:
                distraction_indicators['unusual_positioning'] = pose_3d['complexity'].get('overall_complexity', 0)
            
            # 과도한 움직임 (추후 이력 기반으로 구현 가능)
            distraction_indicators['excessive_movement'] = 0.0  # 현재는 기본값
            
            # 비대칭적 자세
            if pose_2d.get('available') and 'symmetry_analysis' in pose_2d:
                distraction_indicators['asymmetric_posture'] = 1.0 - pose_2d['symmetry_analysis'].get('symmetry_score', 1.0)
            
            # 전체 주의산만 정도
            weights = [0.4, 0.3, 0.3]
            values = [distraction_indicators['unusual_positioning'], distraction_indicators['excessive_movement'], distraction_indicators['asymmetric_posture']]
            distraction_indicators['overall_distraction_from_pose'] = sum(w * v for w, v in zip(weights, values))
            
        except Exception as e:
            logger.error(f"자세 주의산만 지표 추출 중 오류: {e}")
        
        return distraction_indicators
    
    def _analyze_posture_trends(self) -> Dict[str, Any]:
        """자세 변화 추세 분석 (이력 기반)"""
        # 현재는 기본 구현, 추후 이력 데이터 활용하여 확장 가능
        return {
            'trend_available': False,
            'stability_trend': 'stable',
            'deterioration_rate': 0.0,
            'improvement_rate': 0.0
        }
    
    def _generate_posture_recommendation(self, suitability: float, issues: List[str]) -> str:
        """자세 개선 권장사항 생성"""
        if suitability > 0.8:
            return "우수한 운전 자세를 유지하고 있습니다."
        elif suitability > 0.6:
            return "전반적으로 양호한 자세입니다. 지속적인 유지를 권장합니다."
        elif issues:
            return f"다음 사항 개선 필요: {', '.join(issues)}"
        else:
            return "자세 교정이 필요합니다. 등받이를 조정하고 바른 자세로 앉아주세요."
    
    def _update_pose_metrics(self, pose_analysis: Dict[str, Any]):
        """자세 관련 메트릭 업데이트"""
        try:
            # 기본 메트릭 업데이트
            metrics_data = {
                'pose_complexity_score': pose_analysis.get('fatigue_indicators', {}).get('instability', 0.0),
                'slouch_factor': pose_analysis.get('fatigue_indicators', {}).get('slouching', 0.0),
                'posture_suitability': pose_analysis.get('driving_suitability', {}).get('overall_suitability', 0.5)
            }
            
            # 메트릭 업데이터를 통해 업데이트 (인터페이스가 지원하는 경우)
            if hasattr(self.metrics_updater, 'update_pose_metrics'):
                self.metrics_updater.update_pose_metrics(metrics_data)
                
        except Exception as e:
            logger.error(f"자세 메트릭 업데이트 중 오류: {e}")
    
    async def _handle_no_pose_detected(self) -> Dict[str, Any]:
        """자세가 감지되지 않은 상황 처리"""
        logger.warning("자세가 감지되지 않음 - 백업 모드 또는 센서 재보정 필요")
        
        return {
            'pose_detected': False,
            'pose_2d': {'available': False, 'reason': 'no_pose_landmarks'},
            'pose_3d': {'available': False, 'reason': 'no_world_landmarks'},
            'pose_analysis': {
                'overall_score': 0.0,
                'driving_suitability': {'overall_suitability': 0.0, 'is_suitable': False},
                'fatigue_indicators': {'overall_fatigue_from_pose': 0.0},
                'distraction_indicators': {'overall_distraction_from_pose': 0.0}
            }
        }