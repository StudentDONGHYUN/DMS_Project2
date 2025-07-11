"""
얼굴 데이터 전문 처리기
얼굴 랜드마크, 블렌드셰이프, 변환 행렬 등 얼굴 관련 모든 데이터를 전담 처리합니다.
"""

import math
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from cachetools import cached, TTLCache
from mediapipe.framework.formats import landmark_pb2

from core.interfaces import IFaceDataProcessor, IMetricsUpdater, IDrowsinessDetector, IEmotionRecognizer, IGazeClassifier, IDriverIdentifier
from core.constants import MediaPipeConstants, AnalysisConstants, MathConstants
from core.definitions import GazeZone, AnalysisEvent
from config.settings import get_config
import logging

logger = logging.getLogger(__name__)


class FaceDataProcessor(IFaceDataProcessor):
    """
    얼굴 데이터 전문 처리기
    
    이 클래스는 마치 얼굴 분석 전문의처럼 얼굴에서 나오는 모든 신호를 해석합니다.
    - 졸음 상태 분석 (눈 깜빡임, EAR 등)
    - 감정 상태 인식 (52개 블렌드셰이프 기반)
    - 머리 자세 분석 (yaw, pitch, roll)
    - 시선 방향 분류
    - 운전자 신원 확인
    """
    
    def __init__(
        self,
        metrics_updater: IMetricsUpdater,
        drowsiness_detector: IDrowsinessDetector,
        emotion_recognizer: IEmotionRecognizer,
        gaze_classifier: IGazeClassifier,
        driver_identifier: IDriverIdentifier
    ):
        self.metrics_updater = metrics_updater
        self.drowsiness_detector = drowsiness_detector
        self.emotion_recognizer = emotion_recognizer
        self.gaze_classifier = gaze_classifier
        self.driver_identifier = driver_identifier
        
        # 설정 로드
        self.config = get_config()
        
        # 현재 시선 상태 추적
        self.current_gaze_zone = GazeZone.FRONT
        self.gaze_zone_start_time = time.time()
        
        logger.info("FaceDataProcessor 초기화 완료 - 얼굴 분석 전문가 준비됨")
    
    def get_processor_name(self) -> str:
        return "FaceDataProcessor"
    
    def get_required_data_types(self) -> list:
        return ["face_landmarks", "face_blendshapes", "facial_transformation_matrixes"]
    
    async def process_data(self, data: Any, timestamp: float) -> Dict[str, Any]:
        """
        얼굴 데이터 통합 처리 메인 메서드
        
        이 메서드는 마치 지휘자처럼 모든 얼굴 분석 작업을 조율합니다.
        각 전문 분석기들에게 작업을 할당하고 결과를 통합합니다.
        """
        if not data or not data.face_landmarks:
            # 얼굴이 감지되지 않은 경우
            return await self._handle_no_face_detected()
        
        face_result = data
        landmarks = face_result.face_landmarks[0]
        
        # 병렬 처리 가능한 작업들을 비동기로 실행
        results = {}
        
        # 1. 졸음 감지 분석
        drowsiness_data = await self.process_drowsiness_analysis(landmarks, timestamp)
        results.update(drowsiness_data)
        
        # 2. 감정 인식 분석 (블렌드셰이프가 있는 경우)
        if face_result.face_blendshapes:
            emotion_data = await self.process_emotion_analysis(face_result.face_blendshapes[0], timestamp)
            results.update(emotion_data)
        
        # 3. 머리 자세 및 시선 분석
        if face_result.facial_transformation_matrixes:
            pose_gaze_data = await self.process_head_pose_and_gaze(
                face_result.facial_transformation_matrixes[0], timestamp
            )
            results.update(pose_gaze_data)
        
        # 4. 운전자 신원 확인 (캐시 사용으로 성능 최적화)
        driver_data = await self.process_driver_identification(landmarks)
        results.update(driver_data)
        
        # 5. 블렌드셰이프 기반 추가 분석
        if face_result.face_blendshapes:
            additional_data = await self.process_additional_blendshape_analysis(face_result.face_blendshapes[0])
            results.update(additional_data)
        
        # 메트릭 업데이트
        self.metrics_updater.update_drowsiness_metrics(results.get('drowsiness', {}))
        self.metrics_updater.update_emotion_metrics(results.get('emotion', {}))
        self.metrics_updater.update_gaze_metrics(results.get('gaze', {}))
        
        return results
    
    async def process_face_landmarks(self, landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """얼굴 랜드마크 전용 처리"""
        return await self.process_drowsiness_analysis(landmarks, timestamp)
    
    async def process_face_blendshapes(self, blendshapes: Any, timestamp: float) -> Dict[str, Any]:
        """얼굴 블렌드셰이프 전용 처리"""
        emotion_data = await self.process_emotion_analysis(blendshapes, timestamp)
        additional_data = await self.process_additional_blendshape_analysis(blendshapes)
        
        return {**emotion_data, **additional_data}
    
    async def process_facial_transformation(self, transformation: Any, timestamp: float) -> Dict[str, Any]:
        """얼굴 변환 행렬 전용 처리"""
        return await self.process_head_pose_and_gaze(transformation, timestamp)
    
    async def process_drowsiness_analysis(self, landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """
        졸음 감지 전문 분석
        
        마치 수면 전문의가 환자의 눈 상태를 면밀히 관찰하는 것처럼,
        눈의 미세한 움직임과 패턴을 분석하여 졸음 상태를 판단합니다.
        """
        try:
            # 고도화된 졸음 감지 실행
            drowsiness_result = self.drowsiness_detector.detect_drowsiness(landmarks, timestamp)
            
            return {
                'drowsiness': {
                    'status': drowsiness_result.get('status', 'unknown'),
                    'confidence': drowsiness_result.get('confidence', 0.0),
                    'enhanced_ear': drowsiness_result.get('enhanced_ear', 0.0),
                    'threshold': drowsiness_result.get('threshold', 0.25),
                    'perclos': drowsiness_result.get('perclos', 0.0),
                    'temporal_attention_score': drowsiness_result.get('temporal_attention_score', 0.0),
                    'microsleep': drowsiness_result.get('microsleep', {'detected': False})
                }
            }
        except Exception as e:
            logger.error(f"졸음 분석 중 오류 발생: {e}")
            return {'drowsiness': {'status': 'error', 'confidence': 0.0}}
    
    async def process_emotion_analysis(self, blendshapes: Any, timestamp: float) -> Dict[str, Any]:
        """
        감정 인식 전문 분석
        
        52개의 얼굴 블렌드셰이프를 분석하여 운전자의 감정 상태를 파악합니다.
        마치 심리학자가 표정을 읽는 것처럼 정교한 분석을 수행합니다.
        """
        try:
            emotion_result = self.emotion_recognizer.analyze_emotion(blendshapes, timestamp)
            
            return {
                'emotion': {
                    'state': emotion_result.get('emotion'),
                    'confidence': emotion_result.get('confidence', 0.0),
                    'arousal': emotion_result.get('arousal', 0.5),
                    'valence': emotion_result.get('valence', 0.5),
                    'stress_level': emotion_result.get('stress_level', 0.0)
                }
            }
        except Exception as e:
            logger.error(f"감정 분석 중 오류 발생: {e}")
            return {'emotion': {'state': None, 'confidence': 0.0}}
    
    async def process_head_pose_and_gaze(self, transformation_matrix: Any, timestamp: float) -> Dict[str, Any]:
        """
        머리 자세 및 시선 분석
        
        3D 변환 행렬을 분해하여 머리의 yaw, pitch, roll을 계산하고,
        이를 통해 운전자가 어느 방향을 보고 있는지 정확히 파악합니다.
        """
        try:
            if transformation_matrix is None:
                return self._get_default_pose_gaze_data()
            
            # 3D 변환 행렬에서 오일러 각도 추출
            head_pose = self._extract_euler_angles_from_matrix(transformation_matrix)
            
            # 시선 구역 분류
            new_gaze_zone = self.gaze_classifier.classify(
                head_pose['yaw'], head_pose['pitch'], timestamp
            )
            
            # 시선 구역 변경 감지 및 지속 시간 계산
            gaze_zone_duration = self._update_gaze_zone_tracking(new_gaze_zone)
            
            # 시선 안정성 및 주의집중 점수 계산
            gaze_stability = self.gaze_classifier.get_gaze_stability()
            attention_focus = self.gaze_classifier.get_attention_focus_score()
            
            return {
                'gaze': {
                    'head_yaw': head_pose['yaw'],
                    'head_pitch': head_pose['pitch'],
                    'head_roll': head_pose['roll'],
                    'current_zone': new_gaze_zone,
                    'zone_duration': gaze_zone_duration,
                    'stability': gaze_stability,
                    'attention_focus': attention_focus,
                    'deviation_score': self._calculate_gaze_deviation_score(
                        head_pose, new_gaze_zone, gaze_zone_duration, gaze_stability
                    )
                }
            }
        except Exception as e:
            logger.error(f"머리 자세 분석 중 오류 발생: {e}")
            return self._get_default_pose_gaze_data()
    
    async def process_driver_identification(self, landmarks: Any) -> Dict[str, Any]:
        """
        운전자 신원 확인
        
        얼굴의 기하학적 특징을 추출하여 운전자를 식별합니다.
        캐시를 사용하여 같은 얼굴에 대해 반복 계산을 방지합니다.
        """
        try:
            # 랜드마크를 튜플로 변환하여 캐시 키로 사용
            landmarks_tuple = tuple((lm.x, lm.y, lm.z) for lm in landmarks)
            driver_info = self._cached_identify_driver(landmarks_tuple)
            
            return {
                'driver': {
                    'identity': driver_info.get('driver_id', 'unknown'),
                    'confidence': driver_info.get('confidence', 0.0),
                    'is_new_driver': driver_info.get('is_new_driver', False)
                }
            }
        except Exception as e:
            logger.error(f"운전자 식별 중 오류 발생: {e}")
            return {'driver': {'identity': 'unknown', 'confidence': 0.0}}
    
    async def process_additional_blendshape_analysis(self, blendshapes: Any) -> Dict[str, Any]:
        """
        추가 블렌드셰이프 분석
        
        하품, 눈 깜빡임 등의 기본적인 행동 패턴을 분석합니다.
        """
        try:
            # 블렌드셰이프 딕셔너리 생성
            blendshapes_dict = {cat.category_name: cat.score for cat in blendshapes}
            
            # 하품 감지
            yawn_score = blendshapes_dict.get("jawOpen", 0.0)
            is_yawning = yawn_score > self.config.drowsiness.default_ear_threshold
            
            # 눈 감음 정도
            left_eye_closure = blendshapes_dict.get("eyeBlinkLeft", 0.0)
            right_eye_closure = blendshapes_dict.get("eyeBlinkRight", 0.0)
            avg_eye_closure = (left_eye_closure + right_eye_closure) / 2.0
            is_blinking = avg_eye_closure > AnalysisConstants.Thresholds.BLINK_EAR_THRESHOLD
            
            return {
                'additional_face_analysis': {
                    'yawn_score': yawn_score,
                    'is_yawning': is_yawning,
                    'left_eye_closure': left_eye_closure,
                    'right_eye_closure': right_eye_closure,
                    'avg_eye_closure': avg_eye_closure,
                    'is_blinking': is_blinking
                }
            }
        except Exception as e:
            logger.error(f"추가 블렌드셰이프 분석 중 오류 발생: {e}")
            return {'additional_face_analysis': {}}
    
    async def _handle_no_face_detected(self) -> Dict[str, Any]:
        """얼굴이 감지되지 않은 상황 처리"""
        logger.warning("얼굴이 감지되지 않음 - 백업 모드 필요")
        
        return {
            'face_detected': False,
            'drowsiness': {'status': 'no_face', 'confidence': 0.0},
            'emotion': {'state': None, 'confidence': 0.0},
            'gaze': self._get_default_pose_gaze_data()['gaze'],
            'driver': {'identity': 'unknown', 'confidence': 0.0}
        }
    
    def _extract_euler_angles_from_matrix(self, transform_matrix: Any) -> Dict[str, float]:
        """
        3D 변환 행렬에서 오일러 각도 추출
        
        복잡한 수학적 변환을 통해 머리의 3차원 회전을 
        사람이 이해하기 쉬운 yaw, pitch, roll 각도로 변환합니다.
        """
        try:
            # 4x4 변환 행렬에서 3x3 회전 행렬 추출
            R = np.array(transform_matrix.data).reshape(4, 4)[:3, :3]
            
            # 짐벌 락 회피를 위한 안전한 각도 추출
            sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            
            if sy > MathConstants.EPSILON:
                # 일반적인 경우
                x = math.atan2(R[2, 1], R[2, 2])  # roll
                y = math.atan2(-R[2, 0], sy)       # pitch  
                z = math.atan2(R[1, 0], R[0, 0])  # yaw
            else:
                # 짐벌 락이 발생한 경우
                x = math.atan2(-R[1, 2], R[1, 1])
                y = math.atan2(-R[2, 0], sy)
                z = 0
            
            # 라디안을 도로 변환하고 부호 조정
            return {
                'yaw': -math.degrees(y),
                'pitch': -math.degrees(x), 
                'roll': math.degrees(z)
            }
        except Exception as e:
            logger.error(f"오일러 각도 추출 중 오류 발생: {e}")
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
    
    def _update_gaze_zone_tracking(self, new_gaze_zone: GazeZone) -> float:
        """시선 구역 변경 추적 및 지속 시간 계산"""
        current_time = time.time()
        
        if new_gaze_zone != self.current_gaze_zone:
            # 새로운 구역으로 시선이 이동
            self.current_gaze_zone = new_gaze_zone
            self.gaze_zone_start_time = current_time
            zone_duration = 0.0
        else:
            # 같은 구역에서 지속
            zone_duration = current_time - self.gaze_zone_start_time
        
        return zone_duration
    
    def _calculate_gaze_deviation_score(
        self, head_pose: Dict[str, float], gaze_zone: GazeZone, 
        zone_duration: float, gaze_stability: float
    ) -> float:
        """
        시선 편차 점수 계산
        
        여러 요소를 종합하여 운전자의 시선이 얼마나 위험한지 평가합니다:
        - 머리 각도의 극단성
        - 현재 보고 있는 구역의 위험도
        - 해당 구역을 본 시간
        - 전반적인 시선 안정성
        """
        # 기본 머리 각도 기반 점수
        base_score = 0.0
        if abs(head_pose['yaw']) > AnalysisConstants.Thresholds.HEAD_YAW_EXTREME:
            base_score = min(1.0, 0.5 + zone_duration / 2.0)
        
        # 시선 불안정성 페널티
        instability_penalty = (1.0 - gaze_stability) * 0.3
        
        # 구역별 위험도 가중치
        zone_weights = {
            GazeZone.FRONT: 0.0,
            GazeZone.REARVIEW_MIRROR: 0.2,
            GazeZone.LEFT_SIDE_MIRROR: 0.2,
            GazeZone.RIGHT_SIDE_MIRROR: 0.2,
            GazeZone.INSTRUMENT_CLUSTER: 0.1,
            GazeZone.CENTER_STACK: 0.4,
            GazeZone.FLOOR: 0.8,
            GazeZone.ROOF: 0.6,
            GazeZone.PASSENGER: 0.7,
            GazeZone.DRIVER_WINDOW: 0.5,
            GazeZone.BLIND_SPOT_LEFT: 0.9,
        }
        
        zone_weight = zone_weights.get(gaze_zone, 0.5)
        duration_factor = min(1.0, zone_duration / 3.0)
        
        final_score = max(base_score, zone_weight * duration_factor) + instability_penalty
        return min(1.0, final_score)
    
    def _get_default_pose_gaze_data(self) -> Dict[str, Any]:
        """기본 자세 및 시선 데이터 반환"""
        return {
            'gaze': {
                'head_yaw': 0.0,
                'head_pitch': 0.0,
                'head_roll': 0.0,
                'current_zone': GazeZone.FRONT,
                'zone_duration': 0.0,
                'stability': 1.0,
                'attention_focus': 1.0,
                'deviation_score': 0.0
            }
        }
    
    @cached(cache=TTLCache(maxsize=5, ttl=300))
    def _cached_identify_driver(self, landmarks_tuple: Tuple) -> Dict[str, Any]:
        """
        캐시된 운전자 식별
        
        같은 얼굴에 대해 반복적으로 식별 작업을 수행하지 않도록
        5분간 결과를 캐시합니다. 이는 성능을 크게 향상시킵니다.
        """
        # 튜플을 다시 랜드마크 객체로 변환
        landmarks = [
            landmark_pb2.NormalizedLandmark(x=lm[0], y=lm[1], z=lm[2]) 
            for lm in landmarks_tuple
        ]
        return self.driver_identifier.identify_driver(landmarks)