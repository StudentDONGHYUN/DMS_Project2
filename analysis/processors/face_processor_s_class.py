"""
Face Processor (S-Class: 최종 통합 버전) - 추상 메서드 구현 완료
'디지털 심리학자'로서 운전자의 얼굴에서 나타나는 모든 생체 신호를
인지 심리학, 안구 운동 분석, 원격 광혈류측정(rPPG) 등의
연구 기반으로 심층 분석합니다.
"""

import asyncio
import logging
import math
import time
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np
from cachetools import TTLCache, cached
from mediapipe.framework.formats import landmark_pb2
from scipy.signal import butter, detrend, filtfilt, find_peaks
from scipy.fft import rfft, rfftfreq
import cv2

from config.settings import get_config
from core.constants import AnalysisConstants, MathConstants, MediaPipeConstants
from core.definitions import GazeZone
from core.interfaces import (IDriverIdentifier, IDrowsinessDetector,
                             IEmotionRecognizer, IFaceDataProcessor,
                             IGazeClassifier, IMetricsUpdater)

logger = logging.getLogger(__name__)


class FaceDataProcessor(IFaceDataProcessor):
    """
    얼굴 데이터 전문 처리기 (S-Class)
    
    🎯 수정 사항: 
    - 인터페이스에서 요구하는 세 개의 추상 메서드 구현 완료
    - 기존의 고급 기능들은 그대로 유지
    """

    def __init__(
        self,
        metrics_updater: IMetricsUpdater,
        drowsiness_detector: IDrowsinessDetector,
        emotion_recognizer: IEmotionRecognizer,
        gaze_classifier: IGazeClassifier,
        driver_identifier: IDriverIdentifier,
    ):
        self.metrics_updater = metrics_updater
        self.drowsiness_detector = drowsiness_detector
        self.emotion_recognizer = emotion_recognizer
        self.gaze_classifier = gaze_classifier
        self.driver_identifier = driver_identifier

        self.config = get_config()

        # --- 상태 추적 변수 ---
        self.current_gaze_zone = GazeZone.FRONT
        self.gaze_zone_start_time = time.time()

        # EMA 필터 설정
        self.head_pose_history = deque(maxlen=self.config.face.ema_filter_size)
        self.ema_alpha = self.config.face.ema_alpha

        # [S-Class] 안구 운동 및 동공 분석을 위한 이력 버퍼
        history_size = self.config.face.saccade_history_size
        self.left_eye_history = deque(maxlen=history_size)
        self.right_eye_history = deque(maxlen=history_size)
        self.pupil_size_history = deque(maxlen=history_size * 2)  # 동공은 좀 더 긴 이력 추적

        # [S-Class] rPPG 신호 처리를 위한 버퍼
        rppg_buffer_size = int(self.config.rppg.fps * self.config.rppg.window_size_s)
        self.rppg_signal_buffer = deque(maxlen=rppg_buffer_size)

        logger.info("FaceDataProcessor (S-Class) 초기화 완료 - 디지털 심리학자 준비됨")

    def get_processor_name(self) -> str:
        return "FaceDataProcessor"

    def get_required_data_types(self) -> List[str]:
        return ["face_landmarks", "face_blendshapes", "facial_transformation_matrixes"]

    # =============================================================================
    # 🎯 **추가된 부분**: 인터페이스에서 요구하는 세 개의 추상 메서드 구현
    # =============================================================================
    
    async def process_face_landmarks(self, landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """
        얼굴 랜드마크 전용 처리 (인터페이스 요구사항)
        
        이 메서드는 랜드마크 데이터를 받아서 졸음 분석과 사케이드 분석을 수행합니다.
        S-Class 버전에서는 고급 안구 운동 분석도 포함됩니다.
        """
        try:
            # 기본 졸음 분석
            drowsiness_data = await self.process_drowsiness_analysis(landmarks, timestamp)
            
            # [S-Class] 고급 안구 운동 분석
            saccade_data = self._analyze_saccadic_movement(landmarks, timestamp)
            
            # 운전자 신원 확인
            driver_data = await self.process_driver_identification(landmarks)
            
            # 결과 통합
            result = {}
            result.update(drowsiness_data)
            result.update({'saccade': saccade_data})
            result.update(driver_data)
            
            return result
            
        except Exception as e:
            logger.error(f"얼굴 랜드마크 처리 중 오류 발생: {e}")
            return {
                'drowsiness': {'status': 'error', 'confidence': 0.0},
                'saccade': {'saccade_velocity_norm': 0.0, 'saccade_count_per_s': 0.0},
                'driver': {'identity': 'unknown', 'confidence': 0.0}
            }

    async def process_face_blendshapes(self, blendshapes: Any, timestamp: float) -> Dict[str, Any]:
        """
        얼굴 블렌드셰이프 전용 처리 (인터페이스 요구사항)
        
        이 메서드는 블렌드셰이프 데이터를 받아서 감정 분석과 동공 분석을 수행합니다.
        S-Class 버전에서는 인지 부하 분석도 포함됩니다.
        """
        try:
            # 기본 감정 분석
            emotion_data = await self.process_emotion_analysis(blendshapes, timestamp)
            
            # [S-Class] 고급 동공 역학 분석
            pupil_data = self._analyze_pupil_dynamics(blendshapes)
            
            # 결과 통합
            result = {}
            result.update(emotion_data)
            result.update({'pupil': pupil_data})
            
            return result
            
        except Exception as e:
            logger.error(f"얼굴 블렌드셰이프 처리 중 오류 발생: {e}")
            return {
                'emotion': {'state': None, 'confidence': 0.0},
                'pupil': {'estimated_pupil_diameter': 0.0, 'pupil_variability': 0.0}
            }

    async def process_facial_transformation(self, transformation: Any, timestamp: float) -> Dict[str, Any]:
        """
        얼굴 변환 행렬 전용 처리 (인터페이스 요구사항)
        
        이 메서드는 3D 변환 행렬을 받아서 머리 자세와 시선 분석을 수행합니다.
        S-Class 버전에서는 안정화된 머리 자세 추적이 포함됩니다.
        """
        try:
            # [S-Class] 고급 머리 자세 및 시선 분석 (안정화 포함)
            gaze_data = await self.process_head_pose_and_gaze(transformation, timestamp)
            
            return gaze_data
            
        except Exception as e:
            logger.error(f"얼굴 변환 행렬 처리 중 오류 발생: {e}")
            return self._get_default_pose_gaze_data()

    # =============================================================================
    # 기존 S-Class 메서드들 (변경 없음)
    # =============================================================================

    async def process_data(self, data: Any, image: np.ndarray, timestamp: float) -> Dict[str, Any]:
        logger.debug(f"[face_processor_s_class] process_data input: {data}")
        if hasattr(data, 'face_landmarks'):
            logger.debug(f"[face_processor_s_class] face_landmarks: {getattr(data, 'face_landmarks', None)}")
        if not data or not data.face_landmarks:
            return await self._handle_no_face_detected()

        landmarks = data.face_landmarks[0]
        results = {'face_detected': True}

        # 0.5. rPPG 신호 추출 (매 프레임 수행)
        self._extract_rppg_signal(image, landmarks, timestamp)

        # 1. 기본 분석 (졸음, 감정, 자세, 신원)
        tasks = [
            self.process_drowsiness_analysis(landmarks, timestamp),
            self.process_driver_identification(landmarks)
        ]
        if data.face_blendshapes:
            tasks.append(self.process_emotion_analysis(data.face_blendshapes[0], timestamp))
        if data.facial_transformation_matrixes:
            tasks.append(self.process_head_pose_and_gaze(data.facial_transformation_matrixes[0], timestamp))

        analysis_results = await asyncio.gather(*tasks)
        for res in analysis_results:
            results.update(res)

        # 2. [S-Class] 고급 인지/생리 상태 분석
        results['saccade'] = self._analyze_saccadic_movement(landmarks, timestamp)

        if data.face_blendshapes:
            results['pupil'] = self._analyze_pupil_dynamics(data.face_blendshapes[0])

        # rPPG는 리소스 소모가 있으므로 주기적으로 실행
        if int(timestamp * 10) % self.config.rppg.run_interval == 0:
            results['rppg'] = self._estimate_heart_rate_from_rppg()

        # 3. 메트릭 업데이트
        #    rPPG 버퍼가 찼을 때만 심박수 계산
        if len(self.rppg_signal_buffer) == self.rppg_signal_buffer.maxlen:
            results['rppg'] = self._estimate_heart_rate_from_rppg()
        
        self._update_all_metrics(results)
        return results

    async def process_drowsiness_analysis(self, landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """ 졸음 감지 전문 분석 """
        try:
            drowsiness_result = self.drowsiness_detector.detect_drowsiness(landmarks, timestamp)
            return {'drowsiness': drowsiness_result}
        except Exception as e:
            logger.error(f"졸음 분석 중 오류 발생: {e}")
            return {'drowsiness': {'status': 'error', 'confidence': 0.0}}

    async def process_emotion_analysis(self, blendshapes: Any, timestamp: float) -> Dict[str, Any]:
        """ 감정 인식 전문 분석 """
        try:
            emotion_result = self.emotion_recognizer.analyze_emotion(blendshapes, timestamp)
            return {'emotion': emotion_result}
        except Exception as e:
            logger.error(f"감정 분석 중 오류 발생: {e}")
            return {'emotion': {'state': None, 'confidence': 0.0}}

    async def process_head_pose_and_gaze(self, transformation_matrix: Any, timestamp: float) -> Dict[str, Any]:
        """ [고도화] 머리 자세 및 시선 분석 """
        try:
            raw_head_pose = self._extract_euler_angles_from_matrix(transformation_matrix)
            stable_head_pose = self._stabilize_head_pose(raw_head_pose)
            new_gaze_zone = self.gaze_classifier.classify(stable_head_pose['yaw'], stable_head_pose['pitch'], timestamp)
            gaze_zone_duration = self._update_gaze_zone_tracking(new_gaze_zone)
            gaze_stability = self.gaze_classifier.get_gaze_stability()
            attention_focus = self.gaze_classifier.get_attention_focus_score()
            deviation_score = self._calculate_gaze_deviation_score(
                stable_head_pose, new_gaze_zone, gaze_zone_duration, gaze_stability
            )
            return {
                'gaze': {
                    'head_yaw': stable_head_pose['yaw'],
                    'head_pitch': stable_head_pose['pitch'],
                    'head_roll': stable_head_pose['roll'],
                    'current_zone': new_gaze_zone,
                    'zone_duration': gaze_zone_duration,
                    'stability': gaze_stability,
                    'attention_focus': attention_focus,
                    'deviation_score': deviation_score
                }
            }
        except Exception as e:
            logger.error(f"머리 자세 분석 중 오류 발생: {e}")
            return self._get_default_pose_gaze_data()

    @cached(cache=TTLCache(maxsize=5, ttl=300))
    def _cached_identify_driver(self, landmarks_tuple: Tuple) -> Dict[str, Any]:
        """ [고도화] 캐시된 운전자 식별 """
        landmarks = [landmark_pb2.NormalizedLandmark(x=lm[0], y=lm[1], z=lm[2]) for lm in landmarks_tuple]
        return self.driver_identifier.identify_driver(landmarks)

    async def process_driver_identification(self, landmarks: Any) -> Dict[str, Any]:
        """ 운전자 신원 확인 """
        try:
            landmarks_tuple = tuple((lm.x, lm.y, lm.z) for lm in landmarks)
            driver_info = self._cached_identify_driver(landmarks_tuple)
            return {'driver': driver_info}
        except Exception as e:
            logger.error(f"운전자 식별 중 오류 발생: {e}")
            return {'driver': {'identity': 'unknown', 'confidence': 0.0}}

    def _analyze_saccadic_movement(self, landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """ [S-Class] 안구 도약 운동(Saccade) 및 시선 고정(Fixation) 실제 구현 """
        try:
            left_iris_lm = landmarks[MediaPipeConstants.EyeLandmarks.LEFT_IRIS_CENTER]
            right_iris_lm = landmarks[MediaPipeConstants.EyeLandmarks.RIGHT_IRIS_CENTER]
            current_left_pos = np.array([left_iris_lm.x, left_iris_lm.y])
            current_right_pos = np.array([right_iris_lm.x, right_iris_lm.y])

            self.left_eye_history.append({'time': timestamp, 'pos': current_left_pos})
            self.right_eye_history.append({'time': timestamp, 'pos': current_right_pos})

            if len(self.left_eye_history) < self.config.face.saccade_min_samples:
                return {'saccade_velocity_norm': 0.0, 'saccade_count_per_s': 0.0, 'gaze_fixation_stability': 1.0}

            left_analysis = self._calculate_eye_movement_metrics(self.left_eye_history)
            right_analysis = self._calculate_eye_movement_metrics(self.right_eye_history)

            return {
                'saccade_velocity_norm': (left_analysis['velocity'] + right_analysis['velocity']) / 2,
                'saccade_count_per_s': (left_analysis['count'] + right_analysis['count']) / 2,
                'gaze_fixation_stability': (left_analysis['stability'] + right_analysis['stability']) / 2
            }
        except Exception as e:
            logger.error(f"Saccade 분석 중 오류: {e}")
            return {'saccade_velocity_norm': 0.0, 'saccade_count_per_s': 0.0, 'gaze_fixation_stability': 0.0}

    def _calculate_eye_movement_metrics(self, history: deque) -> Dict[str, float]:
        """ 단일 눈의 이력 데이터를 받아 Saccade 및 Fixation 지표를 계산하는 헬퍼 함수 """
        timestamps = np.array([item['time'] for item in history])
        positions = np.array([item['pos'] for item in history])
        dt = np.diff(timestamps)
        dt[dt == 0] = 1e-6
        ds = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        velocities = ds / dt

        saccade_velocity_threshold = self.config.face.saccade_velocity_threshold
        peaks, properties = find_peaks(velocities, height=saccade_velocity_threshold, distance=3)
        total_duration = timestamps[-1] - timestamps[0]
        if total_duration == 0:
            total_duration = 1.0

        saccade_count_per_s = len(peaks) / total_duration
        avg_saccade_velocity = np.mean(properties['peak_heights']) if len(peaks) > 0 else 0.0

        is_fixation = velocities < saccade_velocity_threshold
        fixation_points = positions[1:][is_fixation]

        if len(fixation_points) > 2:
            dispersion = np.mean(np.std(fixation_points, axis=0))
            stability_score = max(0.0, 1.0 - dispersion / self.config.face.fixation_dispersion_max)
        else:
            stability_score = 1.0

        return {'velocity': avg_saccade_velocity, 'count': saccade_count_per_s, 'stability': stability_score}

    def _analyze_pupil_dynamics(self, blendshapes: Any) -> Dict[str, Any]:
        """ [S-Class] 동공 반응 및 변화율 분석 """
        try:
            blendshapes_dict = {cat.category_name: cat.score for cat in blendshapes}
            pupil_size_y = (blendshapes_dict.get('eyeLookUpLeft', 0) - blendshapes_dict.get('eyeLookDownLeft', 0))
            pupil_size_x = (blendshapes_dict.get('eyeLookOutLeft', 0) - blendshapes_dict.get('eyeLookInLeft', 0))
            pupil_diameter_est = np.linalg.norm([pupil_size_x, pupil_size_y])
            self.pupil_size_history.append(pupil_diameter_est)

            pupil_variability = np.std(self.pupil_size_history) if len(self.pupil_size_history) > 1 else 0.0
            cognitive_load = (blendshapes_dict.get('browDownLeft', 0) + blendshapes_dict.get('browDownRight', 0)) / 2

            return {
                'estimated_pupil_diameter': pupil_diameter_est,
                'pupil_variability': pupil_variability,
                'cognitive_load_indicator': cognitive_load
            }
        except Exception as e:
            logger.error(f"동공 분석 중 오류: {e}")
            return {'estimated_pupil_diameter': 0.0, 'pupil_variability': 0.0, 'cognitive_load_indicator': 0.0}

    def _extract_rppg_signal(self, image: np.ndarray, landmarks: Any, timestamp: float):
        """ [S-Class] 매 프레임에서 이마 영역의 평균 녹색 채널 값을 추출하여 버퍼에 저장 """
        try:
            img_h, img_w, _ = image.shape
            
            # 1. 이마 ROI 랜드마크 정의
            forehead_indices = MediaPipeConstants.FaceROIs.FOREHEAD_ROI
            forehead_points = np.array(
                [[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in forehead_indices],
                dtype=np.int32
            )
            
            # 2. ROI 마스크 생성
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [forehead_points], 255)
            
            # 3. ROI 영역의 평균 녹색 채널 값 계산
            green_channel_mean = cv2.mean(image, mask=mask)[1] # G 채널은 인덱스 1
            
            # 4. 버퍼에 (타임스탬프, 신호값) 저장
            if green_channel_mean > 0:
                self.rppg_signal_buffer.append((timestamp, green_channel_mean))
        except Exception as e:
            logger.error(f"rPPG 신호 추출 중 오류: {e}")

    def _estimate_heart_rate_from_rppg(self) -> Dict[str, Any]:
        """
        [S-Class] 버퍼링된 rPPG 신호로 심박수 및 HRV를 계산하는 실제 구현
        - [고도화] 3단계 신호 품질 검증을 통해 신뢰할 수 없는 HRV 값을 폐기
        """
        try:
            signal_data = list(self.rppg_signal_buffer)
            timestamps = np.array([item[0] for item in signal_data])
            raw_signal = np.array([item[1] for item in signal_data])

            # 1. 신호 전처리: Detrending 및 Band-pass 필터링
            detrended_signal = detrend(raw_signal)
            fs = self.config.rppg.fps
            lowcut, highcut = self.config.rppg.low_cut_hz, self.config.rppg.high_cut_hz
            nyquist = 0.5 * fs
            low, high = lowcut / nyquist, highcut / nyquist
            b, a = butter(1, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, detrended_signal)

            # 2. 주파수 분석 (FFT) 및 HR 계산
            N = len(filtered_signal)
            yf = rfft(filtered_signal)
            xf = rfftfreq(N, 1 / fs)
            freq_mask = (xf >= lowcut) & (xf <= highcut)
            
            if not np.any(freq_mask):
                raise ValueError("유효 주파수 대역을 찾을 수 없음")

            fft_power = np.abs(yf[freq_mask])**2
            peak_freq_index = np.argmax(fft_power)
            peak_freq = xf[freq_mask][peak_freq_index]
            hr_bpm = peak_freq * 60

            # --- HRV 계산을 위한 3단계 검증 시작 ---
            
            # 3-1단계: FFT 기반 신호 품질(SNR) 검증
            peak_power = fft_power[peak_freq_index]
            noise_power = np.mean(np.delete(fft_power, peak_freq_index)) if len(fft_power) > 1 else 1
            snr = peak_power / noise_power if noise_power > 0 else 0
            
            signal_quality = 0.6  # 기본적으로 HR은 계산되었다고 가정
            hrv_ms = 0.0

            if snr < self.config.rppg.snr_threshold:
                # SNR이 너무 낮으면 신호 품질이 나빠 HRV 계산을 시도하지 않음
                signal_quality = 0.3 # 신호 품질 낮음
                logger.warning(f"rPPG SNR이 임계값 미만({snr:.2f} < {self.config.rppg.snr_threshold}). HRV 계산을 건너뜁니다.")
            else:
                # 3-2단계: 신뢰도 높은 피크 탐지 (prominence 사용)
                prominence_threshold = np.std(filtered_signal) * 0.4
                ibi_peaks, _ = find_peaks(filtered_signal, height=0, distance=fs*0.5, prominence=prominence_threshold)
                
                if len(ibi_peaks) > 3: # 최소 3개 이상의 IBI가 있어야 HRV가 의미 있음
                    # 3-3단계: IBI 합리성 검증
                    ibi_s = np.diff(timestamps[ibi_peaks])  # 초 단위
                    
                    # 비정상적인 IBI 제거 (예: 2초 이상 또는 0.3초 미만)
                    valid_ibi_mask = (ibi_s > 0.3) & (ibi_s < 2.0)
                    ibi_ms_valid = ibi_s[valid_ibi_mask] * 1000

                    if len(ibi_ms_valid) > 2:
                        hrv_std = np.std(ibi_ms_valid)
                        # IBI의 표준편차가 너무 크면 노이즈로 간주하고 폐기
                        if hrv_std < self.config.rppg.hrv_std_threshold:
                            hrv_ms = hrv_std
                            signal_quality = 0.9 # HRV까지 성공적으로 계산됨
                        else:
                            logger.warning(f"HRV 표준편차가 너무 큽니다({hrv_std:.2f}ms). 노이즈로 간주하여 폐기합니다.")
                            signal_quality = 0.5 # HR은 신뢰하지만 HRV는 폐기
                    else:
                        signal_quality = 0.5 # 유효한 IBI가 부족
                else:
                    signal_quality = 0.4 # 피크 검출 실패

            return {
                'estimated_hr_bpm': hr_bpm,
                'estimated_hrv_ms': hrv_ms,
                'signal_quality': signal_quality
            }

        except Exception as e:
            logger.error(f"rPPG 심박수 계산 중 오류: {e}")
            return {'estimated_hr_bpm': 0, 'estimated_hrv_ms': 0, 'signal_quality': 0.1}

    def _extract_euler_angles_from_matrix(self, transform_matrix: Any) -> Dict[str, float]:
        """ 3D 변환 행렬에서 오일러 각도 추출 """
        try:
            R = np.array(transform_matrix).reshape(4, 4)[:3, :3]
            sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            
            if sy > MathConstants.EPSILON:
                x = math.atan2(R[2, 1], R[2, 2])
                y = math.atan2(-R[2, 0], sy)
                z = math.atan2(R[1, 0], R[0, 0])
            else:
                x = math.atan2(-R[1, 2], R[1, 1])
                y = math.atan2(-R[2, 0], sy)
                z = 0
            
            return {'yaw': -math.degrees(y), 'pitch': -math.degrees(x), 'roll': math.degrees(z)}
        except Exception as e:
            logger.error(f"오일러 각도 추출 중 오류 발생: {e}")
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}

    def _stabilize_head_pose(self, raw_pose: Dict[str, float]) -> Dict[str, float]:
        """ [고도화] EMA 필터를 이용한 머리 자세 안정화 """
        if not self.head_pose_history:
            stable_pose = raw_pose
        else:
            last_pose = self.head_pose_history[-1]
            stable_pose = {
                'yaw': self.ema_alpha * raw_pose['yaw'] + (1 - self.ema_alpha) * last_pose['yaw'],
                'pitch': self.ema_alpha * raw_pose['pitch'] + (1 - self.ema_alpha) * last_pose['pitch'],
                'roll': self.ema_alpha * raw_pose['roll'] + (1 - self.ema_alpha) * last_pose['roll'],
            }
        self.head_pose_history.append(stable_pose)
        return stable_pose

    def _update_gaze_zone_tracking(self, new_gaze_zone: GazeZone) -> float:
        """ 시선 구역 변경 추적 및 지속 시간 계산 """
        current_time = time.time()
        if new_gaze_zone != self.current_gaze_zone:
            self.current_gaze_zone = new_gaze_zone
            self.gaze_zone_start_time = current_time
            return 0.0
        else:
            return current_time - self.gaze_zone_start_time

    def _calculate_gaze_deviation_score(
        self, head_pose: Dict[str, float], gaze_zone: GazeZone,
        zone_duration: float, gaze_stability: float
    ) -> float:
        """ [고도화] 시선 편차 위험도 점수 계산 """
        yaw_score = min(1.0, abs(head_pose['yaw']) / AnalysisConstants.Thresholds.HEAD_YAW_EXTREME)
        pitch_score = min(1.0, abs(head_pose['pitch']) / AnalysisConstants.Thresholds.HEAD_PITCH_LIMIT)
        base_angle_score = max(yaw_score, pitch_score)
        
        instability_penalty = (1.0 - gaze_stability) * 0.3
        
        zone_risk = AnalysisConstants.GazeZoneRisk.get(gaze_zone, 0.5)
        duration_factor = min(1.0, zone_duration / 3.0)
        
        final_score = (base_angle_score * 0.4) + (zone_risk * duration_factor * 0.4) + (instability_penalty * 0.2)
        return min(1.0, final_score)

    def _update_all_metrics(self, results: Dict[str, Any]):
        """ 모든 분석 결과를 중앙 메트릭 관리자에 업데이트 """
        if 'drowsiness' in results: self.metrics_updater.update_drowsiness_metrics(results['drowsiness'])
        if 'emotion' in results: self.metrics_updater.update_emotion_metrics(results['emotion'])
        if 'gaze' in results: self.metrics_updater.update_gaze_metrics(results['gaze'])
        # S-Class 전용 메트릭 업데이트도 안전하게 처리
        if hasattr(self.metrics_updater, 'update_saccade_metrics') and 'saccade' in results:
            self.metrics_updater.update_saccade_metrics(results['saccade'])
        if hasattr(self.metrics_updater, 'update_pupil_metrics') and 'pupil' in results:
            self.metrics_updater.update_pupil_metrics(results['pupil'])
        if hasattr(self.metrics_updater, 'update_rppg_metrics') and 'rppg' in results:
            self.metrics_updater.update_rppg_metrics(results['rppg'])

    async def _handle_no_face_detected(self) -> Dict[str, Any]:
        """ 얼굴 미감지 시 기본값 반환 """
        logger.warning("얼굴이 감지되지 않음 - 모든 얼굴 관련 지표를 기본값으로 설정")
        default_gaze = self._get_default_pose_gaze_data()['gaze']
        return {
            'face_detected': False,
            'drowsiness': {'status': 'no_face', 'confidence': 0.0},
            'emotion': {'state': None, 'confidence': 0.0},
            'gaze': default_gaze,
            'driver': {'identity': 'unknown', 'confidence': 0.0},
            'saccade': {'saccade_velocity_norm': 0.0, 'saccade_count_per_s': 0.0, 'gaze_fixation_stability': 0.0},
            'pupil': {'estimated_pupil_diameter': 0.0, 'pupil_variability': 0.0, 'cognitive_load_indicator': 0.0},
            'rppg': {'estimated_hr_bpm': 0, 'estimated_hrv_ms': 0, 'signal_quality': 0.0}
        }

    def _get_default_pose_gaze_data(self) -> Dict[str, Any]:
        """ 기본 자세 및 시선 데이터 """
        return {
            'gaze': {
                'head_yaw': 0.0, 'head_pitch': 0.0, 'head_roll': 0.0,
                'current_zone': GazeZone.FRONT, 'zone_duration': 0.0,
                'stability': 1.0, 'attention_focus': 1.0, 'deviation_score': 0.0
            }
        }