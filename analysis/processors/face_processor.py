"""
Face Data Processor - Digital Psychologist
Advanced facial analysis processor that analyzes all biometric signals from the driver's face
using cognitive psychology, eye movement analysis, remote photoplethysmography (rPPG),
and other research-based methodologies.

Features:
- rPPG heart rate estimation from forehead region blood flow analysis
- Saccadic eye movement analysis for cognitive load assessment  
- Pupil dynamics analysis for cognitive state tracking
- EMA filtering for head pose stabilization and noise reduction
- Advanced emotion recognition using facial blendshapes
- Comprehensive gaze zone analysis and tracking
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
    Face Data Processor - Digital Psychologist
    
    Advanced facial analysis processor that implements the complete IFaceDataProcessor interface
    with enhanced S-Class features for comprehensive driver monitoring.
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

        # --- State tracking variables ---
        self.current_gaze_zone = GazeZone.FRONT
        self.gaze_zone_start_time = time.time()

        # EMA filter settings
        self.head_pose_history = deque(maxlen=self.config.face.ema_filter_size)
        self.ema_alpha = self.config.face.ema_alpha

        # Advanced S-Class: Eye movement and pupil analysis history buffers
        history_size = self.config.face.saccade_history_size
        self.left_eye_history = deque(maxlen=history_size)
        self.right_eye_history = deque(maxlen=history_size)
        self.pupil_size_history = deque(maxlen=history_size * 2)

        # Advanced S-Class: rPPG signal processing buffer
        rppg_buffer_size = int(self.config.rppg.fps * self.config.rppg.window_size_s)
        self.rppg_signal_buffer = deque(maxlen=rppg_buffer_size)

        logger.info("FaceDataProcessor initialized - Digital Psychologist ready")

    def get_processor_name(self) -> str:
        return "FaceDataProcessor"

    def get_required_data_types(self) -> List[str]:
        return ["face_landmarks", "face_blendshapes", "facial_transformation_matrixes"]

    # =============================================================================
    # Interface Implementation: Required abstract methods
    # =============================================================================
    
    async def process_face_landmarks(self, landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """
        Process facial landmarks exclusively (interface requirement)
        
        Performs drowsiness analysis and saccadic movement analysis from landmark data.
        The S-Class version includes advanced eye movement analysis.
        """
        try:
            # Basic drowsiness analysis
            drowsiness_data = await self.process_drowsiness_analysis(landmarks, timestamp)
            
            # Advanced S-Class: Saccadic eye movement analysis
            saccade_data = self._analyze_saccadic_movement(landmarks, timestamp)
            
            # Driver identification
            driver_data = await self.process_driver_identification(landmarks)
            
            # Integrate results
            result = {}
            result.update(drowsiness_data)
            result.update({'saccade': saccade_data})
            result.update(driver_data)
            
            return result
            
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error processing facial landmarks: {e}")
            return {
                'drowsiness': {'status': 'error', 'confidence': 0.0},
                'saccade': {'saccade_velocity_norm': 0.0, 'saccade_count_per_s': 0.0},
                'driver': {'identity': 'unknown', 'confidence': 0.0}
            }

    async def process_face_blendshapes(self, blendshapes: Any, timestamp: float) -> Dict[str, Any]:
        """
        Process facial blendshapes exclusively (interface requirement)
        
        Performs emotion analysis and pupil dynamics analysis from blendshape data.
        The S-Class version includes cognitive load analysis.
        """
        try:
            # Basic emotion analysis
            emotion_data = await self.process_emotion_analysis(blendshapes, timestamp)
            
            # Advanced S-Class: Pupil dynamics analysis
            pupil_data = self._analyze_pupil_dynamics(blendshapes)
            
            # Integrate results
            result = {}
            result.update(emotion_data)
            result.update({'pupil': pupil_data})
            
            return result
            
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error processing facial blendshapes: {e}")
            return {
                'emotion': {'state': None, 'confidence': 0.0},
                'pupil': {'estimated_pupil_diameter': 0.0, 'pupil_variability': 0.0}
            }

    async def process_facial_transformation(self, transformation: Any, timestamp: float) -> Dict[str, Any]:
        """
        Process facial transformation matrix exclusively (interface requirement)
        
        Performs head pose and gaze analysis from 3D transformation matrix.
        The S-Class version includes stabilized head pose tracking.
        """
        try:
            # Advanced S-Class: Head pose and gaze analysis with stabilization
            gaze_data = await self.process_head_pose_and_gaze(transformation, timestamp)
            
            return gaze_data
            
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error processing facial transformation matrix: {e}")
            return self._get_default_pose_gaze_data()

    # =============================================================================
    # Advanced S-Class Methods
    # =============================================================================

    async def process_data(self, data: Any, image: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """
        Comprehensive facial data processing (S-Class Digital Psychologist)
        
        Input:
            data: MediaPipe FaceLandmarker result object
                - face_landmarks: List[NormalizedLandmark] - 468 facial landmark coordinates
                - face_blendshapes: List[ClassificationResult] - 52 expression blendshapes
                - facial_transformation_matrixes: List[Matrix] - 4x4 3D transformation matrix
            image: np.ndarray - Original image for rPPG signal extraction
                - shape: (height, width, 3) - BGR color image
            timestamp: float - Current frame timestamp (seconds)
        
        Processing:
            1. rPPG heart rate analysis (forehead region blood flow changes)
            2. Drowsiness detection (PERCLOS, EAR, microsleep)
            3. Emotion recognition (7 basic emotions + stress analysis)
            4. Saccadic eye movement analysis (gaze fixation, saccade movements)
            5. Pupil dynamics analysis (cognitive load, arousal level)
            6. Head pose stabilization (EMA filtering)
            7. Gaze zone classification and tracking
            8. Driver identity verification
        
        Output:
            Dict[str, Any] with structure:
            {
                'face_detected': bool,
                'drowsiness': {
                    'status': str,  # 'awake', 'drowsy', 'microsleep'
                    'confidence': float,  # 0.0-1.0
                    'enhanced_ear': float,  # Enhanced EAR value
                    'perclos': float,  # Eye closure time ratio
                    'temporal_attention': float  # Temporal attention score
                },
                'emotion': {
                    'state': str,  # 'neutral', 'happy', 'stressed', etc.
                    'confidence': float,  # 0.0-1.0
                    'arousal_level': float  # Arousal level
                },
                'gaze': {
                    'head_yaw': float,  # Head left-right rotation (degrees)
                    'head_pitch': float,  # Head up-down rotation (degrees)
                    'head_roll': float,  # Head tilt (degrees)
                    'current_zone': GazeZone,  # Current gaze zone
                    'zone_duration': float,  # Zone duration (seconds)
                    'stability': float,  # Gaze stability 0.0-1.0
                    'attention_focus': float,  # Attention focus 0.0-1.0
                    'deviation_score': float  # Gaze deviation risk 0.0-1.0
                },
                'saccade': {
                    'saccade_velocity_norm': float,  # Normalized saccade velocity
                    'saccade_count_per_s': float,  # Saccades per second
                    'gaze_fixation_stability': float  # Gaze fixation stability
                },
                'pupil': {
                    'estimated_pupil_diameter': float,  # Estimated pupil size
                    'pupil_variability': float,  # Pupil change rate
                    'cognitive_load_indicator': float  # Cognitive load indicator
                },
                'rppg': {  # Generated periodically when rPPG buffer is full
                    'estimated_hr_bpm': float,  # Estimated heart rate (BPM)
                    'estimated_hrv_ms': float,  # Heart rate variability (ms)
                    'signal_quality': float  # Signal quality 0.0-1.0
                },
                'driver': {
                    'identity': str,  # Driver ID or 'unknown'
                    'confidence': float  # Identity confidence 0.0-1.0
                }
            }
        """
        logger.debug(f"[face_processor] process_data input: {data}")
        if hasattr(data, 'face_landmarks'):
            logger.debug(f"[face_processor] face_landmarks: {getattr(data, 'face_landmarks', None)}")
        if not data or not data.face_landmarks:
            return await self._handle_no_face_detected()

        landmarks = data.face_landmarks[0]
        results = {'face_detected': True}

        # 0.5. rPPG signal extraction (performed every frame)
        self._extract_rppg_signal(image, landmarks, timestamp)

        # 1. Basic analysis (drowsiness, emotion, pose, identity)
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

        # 2. Advanced S-Class cognitive/physiological state analysis
        results['saccade'] = self._analyze_saccadic_movement(landmarks, timestamp)

        if data.face_blendshapes:
            results['pupil'] = self._analyze_pupil_dynamics(data.face_blendshapes[0])

        # rPPG is resource-intensive so run periodically
        if int(timestamp * 10) % self.config.rppg.run_interval == 0:
            results['rppg'] = self._estimate_heart_rate_from_rppg()

        # 3. Calculate heart rate only when rPPG buffer is full
        if len(self.rppg_signal_buffer) == self.rppg_signal_buffer.maxlen:
            results['rppg'] = self._estimate_heart_rate_from_rppg()
        
        self._update_all_metrics(results)
        return results

    async def process_drowsiness_analysis(self, landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """Specialized drowsiness detection analysis"""
        try:
            drowsiness_result = self.drowsiness_detector.detect_drowsiness(landmarks, timestamp)
            return {'drowsiness': drowsiness_result}
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error in drowsiness analysis: {e}")
            return {'drowsiness': {'status': 'error', 'confidence': 0.0}}

    async def process_emotion_analysis(self, blendshapes: Any, timestamp: float) -> Dict[str, Any]:
        """Specialized emotion recognition analysis"""
        try:
            emotion_result = self.emotion_recognizer.analyze_emotion(blendshapes, timestamp)
            return {'emotion': emotion_result}
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error in emotion analysis: {e}")
            return {'emotion': {'state': None, 'confidence': 0.0}}

    async def process_head_pose_and_gaze(self, transformation_matrix: Any, timestamp: float) -> Dict[str, Any]:
        """Advanced head pose and gaze analysis"""
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
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error in head pose analysis: {e}")
            return self._get_default_pose_gaze_data()

    @cached(cache=TTLCache(maxsize=5, ttl=300))
    def _cached_identify_driver(self, landmarks_tuple: Tuple) -> Dict[str, Any]:
        """Cached driver identification"""
        landmarks = [landmark_pb2.NormalizedLandmark(x=lm[0], y=lm[1], z=lm[2]) for lm in landmarks_tuple]
        return self.driver_identifier.identify_driver(landmarks)

    async def process_driver_identification(self, landmarks: Any) -> Dict[str, Any]:
        """Driver identity verification"""
        try:
            landmarks_tuple = tuple((lm.x, lm.y, lm.z) for lm in landmarks)
            driver_info = self._cached_identify_driver(landmarks_tuple)
            return {'driver': driver_info}
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error in driver identification: {e}")
            return {'driver': {'identity': 'unknown', 'confidence': 0.0}}

    def _analyze_saccadic_movement(self, landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """Advanced S-Class: Saccadic eye movement and gaze fixation analysis"""
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
        except (AttributeError, TypeError, IndexError) as e:
            logger.error(f"Error in saccade analysis: {e}")
            return {'saccade_velocity_norm': 0.0, 'saccade_count_per_s': 0.0, 'gaze_fixation_stability': 0.0}

    def _calculate_eye_movement_metrics(self, history: deque) -> Dict[str, float]:
        """Calculate saccade and fixation metrics from single eye history"""
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
        """Advanced S-Class: Pupil response and variability analysis"""
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
        except (AttributeError, TypeError, KeyError) as e:
            logger.error(f"Error in pupil analysis: {e}")
            return {'estimated_pupil_diameter': 0.0, 'pupil_variability': 0.0, 'cognitive_load_indicator': 0.0}

    def _extract_rppg_signal(self, image: np.ndarray, landmarks: Any, timestamp: float):
        """Advanced S-Class: Extract average green channel value from forehead region for rPPG"""
        try:
            img_h, img_w, _ = image.shape
            
            # 1. Define forehead ROI landmarks
            forehead_indices = MediaPipeConstants.FaceROIs.FOREHEAD_ROI
            forehead_points = np.array(
                [[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in forehead_indices],
                dtype=np.int32
            )
            
            # 2. Create ROI mask
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [forehead_points], 255)
            
            # 3. Calculate average green channel value in ROI
            green_channel_mean = cv2.mean(image, mask=mask)[1] # G channel is index 1
            
            # 4. Store (timestamp, signal_value) in buffer
            if green_channel_mean > 0:
                self.rppg_signal_buffer.append((timestamp, green_channel_mean))
        except (AttributeError, TypeError, IndexError, cv2.error) as e:
            logger.error(f"Error in rPPG signal extraction: {e}")

    def _estimate_heart_rate_from_rppg(self) -> Dict[str, Any]:
        """
        Advanced S-Class: Calculate heart rate and HRV from buffered rPPG signal
        - Enhanced with 3-stage signal quality validation to discard unreliable HRV values
        """
        try:
            signal_data = list(self.rppg_signal_buffer)
            timestamps = np.array([item[0] for item in signal_data])
            raw_signal = np.array([item[1] for item in signal_data])

            # 1. Signal preprocessing: Detrending and band-pass filtering
            detrended_signal = detrend(raw_signal)
            fs = self.config.rppg.fps
            lowcut, highcut = self.config.rppg.low_cut_hz, self.config.rppg.high_cut_hz
            nyquist = 0.5 * fs
            low, high = lowcut / nyquist, highcut / nyquist
            b, a = butter(1, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, detrended_signal)

            # 2. Frequency analysis (FFT) and HR calculation
            N = len(filtered_signal)
            yf = rfft(filtered_signal)
            xf = rfftfreq(N, 1 / fs)
            freq_mask = (xf >= lowcut) & (xf <= highcut)
            
            if not np.any(freq_mask):
                raise ValueError("No valid frequency range found")

            fft_power = np.abs(yf[freq_mask])**2
            peak_freq_index = np.argmax(fft_power)
            peak_freq = xf[freq_mask][peak_freq_index]
            hr_bpm = peak_freq * 60

            # --- 3-stage HRV validation begins ---
            
            # Stage 3-1: FFT-based signal quality (SNR) validation
            peak_power = fft_power[peak_freq_index]
            noise_power = np.mean(np.delete(fft_power, peak_freq_index)) if len(fft_power) > 1 else 1
            snr = peak_power / noise_power if noise_power > 0 else 0
            
            signal_quality = 0.6  # Assume HR is calculated by default
            hrv_ms = 0.0

            if snr < self.config.rppg.snr_threshold:
                # SNR too low - signal quality poor, skip HRV calculation
                signal_quality = 0.3 # Low signal quality
                logger.warning(f"rPPG SNR below threshold({snr:.2f} < {self.config.rppg.snr_threshold}). Skipping HRV calculation.")
            else:
                # Stage 3-2: Reliable peak detection (using prominence)
                prominence_threshold = np.std(filtered_signal) * 0.4
                ibi_peaks, _ = find_peaks(filtered_signal, height=0, distance=fs*0.5, prominence=prominence_threshold)
                
                if len(ibi_peaks) > 3: # Need at least 3 IBIs for meaningful HRV
                    # Stage 3-3: IBI reasonableness validation
                    ibi_s = np.diff(timestamps[ibi_peaks])  # in seconds
                    
                    # Remove abnormal IBIs (e.g., >2s or <0.3s)
                    valid_ibi_mask = (ibi_s > 0.3) & (ibi_s < 2.0)
                    ibi_ms_valid = ibi_s[valid_ibi_mask] * 1000

                    if len(ibi_ms_valid) > 2:
                        hrv_std = np.std(ibi_ms_valid)
                        # If IBI standard deviation too high, consider as noise and discard
                        if hrv_std < self.config.rppg.hrv_std_threshold:
                            hrv_ms = hrv_std
                            signal_quality = 0.9 # HRV successfully calculated
                        else:
                            logger.warning(f"HRV standard deviation too high({hrv_std:.2f}ms). Considered noise and discarded.")
                            signal_quality = 0.5 # Trust HR but discard HRV
                    else:
                        signal_quality = 0.5 # Insufficient valid IBIs
                else:
                    signal_quality = 0.4 # Peak detection failed

            return {
                'estimated_hr_bpm': hr_bpm,
                'estimated_hrv_ms': hrv_ms,
                'signal_quality': signal_quality
            }

        except (ValueError, IndexError, np.linalg.LinAlgError) as e:
            logger.error(f"Error in rPPG heart rate calculation: {e}")
            return {'estimated_hr_bpm': 0, 'estimated_hrv_ms': 0, 'signal_quality': 0.1}

    def _extract_euler_angles_from_matrix(self, transform_matrix: Any) -> Dict[str, float]:
        """Extract Euler angles from 3D transformation matrix"""
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
        except (AttributeError, TypeError, IndexError, ValueError) as e:
            logger.error(f"Error extracting Euler angles: {e}")
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}

    def _stabilize_head_pose(self, raw_pose: Dict[str, float]) -> Dict[str, float]:
        """Advanced: Head pose stabilization using EMA filter"""
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
        """Track gaze zone changes and calculate duration"""
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
        """Advanced: Calculate gaze deviation risk score"""
        yaw_score = min(1.0, abs(head_pose['yaw']) / AnalysisConstants.Thresholds.HEAD_YAW_EXTREME)
        pitch_score = min(1.0, abs(head_pose['pitch']) / AnalysisConstants.Thresholds.HEAD_PITCH_LIMIT)
        base_angle_score = max(yaw_score, pitch_score)
        
        instability_penalty = (1.0 - gaze_stability) * 0.3
        
        zone_risk = AnalysisConstants.GazeZoneRisk.get(gaze_zone, 0.5)
        duration_factor = min(1.0, zone_duration / 3.0)
        
        final_score = (base_angle_score * 0.4) + (zone_risk * duration_factor * 0.4) + (instability_penalty * 0.2)
        return min(1.0, final_score)

    def _update_all_metrics(self, results: Dict[str, Any]):
        """Update all analysis results to central metrics manager"""
        if 'drowsiness' in results: self.metrics_updater.update_drowsiness_metrics(results['drowsiness'])
        if 'emotion' in results: self.metrics_updater.update_emotion_metrics(results['emotion'])
        if 'gaze' in results: self.metrics_updater.update_gaze_metrics(results['gaze'])
        # S-Class specific metrics updates handled safely
        if hasattr(self.metrics_updater, 'update_saccade_metrics') and 'saccade' in results:
            self.metrics_updater.update_saccade_metrics(results['saccade'])
        if hasattr(self.metrics_updater, 'update_pupil_metrics') and 'pupil' in results:
            self.metrics_updater.update_pupil_metrics(results['pupil'])
        if hasattr(self.metrics_updater, 'update_rppg_metrics') and 'rppg' in results:
            self.metrics_updater.update_rppg_metrics(results['rppg'])

    async def _handle_no_face_detected(self) -> Dict[str, Any]:
        """Return default values when no face is detected"""
        logger.warning("No face detected - setting all facial metrics to default values")
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
        """Default pose and gaze data"""
        return {
            'gaze': {
                'head_yaw': 0.0, 'head_pitch': 0.0, 'head_roll': 0.0,
                'current_zone': GazeZone.FRONT, 'zone_duration': 0.0,
                'stability': 1.0, 'attention_focus': 1.0, 'deviation_score': 0.0
            }
        }