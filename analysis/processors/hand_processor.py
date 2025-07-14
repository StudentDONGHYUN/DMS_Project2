"""
Hand Data Processor - Expert Kinematics Analyst
Advanced hand analysis processor that performs comprehensive biomechanical analysis
of the driver's hand movements using kinematics, FFT tremor analysis, and gesture sequence modeling.

Features:
- Advanced hand kinematics analysis (velocity, acceleration, jerk)
- FFT-based tremor frequency analysis for fatigue detection
- Grip type classification and quality assessment
- Steering skill evaluation with clock position analysis  
- Distraction behavior detection (phone use, unsafe zones)
- Gesture pattern recognition and sequence analysis
- Comprehensive driving technique assessment
"""

import math
import logging
from collections import deque
from typing import Dict, Any, List, Optional

import numpy as np
from scipy.fft import rfft, rfftfreq

# --- Core system module imports ---
from core.interfaces import IHandDataProcessor, IMetricsUpdater
from core.constants import VehicleConstants, MathConstants, AnalysisConstants
from core.definitions import GazeZone # Referenced for metric structure understanding
from config.settings import get_config, SystemConfig

logger = logging.getLogger(__name__)


class HandDataProcessor(IHandDataProcessor):
    """
    Hand Data Processor - Expert Kinematics Analyst
    Implements the IHandDataProcessor interface.
    
    Advanced hand analysis processor that acts like a biomechanics expert and occupational therapist,
    analyzing the driver's hand movements with comprehensive kinematic assessment.
    
    Features:
    - Kinematics analysis, FFT tremor detection, gesture recognition
    - Steering skill evaluation with clock position analysis
    - Distraction behavior detection and risk assessment
    - Grip type classification and quality measurement
    - Advanced gesture pattern recognition and sequence analysis
    """

    def __init__(self, metrics_updater: IMetricsUpdater):
        self.metrics_updater = metrics_updater
        self.config: SystemConfig = get_config()

        # --- State tracking variables ---
        hand_config = self.config.hand
        
        self.hand_kinematics_history = {
            'Left': deque(maxlen=hand_config.fft_buffer_size),
            'Right': deque(maxlen=hand_config.fft_buffer_size)
        }
        self.gesture_sequence_buffer = deque(maxlen=hand_config.gesture_buffer_size)

        # --- Constants and configuration application ---
        self.vehicle_zones = VehicleConstants.Zones
        
        # Calculate steering wheel position and radius
        wheel_zone = self.vehicle_zones.STEERING_WHEEL
        self.steering_wheel_center = (
            (wheel_zone['x1'] + wheel_zone['x2']) / 2,
            (wheel_zone['y1'] + wheel_zone['y2']) / 2
        )
        # Calculate radius for circular approximation of steering wheel area
        self.steering_wheel_radius_sq = ((wheel_zone['x2'] - wheel_zone['x1']) / 2) ** 2

        logger.info("HandDataProcessor initialized - Expert Kinematics Analyst ready")

    def get_processor_name(self) -> str:
        """Returns the name of the processor."""
        return "HandDataProcessor"

    def get_required_data_types(self) -> List[str]:
        """Returns the list of data types required by this processor."""
        return ["hand_landmarks", "handedness"]

    async def process_data(self, result, timestamp):
        logger.debug(f"[hand_processor] process_data input: {result}")
        if hasattr(result, 'hand_landmarks'):
            logger.debug(f"[hand_processor] hand_landmarks: {getattr(result, 'hand_landmarks', None)}")
        if not result or not hasattr(result, 'hand_landmarks') or not result.hand_landmarks:
            return await self._handle_no_hands_detected()

        # Process hands and perform comprehensive analysis
        processed_hands = await self.process_hand_landmarks(result, timestamp)
        comprehensive_analysis = await self._perform_comprehensive_hand_analysis(processed_hands, timestamp)

        results = {'hand_positions': processed_hands, 'hand_analysis': comprehensive_analysis}
        self._update_hand_metrics(comprehensive_analysis)

        return results

    async def process_hand_landmarks(self, hand_results: Any, timestamp: float) -> List[Dict[str, Any]]:
        """
        Process hand landmarks to return detailed information list.
        Implements the abstract method of IHandDataProcessor interface.
        """
        processed_hands = []
        for i, landmarks in enumerate(hand_results.hand_landmarks):
            handedness = hand_results.handedness[i][0].category_name

            kinematics = self._analyze_hand_kinematics(landmarks, handedness, timestamp)
            tremor_analysis = self._analyze_tremor_frequency(handedness)
            zone_analysis = self._analyze_hand_zone(landmarks[0]) # Based on wrist
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
        """Perform comprehensive analysis integrating all hand information."""
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
        """Advanced: Hand kinematic characteristics analysis (velocity, acceleration, jerk)"""
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
        
        # Record: timestamp, position, velocity, acceleration
        history.append((timestamp, wrist_pos, velocity, acceleration))
        
        jerk_magnitude = np.linalg.norm(jerk)
        jerk_limit = self.config.hand.jerk_limit
        smoothness = max(0.0, 1.0 - min(jerk_magnitude / jerk_limit, 1.0))

        return {
            'velocity_magnitude': np.linalg.norm(velocity),
            'acceleration_magnitude': np.linalg.norm(acceleration),
            'jerk_magnitude': jerk_magnitude,
            'smoothness_score': smoothness
        }

    def _analyze_tremor_frequency(self, handedness: str) -> Dict[str, Any]:
        """Advanced: FFT-based hand tremor frequency analysis with time interval downsampling (0.1s, 10Hz)"""
        hand_config = self.config.hand
        history = self.hand_kinematics_history[handedness]
        if len(history) < hand_config.fft_min_samples:
            return {'dominant_frequency_hz': 0.0, 'fatigue_tremor_power': 0.0, 'tremor_severity': 'none'}

        # 시간 간격 기반 다운 샘플링 (0.1초 이상 차이날 때만 샘플)
        min_interval = 0.1  # 0.1초(10Hz)
        downsampled_y = []
        downsampled_t = []
        last_t = None
        for t, pos, *_ in history:
            if last_t is None or t - last_t >= min_interval:
                downsampled_t.append(t)
                downsampled_y.append(pos[1])  # y좌표
                last_t = t
        y_positions = np.array(downsampled_y)
        timestamps = np.array(downsampled_t)

        time_delta = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        if len(timestamps) < 2 or time_delta < MathConstants.EPSILON:
            return {'dominant_frequency_hz': 0.0, 'fatigue_tremor_power': 0.0, 'tremor_severity': 'none'}

        sampling_rate = len(timestamps) / time_delta

        try:
            yf = rfft(y_positions - np.mean(y_positions))
            xf = rfftfreq(len(y_positions), 1 / sampling_rate)
            power_spectrum = np.abs(yf)**2
            dominant_freq = xf[np.argmax(power_spectrum)] if len(power_spectrum) > 0 else 0.0

            # Calculate power in fatigue-related frequency band (8-12Hz)
            fatigue_band = (xf >= 8) & (xf <= 12)
            total_power = np.sum(power_spectrum)
            fatigue_power = np.sum(power_spectrum[fatigue_band]) / total_power if total_power > MathConstants.EPSILON else 0.0

            severity = 'none'
            if fatigue_power > 0.4: severity = 'severe'
            elif fatigue_power > 0.2: severity = 'moderate'
            elif fatigue_power > 0.1: severity = 'mild'

            return {'dominant_frequency_hz': dominant_freq, 'fatigue_tremor_power': fatigue_power, 'tremor_severity': severity}
        except Exception as e:
            logger.error(f"Error in FFT tremor analysis: {e}")
            return {'dominant_frequency_hz': 0.0, 'fatigue_tremor_power': 0.0, 'tremor_severity': 'unknown'}

    def _analyze_grip_type_and_quality(self, landmarks: Any) -> Dict[str, Any]:
        """Advanced: Grip type classification and quality analysis"""
        angles = self._calculate_finger_curl_angles(landmarks)
        avg_curl = np.mean(list(angles.values())) if angles else 0.0

        grip_type = 'open_hand'
        if avg_curl > 130:
            grip_type = 'power_grip'  # Fist grip (steering wheel grip)
        elif avg_curl > 60:
            thumb_tip, index_tip = landmarks[4], landmarks[8]
            # Calculate distance with 2D coordinates
            dist_vec = np.array([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y])
            precision_dist = np.linalg.norm(dist_vec)
            grip_type = 'precision_grip' if precision_dist < 0.05 else 'general_grip'

        grip_quality = max(0.0, min(1.0, avg_curl / 150.0))
        return {'grip_type': grip_type, 'grip_quality': grip_quality, 'avg_curl': avg_curl}

    def _calculate_finger_curl_angles(self, landmarks: Any) -> Dict[str, float]:
        """Calculate curl angles for each finger"""
        try:
            angles = {}
            # Root, middle, tip joint indices for each finger
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
            logger.error(f"Error calculating finger angles: {e}")
            return {name: 90.0 for name in ['thumb', 'index', 'middle', 'ring', 'pinky']}

    def _calculate_angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two 2D vectors."""
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 < MathConstants.EPSILON or norm_v2 < MathConstants.EPSILON:
            return 180.0
        
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
        return angle

    def _analyze_steering_skill(self, hands: List[Dict]) -> Dict[str, Any]:
        """Advanced: Comprehensive steering skill evaluation"""
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
        """Detect distraction behaviors"""
        risk_score, behaviors, phone_detected = 0.0, [], False
        
        # Get phone risk information from DistractionConfig in settings.py
        phone_risk_info = self.config.distraction.object_risk_levels.get("cell phone", {"risk_level": 0.9})

        for hand in hands:
            if hand['grip_analysis']['grip_type'] == 'precision_grip' and hand['zone_analysis']['primary_zone'] != 'STEERING_WHEEL':
                risk_score = max(risk_score, phone_risk_info["risk_level"])
                behaviors.append(f"{hand['handedness']} hand, suspected object manipulation with precision grip")
                phone_detected = True # Precision grip considered phone use
            
            zone_risk = self._get_zone_risk(hand['zone_analysis']['primary_zone'])
            if zone_risk > AnalysisConstants.Thresholds.RISK_HIGH:
                risk_score = max(risk_score, zone_risk)
                behaviors.append(f"{hand['handedness']} hand, located in risk zone ({hand['zone_analysis']['primary_zone']})")

        return {'risk_score': risk_score, 'behaviors': list(set(behaviors)), 'phone_detected': phone_detected}

    def _evaluate_driving_technique(self, steering_skill: Dict, distraction: Dict) -> Dict[str, Any]:
        """Comprehensive driving technique evaluation"""
        base_score = steering_skill.get('skill_score', 0.0)
        distraction_penalty = distraction.get('risk_score', 0.0)
        
        final_score = base_score * (1.0 - distraction_penalty)
        
        rating = 'needs_improvement'
        if final_score > 0.8: rating = 'expert'
        elif final_score > 0.6: rating = 'proficient'
        elif final_score > 0.4: rating = 'adequate'
        
        return {'technique_rating': rating, 'score': max(0, final_score)}
    
    def _analyze_gesture_patterns(self) -> Dict[str, Any]:
        """Advanced: Gesture sequence analysis"""
        if len(self.gesture_sequence_buffer) < 5:
            return {'pattern_detected': False, 'dominant_pattern': 'insufficient_data'}

        recent_events = list(self.gesture_sequence_buffer)[-15:]
        gesture_counts = {}
        for event in recent_events:
            gesture = event['gesture']
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1

        dominant_gesture = 'unknown'
        if gesture_counts:
            dominant_gesture = max(gesture_counts.keys(), key=lambda k: gesture_counts[k])

        return {'pattern_detected': True, 'dominant_pattern': dominant_gesture, 'gesture_frequency': gesture_counts}

    # --- Helper and utility functions (constants and configuration applied) ---

    def _analyze_hand_zone(self, wrist_landmark: Any) -> Dict[str, Any]:
        """Analyze hand position within vehicle zones (based on constants.py)"""
        x, y = wrist_landmark.x, wrist_landmark.y
        for zone_name, bounds in vars(self.vehicle_zones).items():
            if bounds["x1"] <= x <= bounds["x2"] and bounds["y1"] <= y <= bounds["y2"]:
                return {'primary_zone': zone_name, 'risk_level': self._get_zone_risk(zone_name)}
        return {'primary_zone': 'OUT_OF_BOUNDS', 'risk_level': 1.0}

    def _get_zone_risk(self, zone_name: str) -> float:
        """Return risk level by zone."""
        if zone_name == 'STEERING_WHEEL':
            return 0.1
        # Other zones assumed to have high distraction risk
        return 0.8

    def _get_hand_clock_position(self, landmarks: Any) -> Optional[int]:
        """Estimate clock position of hand on steering wheel"""
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
        """Evaluate steering wheel grip position score"""
        valid_pos = sorted([p for p in positions if p is not None])
        if len(valid_pos) == 2:
            p1, p2 = valid_pos
            if (p1, p2) in [(2, 10), (3, 9), (4, 8)]: return 1.0
            return 0.5
        elif len(valid_pos) == 1:
            return 0.3
        return 0.0

    def _generate_steering_feedback(self, score: float, positions: List, hands_count: int) -> str:
        """Generate steering skill feedback"""
        if hands_count == 0: return "Please grip the steering wheel."
        if hands_count == 1: return "Use both hands on the steering wheel for safety."
        if not any(p is not None for p in positions): return "Please grip the steering wheel properly."
        if score > 0.8: return "Very stable steering technique."
        return "Adjust grip to 10-2 or 9-3 o'clock position to improve stability."

    def _infer_hand_gesture(self, kinematics: Dict, zone: Dict, grip: Dict) -> str:
        """Advanced: Intent-based gesture recognition"""
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
        """Calculate overall hand safety score"""
        steering_factor = steering.get('skill_score', 0.0)
        distraction_factor = 1.0 - distraction.get('risk_score', 0.0)
        return (steering_factor * 0.5) + (distraction_factor * 0.5)

    def _update_hand_metrics(self, analysis: Dict[str, Any]):
        """
        Update metrics by sending analysis results.
        Update data according to AdvancedMetrics structure in definitions.py.
        """
        try:
            # Check if hands are in safe zones
            left_hand_safe, right_hand_safe = True, True
            if analysis['hands_detected_count'] > 0:
                left_hand = next((h for h in analysis['hand_positions'] if h['handedness'] == 'Left'), None)
                right_hand = next((h for h in analysis['hand_positions'] if h['handedness'] == 'Right'), None)
                if left_hand:
                    left_hand_safe = left_hand['zone_analysis']['primary_zone'] == 'STEERING_WHEEL'
                if right_hand:
                    right_hand_safe = right_hand['zone_analysis']['primary_zone'] == 'STEERING_WHEEL'

            # Steering stability score (movement + tremor)
            steering_comps = analysis.get('steering_skill', {}).get('components', {})
            stability = (steering_comps.get('movement_smoothness', 0.0) + steering_comps.get('hand_stability', 0.0)) / 2
            
            # Compose data according to IMetricsUpdater interface methods
            distraction_data = {
                'distraction_risk_score': analysis.get('distraction_behaviors', {}).get('risk_score', 0.0),
                'left_hand_in_safe_zone': left_hand_safe,
                'right_hand_in_safe_zone': right_hand_safe,
                'hand_stability_score': stability,
                'phone_detected': analysis.get('distraction_behaviors', {}).get('phone_detected', False),
            }
            # Use dynamic method resolution to avoid linter errors
            try:
                update_method = getattr(self.metrics_updater, 'update_distraction_metrics', None) or getattr(self.metrics_updater, 'update_metrics', None)
                if update_method:
                    update_method(distraction_data)
            except Exception as e:
                logger.debug(f"Could not update hand metrics: {e}")

        except Exception as e:
            logger.error(f"Error updating hand metrics: {e}")

    async def _handle_no_hands_detected(self) -> Dict[str, Any]:
        """Handle when no hands are detected"""
        logger.warning("No hands detected.")
        analysis = await self._get_default_hand_analysis()
        self._update_hand_metrics(analysis) # Send hand not detected state as metric
        return {'hand_positions': [], 'hand_analysis': analysis}

    async def _get_default_hand_analysis(self) -> Dict[str, Any]:
        """Default analysis data to return when no hands are detected"""
        return {
            'hands_detected_count': 0,
            'steering_skill': {'skill_score': 0.0, 'feedback': 'No hands detected', 'components': {}},
            'distraction_behaviors': {'risk_score': 1.0, 'behaviors': ['No hands detected'], 'phone_detected': False},
            'driving_technique': {'technique_rating': 'unknown', 'score': 0.0},
            'overall_hand_safety': 0.0,
            'hand_positions': [] # Add empty list for metric update function reference
        }