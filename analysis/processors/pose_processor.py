"""
Pose Data Processor - Digital Biomechanics Expert
Advanced posture analysis processor that performs comprehensive biomechanical analysis
of the driver's body posture using 3D landmark-based spinal alignment and postural sway measurement.

Features:
- 3D spinal alignment analysis with forward head posture detection
- Postural sway measurement for fatigue detection through torso center micro-movements
- Biomechanical health scoring system integrating multiple posture indicators
- Posture trend analysis and predictive health assessment
- Enhanced slouching detection with severity classification and health recommendations
- Comprehensive driving posture suitability evaluation
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
    Pose Data Processor - Digital Biomechanics Expert
    
    Advanced posture analysis processor that acts like a physical therapist or posture specialist,
    analyzing the driver's body posture with comprehensive biomechanical assessment.
    
    Features:
    - Shoulder and torso orientation analysis
    - Posture complexity and stability evaluation
    - Slouching detection with spinal health assessment
    - Body balance and symmetry analysis
    - Advanced S-Class: 3D spinal alignment and forward head posture analysis
    - Advanced S-Class: Postural sway quantitative measurement
    - Advanced S-Class: Posture trend prediction and health recommendations
    """
    
    def __init__(self, metrics_updater: IMetricsUpdater):
        self.metrics_updater = metrics_updater
        self.config = get_config()
        
        # --- Advanced state tracking variables ---
        self.posture_history = deque(maxlen=300) # 10 seconds (30fps*10) history
        self.torso_center_history = deque(maxlen=150) # 5 seconds of torso center tracking

        logger.info("PoseDataProcessor initialized - Digital Biomechanics Expert ready")
    
    def get_processor_name(self) -> str:
        return "PoseDataProcessor"
    
    def get_required_data_types(self) -> List[str]:
        return ["pose_landmarks", "pose_world_landmarks"]
    
    async def process_data(self, result, timestamp):
        """
        Comprehensive posture data processing (Digital Biomechanics Expert)
        
        Input:
            result: MediaPipe PoseLandmarker result object
                - pose_landmarks: List[NormalizedLandmark] - 33 2D pose landmarks
                - pose_world_landmarks: List[Landmark] - 33 3D world coordinate landmarks
            timestamp: float - Current frame timestamp (seconds)
        
        Processing:
            1. 3D spinal alignment analysis (forward head posture, spinal curvature)
            2. Postural sway measurement (Postural Sway Area, Velocity)
            3. Slouching detection (Slouching Detection)
            4. Shoulder tilt and left-right symmetry analysis
            5. Biomechanical health score calculation
            6. Posture trend analysis and prediction
            7. Driving posture suitability assessment
            8. Fatigue/distraction indicator extraction
        
        Output:
            Dict[str, Any] with structure:
            {
                'pose_2d': {
                    'available': bool,
                    'shoulder_analysis': {
                        'angle_degrees': float,  # Shoulder tilt angle
                        'tilt_severity': float   # Tilt severity 0.0-1.0
                    },
                    'symmetry_analysis': {
                        'symmetry_score': float  # Left-right symmetry 0.0-1.0
                    }
                },
                'pose_3d': {
                    'available': bool,
                    'spinal_analysis': {
                        'forward_head_posture_angle': float,  # Forward head posture angle (degrees)
                        'spine_curvature_angle': float,       # Spinal curvature angle (degrees)
                        'neck_health_score': float,           # Neck health 0.0-1.0
                        'spine_health_score': float,          # Spine health 0.0-1.0
                        'cervical_risk_level': str            # 'low', 'medium', 'high'
                    },
                    'postural_sway': {
                        'sway_area_cm2': float,      # Postural sway area (cm²)
                        'sway_velocity_cm_s': float, # Sway velocity (cm/s)
                        'stability_score': float,    # Stability score 0.0-1.0
                        'sway_pattern': str          # 'lateral_dominant', 'anterior_posterior_dominant', 'circular_pattern'
                    },
                    'slouch_detection': {
                        'slouch_angle': float,    # Slouching angle (degrees)
                        'slouch_factor': float,   # Slouching severity 0.0-1.0
                        'is_slouching': bool,     # Whether slouching
                        'severity': str           # 'mild', 'moderate', 'severe'
                    },
                    'balance': {
                        'balance_score': float    # Body balance score 0.0-1.0
                    },
                    'complexity': {
                        'overall_complexity': float  # Posture complexity 0.0-1.0
                    }
                },
                'pose_analysis': {
                    'driving_suitability': float,         # Driving posture suitability 0.0-1.0
                    'identified_issues': List[str],       # Identified issues
                    'recommendation': str,                # Posture improvement recommendation
                    'fatigue_indicators': {
                        'slouching': float,      # Fatigue from slouching 0.0-1.0
                        'instability': float     # Fatigue from instability 0.0-1.0
                    },
                    'distraction_indicators': {
                        'unusual_positioning': float  # Distraction from unusual posture 0.0-1.0
                    },
                    'trend_analysis': {
                        'trend_available': bool,
                        'stability_trend': str,          # 'stable', 'improving', 'deteriorating'
                        'deterioration_rate': float,     # Deterioration rate
                        'health_trend': str,             # Health trend
                        'prediction': str                # Future prediction message
                    },
                    'biomechanical_health': {
                        'overall_score': float,          # Overall biomechanical health 0.0-1.0
                        'spinal_health': float,          # Spinal health
                        'neck_health': float,            # Neck health
                        'stability': float,              # Postural stability
                        'posture_quality': float,        # Posture quality
                        'risk_factors': List[str],       # Risk factors
                        'recommendations': List[str]     # Improvement recommendations
                    },
                    'data_quality': {
                        'pose_2d': bool,  # 2D data availability
                        'pose_3d': bool   # 3D data availability
                    }
                }
            }
        """
        logger.debug(f"[pose_processor] process_data input: {result}")
        if hasattr(result, 'pose_landmarks'):
            logger.debug(f"[pose_processor] pose_landmarks: {getattr(result, 'pose_landmarks', None)}")
        if not result or not result.pose_landmarks:
            return await self._handle_no_pose_detected()
        
        pose_result = result
        results = {}
        
        if pose_result.pose_landmarks:
            results['pose_2d'] = await self.process_pose_landmarks(pose_result.pose_landmarks[0], timestamp)
        
        if pose_result.pose_world_landmarks:
            results['pose_3d'] = await self.process_world_landmarks(pose_result.pose_world_landmarks[0], timestamp)
        
        # Advanced: Comprehensive analysis combining 2D and 3D analysis
        comprehensive_analysis = await self.perform_comprehensive_pose_analysis(results, timestamp)
        results['pose_analysis'] = comprehensive_analysis
        
        # Update metrics
        self._update_pose_metrics(comprehensive_analysis)
        
        return results
    
    async def process_pose_landmarks(self, landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """Process 2D pose landmarks"""
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
            logger.error(f"Error in 2D pose analysis: {e}")
            return {'available': False}

    async def process_world_landmarks(self, world_landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """Process 3D world landmarks"""
        try:
            # Advanced S-Class: New analysis additions
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
            logger.error(f"Error in 3D pose analysis: {e}")
            return {'available': False}

    async def perform_comprehensive_pose_analysis(self, results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Advanced: Comprehensive pose analysis"""
        pose_2d = results.get('pose_2d', {'available': False})
        pose_3d = results.get('pose_3d', {'available': False})
        
        # 1. Driving posture suitability assessment
        suitability, issues = self._evaluate_driving_posture_suitability(pose_2d, pose_3d)
        
        # 2. Extract fatigue and distraction indicators
        fatigue_indicators = self._extract_fatigue_indicators_from_pose(pose_3d)
        distraction_indicators = self._extract_distraction_indicators_from_pose(pose_3d)

        # 3. Store analysis results in history
        current_pose_summary = {
            'timestamp': timestamp,
            'suitability': suitability,
            'slouch_factor': pose_3d.get('slouch_detection', {}).get('slouch_factor', 0),
            'complexity': pose_3d.get('complexity', {}).get('overall_complexity', 0),
            'spinal_health': pose_3d.get('spinal_analysis', {}).get('spine_health_score', 0.5),
            'neck_health': pose_3d.get('spinal_analysis', {}).get('neck_health_score', 0.5),
        }
        self.posture_history.append(current_pose_summary)

        # 4. Posture trend analysis
        trend_analysis = self._analyze_posture_trends()

        # 5. Advanced S-Class: Comprehensive biomechanical health score
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
        """Advanced S-Class: Spinal alignment and neck angle analysis"""
        try:
            lm = MediaPipeConstants.PoseLandmarks
            
            # Extract key 3D points
            nose = np.array([world_landmarks[lm.NOSE].x, world_landmarks[lm.NOSE].y, world_landmarks[lm.NOSE].z])
            ls, rs = world_landmarks[lm.LEFT_SHOULDER], world_landmarks[lm.RIGHT_SHOULDER]
            lh, rh = world_landmarks[lm.LEFT_HIP], world_landmarks[lm.RIGHT_HIP]
            
            neck_base = np.mean([[p.x, p.y, p.z] for p in [ls, rs]], axis=0)
            torso_base = np.mean([[p.x, p.y, p.z] for p in [lh, rh]], axis=0)
            
            # Calculate neck vector and torso vector
            neck_vector = nose - neck_base
            torso_vector = neck_base - torso_base
            
            # Forward Head Posture angle
            # Use only z-axis (anterior-posterior) and y-axis (superior-inferior) for lateral view angle
            neck_vector_2d = np.array([neck_vector[2], neck_vector[1]])
            vertical_vector_2d = np.array([0, 1])
            
            if np.linalg.norm(neck_vector_2d) > MathConstants.VECTOR_NORM_MIN:
                fhp_angle = math.degrees(math.acos(np.clip(
                    np.dot(neck_vector_2d, vertical_vector_2d) / np.linalg.norm(neck_vector_2d), 
                    -1.0, 1.0
                )))
            else:
                fhp_angle = 0.0
            
            # Spinal angle
            if np.linalg.norm(torso_vector) > MathConstants.VECTOR_NORM_MIN and np.linalg.norm(neck_vector) > MathConstants.VECTOR_NORM_MIN:
                spine_angle = math.degrees(math.acos(np.clip(
                    np.dot(torso_vector, neck_vector) / (np.linalg.norm(torso_vector) * np.linalg.norm(neck_vector)), 
                    -1.0, 1.0
                )))
            else:
                spine_angle = 180.0  # Default value
            
            # Health score calculation
            neck_health_score = max(0, 1 - (max(0, fhp_angle - 15) / 30))  # Score decreases if angle > 15 degrees
            spine_health_score = max(0, (spine_angle - 150) / 30)  # Closer to 180 is better
            
            return {
                'forward_head_posture_angle': fhp_angle, # > 20 degrees indicates forward head posture risk
                'spine_curvature_angle': spine_angle, # Closer to 180 indicates straighter posture
                'neck_health_score': neck_health_score,
                'spine_health_score': spine_health_score,
                'cervical_risk_level': 'high' if fhp_angle > 25 else 'medium' if fhp_angle > 15 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error in spinal alignment analysis: {e}")
            return {
                'forward_head_posture_angle': 0.0,
                'spine_curvature_angle': 180.0,
                'neck_health_score': 0.5,
                'spine_health_score': 0.5,
                'cervical_risk_level': 'unknown'
            }

    def _analyze_postural_sway(self, world_landmarks: Any, timestamp: float) -> Dict[str, Any]:
        """Advanced S-Class: Postural sway (instability) measurement"""
        try:
            lm = MediaPipeConstants.PoseLandmarks
            torso_center = np.mean([[p.x, p.y, p.z] for p in [
                world_landmarks[lm.LEFT_SHOULDER], world_landmarks[lm.RIGHT_SHOULDER],
                world_landmarks[lm.LEFT_HIP], world_landmarks[lm.RIGHT_HIP]
            ]], axis=0)
            
            # Add current torso center point to history
            self.torso_center_history.append({
                'timestamp': timestamp,
                'position': torso_center
            })
            
            if len(self.torso_center_history) < 30:
                return {'sway_area_cm2': 0.0, 'sway_velocity_cm_s': 0.0, 'stability_score': 1.0}

            # Calculate Sway Area as 95% confidence interval ellipse area
            recent_positions = np.array([h['position'] for h in list(self.torso_center_history)])
            
            # Sway in X-Z plane (lateral, anterior-posterior)
            x_positions = recent_positions[:, 0]
            z_positions = recent_positions[:, 2]
            
            # Calculate covariance matrix
            cov = np.cov(x_positions, z_positions)
            eigenvalues, _ = np.linalg.eig(cov)
            
            # Scale factor for actual distance (cm) conversion (assumption: 1 unit = 100cm)
            scale_factor = 100 
            sway_area = math.pi * 5.991 * np.sqrt(np.prod(np.abs(eigenvalues))) * (scale_factor**2)
            
            # Calculate Sway Velocity (average distance between consecutive points)
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
            
            # Stability score (lower sway is better)
            stability_score = max(0, 1 - (sway_area / 25.0))  # Unstable if > 25cm²

            return {
                'sway_area_cm2': sway_area,
                'sway_velocity_cm_s': sway_velocity, 
                'stability_score': stability_score,
                'sway_pattern': self._classify_sway_pattern(x_positions, z_positions)
            }
            
        except Exception as e:
            logger.error(f"Error in postural sway analysis: {e}")
            return {'sway_area_cm2': 0.0, 'sway_velocity_cm_s': 0.0, 'stability_score': 1.0}

    def _classify_sway_pattern(self, x_positions: np.ndarray, z_positions: np.ndarray) -> str:
        """Classify sway pattern"""
        try:
            x_var = np.var(x_positions)
            z_var = np.var(z_positions)
            
            if x_var > z_var * 2:
                return 'lateral_dominant'  # Lateral sway dominant
            elif z_var > x_var * 2:
                return 'anterior_posterior_dominant'  # Anterior-posterior sway dominant
            else:
                return 'circular_pattern'  # Circular pattern
                
        except Exception:
            return 'unknown'

    def _calculate_biomechanical_health_score(self, pose_3d: Dict) -> Dict[str, Any]:
        """Advanced S-Class: Comprehensive biomechanical health score"""
        if not pose_3d.get('available'):
            return {'overall_score': 0.5, 'risk_factors': [], 'recommendations': []}
        
        scores = []
        risk_factors = []
        recommendations = []
        
        # Spinal health
        spinal_score = pose_3d.get('spinal_analysis', {}).get('spine_health_score', 0.5)
        neck_score = pose_3d.get('spinal_analysis', {}).get('neck_health_score', 0.5)
        scores.extend([spinal_score, neck_score])
        
        if neck_score < 0.6:
            risk_factors.append('Forward head posture risk')
            recommendations.append('Keep neck straight and pull shoulders back')
        
        # Postural stability
        stability_score = pose_3d.get('postural_sway', {}).get('stability_score', 1.0)
        scores.append(stability_score)
        
        if stability_score < 0.7:
            risk_factors.append('Postural instability')
            recommendations.append('Core muscle strengthening exercises recommended')
        
        # Slouching severity
        slouch_factor = pose_3d.get('slouch_detection', {}).get('slouch_factor', 0)
        slouch_score = 1.0 - slouch_factor
        scores.append(slouch_score)
        
        if slouch_factor > 0.6:
            risk_factors.append('Slouching posture')
            recommendations.append('Sit back against the backrest')
        
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
        """Advanced: 3D-based slouching posture detection"""
        try:
            lm = MediaPipeConstants.PoseLandmarks
            shoulder_center = np.mean([
                [p.x, p.y, p.z] for p in [world_landmarks[lm.LEFT_SHOULDER], world_landmarks[lm.RIGHT_SHOULDER]]
            ], axis=0)
            hip_center = np.mean([
                [p.x, p.y, p.z] for p in [world_landmarks[lm.LEFT_HIP], world_landmarks[lm.RIGHT_HIP]]
            ], axis=0)
            
            torso_vector = shoulder_center - hip_center
            vertical_vector = np.array([0, 1, 0]) # Use y-axis as vertical reference
            
            # Calculate angle with vertical vector
            if np.linalg.norm(torso_vector) > MathConstants.VECTOR_NORM_MIN:
                dot_product = np.dot(torso_vector, vertical_vector) / np.linalg.norm(torso_vector)
                slouch_angle = math.degrees(math.acos(np.clip(dot_product, -1.0, 1.0)))
            else:
                slouch_angle = 0.0
            
            slouch_factor = max(0.0, min(1.0, (slouch_angle - 20) / 70.0)) # Consider slouch starting from 20 degrees
            
            return {
                'slouch_angle': slouch_angle,
                'slouch_factor': slouch_factor,
                'is_slouching': slouch_factor > 0.6,
                'severity': 'severe' if slouch_factor > 0.8 else 'moderate' if slouch_factor > 0.6 else 'mild'
            }
        except Exception as e:
            logger.error(f"Error in slouch detection: {e}")
            return {'slouch_factor': 0.0, 'is_slouching': False, 'severity': 'unknown'}

    def _evaluate_driving_posture_suitability(self, pose_2d: Dict, pose_3d: Dict) -> Tuple[float, List[str]]:
        """Advanced: Driving posture suitability assessment"""
        suitability_factors = []
        issues = []
        
        if pose_2d.get('available'):
            tilt_severity = pose_2d.get('shoulder_analysis', {}).get('tilt_severity', 0)
            if tilt_severity > 0.5: issues.append('Excessive shoulder tilt')
            suitability_factors.append(1.0 - tilt_severity)
            
        if pose_3d.get('available'):
            slouch_factor = pose_3d.get('slouch_detection', {}).get('slouch_factor', 0)
            if slouch_factor > 0.6: issues.append('Slouching posture')
            suitability_factors.append(1.0 - slouch_factor)
            
            balance_score = pose_3d.get('balance', {}).get('balance_score', 0.5)
            if balance_score < 0.4: issues.append('Poor body balance')
            suitability_factors.append(balance_score)
            
            # Advanced S-Class: Add spinal health
            spine_health = pose_3d.get('spinal_analysis', {}).get('spine_health_score', 0.5)
            if spine_health < 0.6: issues.append('Poor spinal alignment')
            suitability_factors.append(spine_health)
            
        return np.mean(suitability_factors) if suitability_factors else 0.5, issues

    def _generate_posture_recommendation(self, suitability: float, issues: List[str], biomech_health: Dict) -> str:
        """Advanced: Generate posture improvement recommendations"""
        if suitability > 0.8 and biomech_health['overall_score'] > 0.8:
            return "Excellent driving posture maintained."
        
        if biomech_health['risk_factors']:
            primary_risk = biomech_health['risk_factors'][0]
            primary_recommendation = biomech_health['recommendations'][0] if biomech_health['recommendations'] else "Correct your posture"
            return f"{primary_risk} detected: {primary_recommendation}"
        
        if issues:
            return f"Need improvement in: {', '.join(issues[:2])}"  # Show maximum 2 issues
        
        return "Posture correction needed. Please sit in proper posture."
        
    def _analyze_posture_trends(self) -> Dict[str, Any]:
        """Advanced: Posture trend analysis"""
        if len(self.posture_history) < 60: # Need minimum 2 seconds of data
            return {'trend_available': False, 'stability_trend': 'stable', 'deterioration_rate': 0.0}

        recent_history = list(self.posture_history)
        timestamps = np.array([h['timestamp'] for h in recent_history])
        suitability_scores = np.array([h['suitability'] for h in recent_history])
        
        # Calculate trend slope using linear regression
        try:
            coeffs = np.polyfit(timestamps, suitability_scores, 1)
            slope = coeffs[0]
        except np.linalg.LinAlgError:
            slope = 0.0

        if slope < -0.01: trend = 'deteriorating' # > 1% decline per hour
        elif slope > 0.01: trend = 'improving'
        else: trend = 'stable'
        
        # Analyze health score trend as well
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
        """Predict future posture changes"""
        if suitability_slope < -0.02 or health_slope < -0.01:
            return "Continuous posture deterioration expected - immediate rest recommended"
        elif suitability_slope > 0.01 and health_slope > 0.005:
            return "Posture improving - maintain current state"
        else:
            return "Stable posture maintained"

    def _extract_key_pose_points(self, landmarks: Any) -> Dict[str, Any]:
        """Extract key pose points"""
        constants = MediaPipeConstants.PoseLandmarks
        return {
            'left_shoulder': landmarks[constants.LEFT_SHOULDER],
            'right_shoulder': landmarks[constants.RIGHT_SHOULDER],
            'left_hip': landmarks[constants.LEFT_HIP],
            'right_hip': landmarks[constants.RIGHT_HIP],
        }

    def _analyze_shoulder_orientation(self, key_points: Dict) -> Dict[str, Any]:
        """Analyze shoulder orientation"""
        left_shoulder = key_points['left_shoulder']
        right_shoulder = key_points['right_shoulder']
        shoulder_angle = math.degrees(math.atan2(left_shoulder.y - right_shoulder.y, left_shoulder.x - right_shoulder.x))
        return { 'angle_degrees': shoulder_angle, 'tilt_severity': min(1.0, abs(shoulder_angle) / 45.0) }

    def _analyze_body_symmetry(self, key_points: Dict) -> Dict[str, Any]:
        """Analyze body left-right symmetry"""
        shoulder_height_diff = abs(key_points['left_shoulder'].y - key_points['right_shoulder'].y)
        hip_height_diff = abs(key_points['left_hip'].y - key_points['right_hip'].y)
        return { 'symmetry_score': 1.0 - min(1.0, (shoulder_height_diff + hip_height_diff) * 2) }

    def _analyze_3d_body_balance(self, world_landmarks: Any) -> Dict[str, Any]:
        """Analyze body balance in 3D space"""
        lm = MediaPipeConstants.PoseLandmarks
        left_center = np.mean([[p.x, p.y, p.z] for p in [world_landmarks[lm.LEFT_SHOULDER], world_landmarks[lm.LEFT_HIP]]], axis=0)
        right_center = np.mean([[p.x, p.y, p.z] for p in [world_landmarks[lm.RIGHT_SHOULDER], world_landmarks[lm.RIGHT_HIP]]], axis=0)
        balance_distance = np.linalg.norm(left_center - right_center)
        return {'balance_score': max(0.0, 1.0 - balance_distance * 5) }

    def _calculate_pose_complexity(self, world_landmarks: Any) -> Dict[str, Any]:
        """Calculate pose complexity"""
        torso_points = [world_landmarks[i] for i in [11, 12, 23, 24]]  # Shoulder and hip points
        positions = np.array([[lm.x, lm.y, lm.z] for lm in torso_points])
        return {'overall_complexity': min(1.0, np.var(positions) * 10) }

    def _extract_fatigue_indicators_from_pose(self, pose_3d: Dict) -> Dict[str, float]:
        """Extract fatigue indicators from posture"""
        if not pose_3d.get('available'): return {'slouching': 0.0, 'instability': 0.0}
        
        slouching = pose_3d.get('slouch_detection', {}).get('slouch_factor', 0.0)
        instability = 1.0 - pose_3d.get('postural_sway', {}).get('stability_score', 1.0)
        
        return {'slouching': slouching, 'instability': instability}

    def _extract_distraction_indicators_from_pose(self, pose_3d: Dict) -> Dict[str, float]:
        """Extract distraction indicators from posture"""
        if not pose_3d.get('available'): return {'unusual_positioning': 0.0}
        return {'unusual_positioning': pose_3d.get('complexity', {}).get('overall_complexity', 0.0)}

    def _update_pose_metrics(self, pose_analysis: Dict[str, Any]):
        """Update posture-related metrics"""
        metrics_data = {
            'pose_complexity_score': pose_analysis.get('distraction_indicators', {}).get('unusual_positioning', 0.0),
            'slouch_factor': pose_analysis.get('fatigue_indicators', {}).get('slouching', 0.0),
            'posture_suitability': pose_analysis.get('driving_suitability', 0.5),
            'spinal_health_score': pose_analysis.get('biomechanical_health', {}).get('spinal_health', 0.5),
            'postural_stability': pose_analysis.get('biomechanical_health', {}).get('stability', 1.0)
        }
        # Metrics update handled by the base interface
        try:
            # Use dynamic method resolution to avoid linter errors
            update_method = getattr(self.metrics_updater, 'update_pose_metrics', None) or getattr(self.metrics_updater, 'update_metrics', None)
            if update_method:
                update_method(metrics_data)
        except Exception as e:
            logger.debug(f"Could not update pose metrics: {e}")

    async def _handle_no_pose_detected(self) -> Dict[str, Any]:
        """Handle no pose detected scenario"""
        logger.warning("No pose detected - backup mode or sensor recalibration needed")
        return { 
            'pose_detected': False, 
            'pose_analysis': {
                'driving_suitability': 0.0,
                'biomechanical_health': {'overall_score': 0.0, 'risk_factors': ['Pose detection failed'], 'recommendations': ['Adjust camera position']}
            }
        }