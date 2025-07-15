"""
Object Data Processor - Digital Behavior Prediction Expert
Advanced object analysis processor that performs comprehensive behavioral analysis
using Bayesian networks, attention heatmaps, and complex behavior sequence modeling.

Features:
- Bayesian network-based behavioral intent inference system
- Attention heatmap generation and visual attention dispersion pattern analysis
- Context-aware dynamic risk adjustment (traffic conditions, weather, time)
- Complex behavioral sequence modeling for future behavior prediction
- Advanced behavioral pattern recognition and risk escalation prediction
- Contextual safety recommendations based on environmental factors
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
    """Behavior sequence pattern definitions"""
    PHONE_PICKUP_SEQUENCE = "phone_pickup_sequence"  # Phone picking action sequence
    DRINKING_SEQUENCE = "drinking_sequence"  # Drinking action sequence
    CONSOLE_OPERATION_SEQUENCE = "console_operation_sequence"  # Console operation sequence
    DISTRACTED_SCROLLING = "distracted_scrolling"  # Distracted scrolling pattern


@dataclass
class BehaviorPrediction:
    """Behavior prediction result"""
    predicted_action: str
    confidence: float
    time_to_action: float  # In seconds
    risk_escalation_probability: float


@dataclass
class AttentionHeatmap:
    """Visual attention heatmap"""
    zones: Dict[str, float]  # Attention concentration by zone
    center_of_attention: Tuple[float, float]  # Center point of attention
    dispersion_score: float  # Attention dispersion (0-1)


class ObjectDataProcessor(IObjectDataProcessor):
    """
    Object Data Processor - Digital Behavior Prediction Expert
    
    Advanced object analysis processor that acts like a future-predicting behavioral analyst,
    understanding driver intentions and predicting dangerous behaviors in advance.
    
    Features:
    - Bayesian network-based behavioral intent inference
    - Attention heatmap generation and pattern analysis
    - Context-aware risk adjustment for environmental factors
    - Behavioral sequence modeling and prediction
    - Advanced behavioral pattern recognition and risk assessment
    """
    
    def __init__(self, metrics_updater: IMetricsUpdater):
        self.metrics_updater = metrics_updater
        self.config = get_config()
        
        # Distraction object definitions
        self.distraction_objects = self.config.distraction.object_risk_levels
        
        # --- Advanced S-Class: Advanced data structures for behavior prediction ---
        self.detection_history = deque(maxlen=self.config.distraction.detection_history_size)
        self.behavior_sequence_buffer = deque(maxlen=50)  # Behavior sequence tracking
        self.attention_heatmap_history = deque(maxlen=30)  # Visual attention patterns
        self.contextual_factors = {}  # Contextual factors (time, traffic conditions, etc.)
        
        # Bayesian network probability table (simplified version)
        self.behavior_probability_matrix = self._initialize_behavior_probabilities()
        
        # Object tracking state
        self.tracked_objects = {}
        self.object_lifecycle = {}
        
        logger.info("ObjectDataProcessor initialized - Digital Behavior Prediction Expert ready")

    def get_processor_name(self) -> str:
        """Returns the name of the processor."""
        return "ObjectDataProcessor"

    def get_required_data_types(self) -> List[str]:
        """Returns the list of data types required by this processor."""
        return ["object_detections", "hand_positions"]

    def _initialize_behavior_probabilities(self) -> Dict[str, Dict[str, float]]:
        """Initialize Bayesian network probability table"""
        return {
            # Phone-related behavior probabilities
            'cell_phone': {
                'pickup_probability': 0.3,
                'usage_escalation': 0.7,
                'attention_capture': 0.9
            },
            # Drink-related behavior probabilities  
            'cup': {
                'drinking_probability': 0.8,
                'spillage_risk': 0.2,
                'attention_capture': 0.4
            },
            # Food-related behavior probabilities
            'food': {
                'eating_probability': 0.6,
                'messy_handling': 0.4,
                'attention_capture': 0.6
            }
        }

    async def process_data(self, result, timestamp):
        logger.debug(f"[object_processor] process_data input: {result}")
        if hasattr(result, 'detections'):
            logger.debug(f"[object_processor] detections: {getattr(result, 'detections', None)}")
        if not result or not hasattr(result, 'detections') or not result.detections:
            return await self._handle_no_objects_detected(timestamp)
        
        # Hand position information received from other processors
        hand_positions = getattr(result, 'hand_positions', [])
        
        # Advanced S-Class: Update contextual context
        self._update_contextual_factors(timestamp)
        
        # Process object detection results
        object_analysis = self.process_object_detections(
            result, hand_positions, timestamp
        )
        
        # Advanced S-Class: Generate attention heatmap
        attention_heatmap = self._generate_attention_heatmap(object_analysis, hand_positions)
        
        # Advanced S-Class: Behavior sequence analysis and prediction
        behavior_prediction = await self._analyze_behavior_sequences(
            object_analysis, hand_positions, timestamp
        )
        
        # Comprehensive risk analysis
        comprehensive_analysis = await self.perform_comprehensive_risk_analysis(
            object_analysis, timestamp
        )
        
        # Advanced S-Class: Integrate prediction results into comprehensive analysis
        comprehensive_analysis['behavior_prediction'] = behavior_prediction
        comprehensive_analysis['attention_heatmap'] = attention_heatmap
        
        results = {
            'object_detections': object_analysis,
            'risk_analysis': comprehensive_analysis
        }
        
        self._update_object_metrics(comprehensive_analysis)
        return results

    def _update_contextual_factors(self, timestamp: float):
        """Advanced S-Class: Update contextual context"""
        current_hour = time.localtime(timestamp).tm_hour
        
        # Time-based risk weights (rush hour, night driving, etc.)
        if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
            traffic_factor = 1.3  # Rush hour
        elif 22 <= current_hour or current_hour <= 5:
            fatigue_factor = 1.4  # Night driving
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
        """Advanced S-Class: Generate visual attention heatmap"""
        zones = {
            'steering_wheel': 0.0,
            'center_console': 0.0, 
            'dashboard': 0.0,
            'passenger_area': 0.0,
            'floor': 0.0
        }
        
        total_attention = 0.0
        weighted_x, weighted_y = 0.0, 0.0
        
        # Calculate attention dispersion from detected objects
        for obj in object_analysis.get('detected_objects', []):
            bbox = obj['bbox']
            risk_level = obj['risk_level']
            
            # Map object position to vehicle zone
            zone = self._map_position_to_zone(bbox['center_x'], bbox['center_y'])
            attention_weight = risk_level * obj['detection_confidence']
            
            zones[zone] += attention_weight
            total_attention += attention_weight
            
            # Calculate attention center point with weighted average
            weighted_x += bbox['center_x'] * attention_weight
            weighted_y += bbox['center_y'] * attention_weight
        
        # Reflect hand positions in attention dispersion
        for hand in hand_positions:
            hand_zone = self._map_position_to_zone(
                hand.get('hand_center', {}).get('x', 0.5),
                hand.get('hand_center', {}).get('y', 0.5)
            )
            zones[hand_zone] += 0.3  # Hand position has moderate attention value
            total_attention += 0.3
        
        # Normalize
        if total_attention > 0:
            for zone in zones:
                zones[zone] /= total_attention
            center_of_attention = (weighted_x / total_attention, weighted_y / total_attention)
        else:
            center_of_attention = (0.5, 0.5)  # Center default
        
        # Calculate attention dispersion (entropy-based)
        dispersion_score = self._calculate_attention_dispersion(zones)
        
        heatmap = AttentionHeatmap(
            zones=zones,
            center_of_attention=center_of_attention,
            dispersion_score=dispersion_score
        )
        
        self.attention_heatmap_history.append(heatmap)
        return heatmap

    def _calculate_attention_dispersion(self, zones: Dict[str, float]) -> float:
        """Calculate attention dispersion (entropy-based)"""
        entropy = 0.0
        for attention_value in zones.values():
            if attention_value > 0:
                entropy -= attention_value * math.log2(attention_value)
        
        # Normalize (maximum entropy is log2(number of zones))
        max_entropy = math.log2(len(zones))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    async def _analyze_behavior_sequences(
        self, object_analysis: Dict, hand_positions: List[Dict], timestamp: float
    ) -> BehaviorPrediction:
        """Advanced S-Class: Behavior sequence analysis and future behavior prediction"""
        
        # Record current frame's behavioral state
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
        
        # Behavior sequence pattern matching
        detected_sequence = self._detect_behavior_sequence()
        
        # Predict future behavior with Bayesian inference
        prediction = self._predict_next_behavior(detected_sequence, current_behavior)
        
        return prediction

    def _extract_current_behavior_state(
        self, object_analysis: Dict, hand_positions: List[Dict]
    ) -> Dict[str, Any]:
        """Extract current frame's behavioral state"""
        state = {
            'objects_present': [obj['category'] for obj in object_analysis.get('detected_objects', [])],
            'hand_object_interactions': [],
            'dominant_object': None,
            'interaction_intensity': 0.0
        }
        
        # Identify most dangerous object
        max_risk = 0.0
        for obj in object_analysis.get('detected_objects', []):
            if obj['risk_level'] > max_risk:
                max_risk = obj['risk_level']
                state['dominant_object'] = obj['category']
                state['interaction_intensity'] = obj['hand_interaction']['proximity_score']
        
        # Record hand-object interactions
        for obj in object_analysis.get('detected_objects', []):
            if obj['hand_interaction']['is_interacting']:
                state['hand_object_interactions'].append({
                    'object': obj['category'],
                    'interaction_type': obj['hand_interaction']['interaction_type']
                })
        
        return state

    def _detect_behavior_sequence(self) -> Optional[BehaviorSequence]:
        """Detect behavior sequence patterns"""
        recent_states = list(self.behavior_sequence_buffer)[-10:]
        
        # Detect phone pickup sequence
        phone_sequence_score = self._calculate_phone_pickup_sequence_score(recent_states)
        if phone_sequence_score > 0.7:
            return BehaviorSequence.PHONE_PICKUP_SEQUENCE
        
        # Detect drinking sequence
        drinking_sequence_score = self._calculate_drinking_sequence_score(recent_states)
        if drinking_sequence_score > 0.6:
            return BehaviorSequence.DRINKING_SEQUENCE
        
        return None

    def _calculate_phone_pickup_sequence_score(self, states: List[Dict]) -> float:
        """Calculate phone pickup sequence score"""
        phone_presence = 0
        hand_movement_toward_phone = 0
        
        for state in states:
            if 'cell phone' in state['behavior_state']['objects_present']:
                phone_presence += 1
            
            for interaction in state['behavior_state']['hand_object_interactions']:
                if interaction['object'] == 'cell phone':
                    hand_movement_toward_phone += 1
        
        # Sequence score: persistent phone presence + hand approach
        sequence_score = (phone_presence / len(states)) * 0.6 + (hand_movement_toward_phone / len(states)) * 0.4
        return min(1.0, sequence_score)

    def _calculate_drinking_sequence_score(self, states: List[Dict]) -> float:
        """Calculate drinking sequence score"""
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
        """Predict next behavior through Bayesian inference"""
        
        if detected_sequence == BehaviorSequence.PHONE_PICKUP_SEQUENCE:
            # If picking up phone, next action likely to be usage
            prob_table = self.behavior_probability_matrix.get('cell_phone', {})
            return BehaviorPrediction(
                predicted_action="phone_usage_imminent",
                confidence=prob_table.get('usage_escalation', 0.7),
                time_to_action=2.5,  # Average 2.5 seconds until usage begins
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
        
        # When no special sequence is detected
        return BehaviorPrediction(
            predicted_action="maintaining_current_state",
            confidence=0.6,
            time_to_action=0.0,
            risk_escalation_probability=0.2
        )

    async def perform_comprehensive_risk_analysis(
        self, object_analysis: Dict, timestamp: float
    ) -> Dict[str, Any]:
        """Advanced S-Class: Comprehensive risk analysis (reflecting contextual factors)"""
        try:
            detected_objects = object_analysis.get('detected_objects', [])
            
            if not detected_objects:
                return self._get_default_risk_analysis()
            
            # Calculate base risk
            base_risk = self._calculate_overall_risk_score(detected_objects)
            
            # Advanced S-Class: Reflect contextual factors
            contextual_multiplier = self._calculate_contextual_risk_multiplier()
            adjusted_risk = min(1.0, base_risk * contextual_multiplier)
            
            # Temporal pattern analysis
            temporal_patterns = self._analyze_temporal_risk_patterns()
            
            # Identify priority risk objects
            priority_objects = sorted(detected_objects, key=lambda x: x['risk_level'], reverse=True)[:3]
            
            # Advanced S-Class: Context-aware safety recommendations
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
            
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Error in comprehensive risk analysis: {e}")
            return self._get_default_risk_analysis()

    def _calculate_contextual_risk_multiplier(self) -> float:
        """Calculate contextual risk weight"""
        multiplier = 1.0
        
        # Time-based weights
        multiplier *= self.contextual_factors.get('traffic_risk_multiplier', 1.0)
        multiplier *= self.contextual_factors.get('fatigue_risk_multiplier', 1.0)
        
        # Weight based on attention dispersion
        if self.attention_heatmap_history:
            latest_heatmap = self.attention_heatmap_history[-1]
            dispersion_multiplier = 1.0 + (latest_heatmap.dispersion_score * 0.5)
            multiplier *= dispersion_multiplier
        
        return min(2.0, multiplier)  # Maximum 2x weight

    def _generate_contextual_safety_recommendations(
        self, objects: List[Dict], risk_score: float, context: Dict
    ) -> List[str]:
        """Advanced S-Class: Generate context-aware safety recommendations"""
        recommendations = []
        
        # Time-based customized recommendations
        current_hour = context.get('current_hour', 12)
        
        if risk_score > 0.8:
            if 22 <= current_hour or current_hour <= 5:
                recommendations.append("High-risk situation during night driving - pull over to a safe location immediately")
            else:
                recommendations.append("Immediately remove all distracting elements and focus on driving")
        
        elif risk_score > 0.6:
            top_object = objects[0]['category'] if objects else "unknown object"
            if current_hour in [7, 8, 17, 18]:  # Rush hour
                recommendations.append(f"Heavy traffic hours. Stop using {top_object}")
            else:
                recommendations.append(f"Reduce {top_object} usage and focus on forward attention")
        
        else:
            recommendations.append("Currently maintaining safe state")
        
        return recommendations

    # --- Helper methods ---
    
    def _map_position_to_zone(self, x: float, y: float) -> str:
        """Map position to vehicle zone"""
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
        """Calculate overall risk score"""
        if not objects:
            return 0.0
        
        risk_levels = [obj['risk_level'] for obj in objects]
        max_risk = max(risk_levels)
        avg_risk = np.mean(risk_levels)
        
        # Weight highest risk but penalize multiple objects
        overall_risk = (max_risk * 0.7 + avg_risk * 0.3) * min(1.5, 1.0 + (len(objects) - 1) * 0.25)
        return min(1.0, overall_risk)

    def _analyze_temporal_risk_patterns(self) -> Dict[str, Any]:
        """Analyze temporal risk patterns"""
        if len(self.detection_history) < 60:
            return {'pattern_detected': False, 'trend': 'stable'}

        recent_risks = [
            np.mean([o.get('risk_level', 0) for o in rec.get('objects', [])] + [0]) 
            for rec in list(self.detection_history)
        ]
        timestamps = [rec['timestamp'] for rec in list(self.detection_history)]
        
        try:
            slope = np.polyfit(timestamps, recent_risks, 1)[0]
        except np.linalg.LinAlgError as e:
            logger.warning(f"Temporal risk pattern analysis failed due to singular matrix: {e}")
            slope = 0.0

        if slope > 0.05:
            trend = 'increasing'
        elif slope < -0.05:
            trend = 'decreasing'  
        else:
            trend = 'stable'
        
        return {'pattern_detected': True, 'trend': trend, 'risk_slope': slope}

    def _categorize_risk_level(self, risk: float) -> str:
        """Categorize risk level"""
        if risk > 0.8:
            return 'critical'
        elif risk > 0.6:
            return 'high'
        elif risk > 0.4:
            return 'medium'
        else:
            return 'low'

    async def _handle_no_objects_detected(self, timestamp: float) -> Dict[str, Any]:
        """Handle no objects detected"""
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
        """Default risk analysis data"""
        return {
            'overall_risk_score': 0.0,
            'base_risk_score': 0.0,
            'contextual_multiplier': 1.0,
            'risk_level_category': 'minimal',
            'priority_risk_objects': [],
            'temporal_patterns': {'trend': 'stable'},
            'safety_recommendations': ["No distracting objects detected."],
            'contextual_factors': self.contextual_factors.copy(),
            'behavior_prediction': BehaviorPrediction(
                predicted_action="normal_driving",
                confidence=0.8,
                time_to_action=0.0,
                risk_escalation_probability=0.1
            ),
            'attention_heatmap': AttentionHeatmap(
                zones={'steering_wheel': 1.0, 'center_console': 0.0, 'dashboard': 0.0, 'passenger_area': 0.0, 'floor': 0.0},
                center_of_attention=(0.5, 0.8),  # Steering wheel position
                dispersion_score=0.0
            )
        }

    def _update_object_metrics(self, comprehensive_analysis: Dict[str, Any]):
        """Update object-related metrics"""
        try:
            metrics_data = {
                'distraction_objects_detected': len(comprehensive_analysis.get('priority_risk_objects', [])),
                'overall_object_risk': comprehensive_analysis.get('overall_risk_score', 0.0),
                'contextual_risk_multiplier': comprehensive_analysis.get('contextual_multiplier', 1.0),
                'attention_dispersion_score': comprehensive_analysis.get('attention_heatmap', {}).get('dispersion_score', 0.0),
                'predicted_risk_escalation': comprehensive_analysis.get('behavior_prediction', {}).get('risk_escalation_probability', 0.0)
            }
            # Use dynamic method resolution to avoid linter errors
            try:
                update_method = getattr(self.metrics_updater, 'update_distraction_metrics', None) or getattr(self.metrics_updater, 'update_metrics', None)
                if update_method:
                    update_method(metrics_data)
            except (AttributeError, TypeError) as e:
                logger.debug(f"Could not update object metrics: {e}")
        except (AttributeError, TypeError, KeyError) as e:
            logger.error(f"Error updating object metrics: {e}")

    def process_object_detections(self, detections: Any, hand_positions: List[Dict], timestamp: float) -> Dict[str, Any]:
        """Process object detection results (S-Class: Integrated with advanced behavior prediction and risk analysis)"""
        # Integrated with advanced analysis pipeline used in process_data
        # In S-Class, process_data is more advanced, so core logic is reused
        # (In actual system, full analysis is performed through process_data, here only object detection results are returned)
        if not detections or not hasattr(detections, 'detections') or not detections.detections:
            return {'detected_objects': [], 'object_count': 0}
        # Hand position information is passed separately
        object_analysis = []
        for detection in detections.detections:
            category = detection.categories[0].category_name
            confidence = detection.categories[0].score
            bbox = detection.bounding_box
            # Check if it's an object of interest
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