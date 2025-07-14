"""
Dynamic Analysis Engine - Enhanced System
통합 시스템과 호환되는 동적 분석 엔진
기존 코드와 개선된 코드를 통합하여 호환성과 성능을 모두 확보
"""

import time
import logging
from typing import Dict, Any, Optional, List
try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


class DynamicAnalysisEngine:
    """통합 동적 분석 엔진 - 기존 코드와 개선된 코드의 통합"""

    def __init__(self):
        """동적 분석 엔진 초기화"""
        self.start_time = time.time()
        self.analysis_history = []
        self.max_history = 100
        
        # Analysis state
        self.current_complexity = 0.0
        self.adaptive_thresholds = {}
        self.performance_mode = "balanced"  # low, balanced, high
        
        # Performance tracking
        self.analysis_count = 0
        self.avg_analysis_time = 0.0
        
        logger.info("Dynamic Analysis Engine initialized")

    def analyze_frame_complexity(self, frame_data: Dict[str, Any]) -> float:
        """프레임 복잡도 분석"""
        try:
            complexity = 0.0
            
            # Face complexity
            if 'face' in frame_data and frame_data['face']:
                face_data = frame_data['face']
                if 'landmarks' in face_data and face_data['landmarks']:
                    complexity += 0.3
                if 'blendshapes' in face_data and face_data['blendshapes']:
                    complexity += 0.2
            
            # Pose complexity
            if 'pose' in frame_data and frame_data['pose']:
                pose_data = frame_data['pose']
                if 'landmarks' in pose_data and pose_data['landmarks']:
                    complexity += 0.2
                if 'visibility' in pose_data:
                    complexity += 0.1
            
            # Hand complexity
            if 'hand' in frame_data and frame_data['hand']:
                hand_data = frame_data['hand']
                if 'landmarks' in hand_data and hand_data['landmarks']:
                    complexity += 0.2
            
            # Object complexity
            if 'object' in frame_data and frame_data['object']:
                object_data = frame_data['object']
                if 'detections' in object_data and object_data['detections']:
                    complexity += 0.1 * len(object_data['detections'])
            
            # Normalize complexity (0.0 to 1.0)
            self.current_complexity = min(1.0, complexity)
            
            return self.current_complexity
            
        except Exception as e:
            logger.error(f"Error analyzing frame complexity: {e}")
            return 0.5  # Default moderate complexity

    def get_adaptive_thresholds(self, base_thresholds: Dict[str, float]) -> Dict[str, float]:
        """적응형 임계값 계산"""
        try:
            adaptive_thresholds = base_thresholds.copy()
            
            # Adjust thresholds based on complexity
            complexity_factor = 1.0 + (self.current_complexity - 0.5) * 0.2
            
            # Adjust based on performance mode
            mode_factors = {
                "low": 1.2,      # More lenient thresholds
                "balanced": 1.0, # Standard thresholds
                "high": 0.8      # Stricter thresholds
            }
            
            mode_factor = mode_factors.get(self.performance_mode, 1.0)
            
            # Apply adjustments
            for key, value in adaptive_thresholds.items():
                adaptive_thresholds[key] = value * complexity_factor * mode_factor
                
            self.adaptive_thresholds = adaptive_thresholds
            return adaptive_thresholds
            
        except Exception as e:
            logger.error(f"Error calculating adaptive thresholds: {e}")
            return base_thresholds

    def update_performance_mode(self, fps: float, target_fps: float = 30.0):
        """성능 모드 업데이트"""
        try:
            if fps < target_fps * 0.7:
                self.performance_mode = "low"
            elif fps < target_fps * 0.9:
                self.performance_mode = "balanced"
            else:
                self.performance_mode = "high"
                
            logger.debug(f"Performance mode updated to: {self.performance_mode}")
            
        except Exception as e:
            logger.error(f"Error updating performance mode: {e}")

    def add_analysis_result(self, result: Dict[str, Any], processing_time: float):
        """분석 결과 추가"""
        try:
            self.analysis_count += 1
            
            # Update average analysis time
            if self.avg_analysis_time == 0.0:
                self.avg_analysis_time = processing_time
            else:
                self.avg_analysis_time = (self.avg_analysis_time + processing_time) / 2
            
            # Store summarized result
            summarized_result = self._summarize_result(result)
            summarized_result['processing_time'] = processing_time
            summarized_result['timestamp'] = time.time()
            summarized_result['complexity'] = self.current_complexity
            
            self.analysis_history.append(summarized_result)
            
            # Keep history manageable
            if len(self.analysis_history) > self.max_history:
                self.analysis_history.pop(0)
                
        except Exception as e:
            logger.error(f"Error adding analysis result: {e}")

    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """결과 요약"""
        try:
            summary = {}
            
            # Summarize each component
            for component in ['drowsiness', 'emotion', 'gaze', 'distraction', 'prediction']:
                if component in result:
                    component_data = result[component]
                    if isinstance(component_data, dict) and 'score' in component_data:
                        summary[f'{component}_score'] = component_data['score']
                    elif isinstance(component_data, (int, float)):
                        summary[f'{component}_score'] = component_data
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing result: {e}")
            return {}

    def get_analysis_stats(self) -> Dict[str, Any]:
        """분석 통계 반환"""
        try:
            if not self.analysis_history:
                return {}
            
            # Calculate statistics for each component
            stats = {
                'analysis_count': self.analysis_count,
                'avg_analysis_time': self.avg_analysis_time,
                'current_complexity': self.current_complexity,
                'performance_mode': self.performance_mode,
                'uptime_seconds': time.time() - self.start_time
            }
            
            # Component statistics
            for component in ['drowsiness', 'emotion', 'gaze', 'distraction', 'prediction']:
                scores = [h.get(f'{component}_score', 0) for h in self.analysis_history 
                         if f'{component}_score' in h]
                if scores:
                    stats[f'{component}_avg'] = sum(scores) / len(scores)
                    stats[f'{component}_max'] = max(scores)
                    stats[f'{component}_min'] = min(scores)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting analysis stats: {e}")
            return {}

    def get_recommendations(self) -> List[str]:
        """최적화 권장사항 반환"""
        try:
            recommendations = []
            
            if self.current_complexity > 0.8:
                recommendations.append("Frame complexity is high. Consider reducing analysis features.")
            
            if self.performance_mode == "low":
                recommendations.append("Performance is low. Consider optimizing system resources.")
            
            if self.avg_analysis_time > 100:  # 100ms
                recommendations.append("Analysis time is high. Consider performance optimizations.")
            
            if len(self.analysis_history) > 50:
                recent_complexity = [h.get('complexity', 0) for h in self.analysis_history[-10:]]
                if recent_complexity and sum(recent_complexity) / len(recent_complexity) > 0.9:
                    recommendations.append("Consistently high complexity detected. Review analysis pipeline.")
            
            if not recommendations:
                recommendations.append("Analysis performance is optimal.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return ["Unable to generate recommendations."]

    def reset(self):
        """분석 엔진 리셋"""
        try:
            self.start_time = time.time()
            self.analysis_history.clear()
            self.analysis_count = 0
            self.avg_analysis_time = 0.0
            self.current_complexity = 0.0
            self.adaptive_thresholds.clear()
            
            logger.info("Dynamic Analysis Engine reset")
            
        except Exception as e:
            logger.error(f"Error resetting analysis engine: {e}")

    def get_performance_info(self) -> Dict[str, Any]:
        """성능 정보 반환"""
        return {
            'analysis_count': self.analysis_count,
            'avg_analysis_time_ms': self.avg_analysis_time * 1000,
            'current_complexity': self.current_complexity,
            'performance_mode': self.performance_mode,
            'history_size': len(self.analysis_history),
            'uptime_seconds': time.time() - self.start_time
        }

    # Legacy compatibility methods
    def analyze_complexity(self, data):
        """Legacy compatibility method"""
        return self.analyze_frame_complexity(data)

    def get_thresholds(self, base_thresholds):
        """Legacy compatibility method"""
        return self.get_adaptive_thresholds(base_thresholds)
