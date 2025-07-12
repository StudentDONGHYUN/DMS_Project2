"""
Dynamic Analysis Engine v2
통합 시스템과 호환되는 동적 분석 엔진
기존 코드와 개선된 코드를 통합하여 호환성과 성능을 모두 확보
"""

import time
import logging
from typing import Dict, Any, Optional, List
import numpy as np

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
                    complexity += 0.3
                if 'world_landmarks' in pose_data and pose_data['world_landmarks']:
                    complexity += 0.2
            
            # Hand complexity
            if 'hand' in frame_data and frame_data['hand']:
                hand_data = frame_data['hand']
                if 'landmarks' in hand_data and hand_data['landmarks']:
                    complexity += 0.2
                if 'handedness' in hand_data and hand_data['handedness']:
                    complexity += 0.1
            
            # Object complexity
            if 'object' in frame_data and frame_data['object']:
                object_data = frame_data['object']
                if 'detections' in object_data and object_data['detections']:
                    complexity += 0.1 * len(object_data['detections'])
            
            self.current_complexity = complexity
            return complexity
            
        except Exception as e:
            logger.error(f"Error analyzing frame complexity: {e}")
            return 0.0

    def get_adaptive_thresholds(self, base_thresholds: Dict[str, float]) -> Dict[str, float]:
        """적응형 임계값 반환"""
        try:
            adaptive = base_thresholds.copy()
            
            # Adjust thresholds based on complexity
            if self.current_complexity > 0.8:
                # High complexity - relax thresholds
                for key in adaptive:
                    adaptive[key] *= 1.2
            elif self.current_complexity < 0.3:
                # Low complexity - tighten thresholds
                for key in adaptive:
                    adaptive[key] *= 0.8
            
            # Adjust based on performance mode
            if self.performance_mode == "low":
                for key in adaptive:
                    adaptive[key] *= 1.3  # More relaxed for low performance
            elif self.performance_mode == "high":
                for key in adaptive:
                    adaptive[key] *= 0.7  # Stricter for high performance
            
            return adaptive
            
        except Exception as e:
            logger.error(f"Error getting adaptive thresholds: {e}")
            return base_thresholds

    def update_performance_mode(self, fps: float, target_fps: float = 30.0):
        """성능 모드 업데이트"""
        try:
            if fps < target_fps * 0.7:
                self.performance_mode = "low"
            elif fps > target_fps * 0.9:
                self.performance_mode = "high"
            else:
                self.performance_mode = "balanced"
            
            logger.debug(f"Performance mode updated: {self.performance_mode} (FPS: {fps:.1f})")
            
        except Exception as e:
            logger.error(f"Error updating performance mode: {e}")

    def add_analysis_result(self, result: Dict[str, Any], processing_time: float):
        """분석 결과 추가"""
        try:
            self.analysis_count += 1
            
            # Update average processing time
            if self.analysis_count == 1:
                self.avg_analysis_time = processing_time
            else:
                self.avg_analysis_time = (self.avg_analysis_time * (self.analysis_count - 1) + processing_time) / self.analysis_count
            
            # Add to history
            history_entry = {
                'timestamp': time.time(),
                'complexity': self.current_complexity,
                'processing_time': processing_time,
                'performance_mode': self.performance_mode,
                'result_summary': self._summarize_result(result)
            }
            
            self.analysis_history.append(history_entry)
            
            # Keep history size manageable
            if len(self.analysis_history) > self.max_history:
                self.analysis_history = self.analysis_history[-self.max_history:]
                
        except Exception as e:
            logger.error(f"Error adding analysis result: {e}")

    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """결과 요약"""
        try:
            summary = {}
            
            # Extract key metrics
            if 'fatigue_risk_score' in result:
                summary['fatigue_risk'] = result['fatigue_risk_score']
            
            if 'distraction_risk_score' in result:
                summary['distraction_risk'] = result['distraction_risk_score']
            
            if 'confidence_score' in result:
                summary['confidence'] = result['confidence_score']
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing result: {e}")
            return {}

    def get_analysis_stats(self) -> Dict[str, Any]:
        """분석 통계 반환"""
        try:
            if not self.analysis_history:
                return {}
            
            # Calculate statistics
            processing_times = [entry['processing_time'] for entry in self.analysis_history]
            complexities = [entry['complexity'] for entry in self.analysis_history]
            
            stats = {
                'total_analyses': self.analysis_count,
                'avg_processing_time': self.avg_analysis_time,
                'min_processing_time': min(processing_times) if processing_times else 0.0,
                'max_processing_time': max(processing_times) if processing_times else 0.0,
                'avg_complexity': sum(complexities) / len(complexities) if complexities else 0.0,
                'current_complexity': self.current_complexity,
                'performance_mode': self.performance_mode,
                'history_size': len(self.analysis_history)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting analysis stats: {e}")
            return {}

    def get_recommendations(self) -> List[str]:
        """분석 기반 권장사항 반환"""
        try:
            recommendations = []
            
            if self.avg_analysis_time > 100:  # 100ms
                recommendations.append("Consider reducing analysis complexity for better performance")
            
            if self.current_complexity > 0.9:
                recommendations.append("High frame complexity detected - consider simplifying processing")
            
            if self.performance_mode == "low":
                recommendations.append("System is in low performance mode - consider reducing workload")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []

    def reset(self):
        """분석 엔진 초기화"""
        try:
            self.analysis_history.clear()
            self.analysis_count = 0
            self.avg_analysis_time = 0.0
            self.current_complexity = 0.0
            self.performance_mode = "balanced"
            
            logger.info("Dynamic Analysis Engine reset")
            
        except Exception as e:
            logger.error(f"Error resetting Dynamic Analysis Engine: {e}")

    def get_performance_info(self) -> Dict[str, Any]:
        """성능 정보 반환"""
        return {
            'analysis_count': self.analysis_count,
            'avg_analysis_time': self.avg_analysis_time,
            'current_complexity': self.current_complexity,
            'performance_mode': self.performance_mode,
            'history_size': len(self.analysis_history),
            'uptime': time.time() - self.start_time
        }