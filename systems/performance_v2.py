"""
Performance Optimizer v2
통합 시스템과 호환되는 성능 최적화기
기존 코드와 개선된 코드를 통합하여 호환성과 성능을 모두 확보
"""

import time
import logging
import psutil
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """통합 성능 최적화기 - 기존 코드와 개선된 코드의 통합"""

    def __init__(self):
        """성능 최적화기 초기화"""
        self.start_time = time.time()
        self.frame_count = 0
        self.processing_times = []
        self.max_history = 100
        
        # Performance thresholds
        self.target_fps = 30.0
        self.max_processing_time_ms = 200.0
        self.memory_threshold_mb = 500.0
        
        # Current performance state
        self.current_fps = 0.0
        self.avg_processing_time_ms = 0.0
        self.memory_usage_mb = 0.0
        self.cpu_usage_percent = 0.0
        
        logger.info("Performance Optimizer initialized")

    def update_frame_processing(self, processing_time_ms: float):
        """프레임 처리 시간 업데이트"""
        try:
            self.frame_count += 1
            
            # Update processing times history
            self.processing_times.append(processing_time_ms)
            if len(self.processing_times) > self.max_history:
                self.processing_times.pop(0)
            
            # Calculate average processing time
            self.avg_processing_time_ms = sum(self.processing_times) / len(self.processing_times)
            
            # Calculate current FPS
            elapsed_time = time.time() - self.start_time
            self.current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0.0
            
            # Update system metrics
            self._update_system_metrics()
            
        except Exception as e:
            logger.error(f"Error updating frame processing: {e}")

    def _update_system_metrics(self):
        """시스템 메트릭 업데이트"""
        try:
            # Memory usage
            process = psutil.Process()
            self.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            
            # CPU usage
            self.cpu_usage_percent = psutil.cpu_percent(interval=0.1)
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

    def get_performance_info(self) -> Dict[str, Any]:
        """성능 정보 반환"""
        return {
            "frame_count": self.frame_count,
            "current_fps": self.current_fps,
            "target_fps": self.target_fps,
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "max_processing_time_ms": self.max_processing_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "memory_threshold_mb": self.memory_threshold_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "elapsed_time": time.time() - self.start_time
        }

    def is_performance_acceptable(self) -> bool:
        """성능이 허용 가능한지 확인"""
        return (
            self.current_fps >= self.target_fps * 0.8 and  # 80% of target FPS
            self.avg_processing_time_ms <= self.max_processing_time_ms and
            self.memory_usage_mb <= self.memory_threshold_mb
        )

    def get_optimization_recommendations(self) -> list:
        """최적화 권장사항 반환"""
        recommendations = []
        
        if self.current_fps < self.target_fps * 0.8:
            recommendations.append("Consider reducing processing complexity to improve FPS")
        
        if self.avg_processing_time_ms > self.max_processing_time_ms:
            recommendations.append("Frame processing time exceeds threshold - optimize algorithms")
        
        if self.memory_usage_mb > self.memory_threshold_mb:
            recommendations.append("Memory usage is high - consider cleanup or reduce buffer sizes")
        
        if self.cpu_usage_percent > 80:
            recommendations.append("CPU usage is high - consider reducing workload")
        
        return recommendations

    def reset(self):
        """성능 통계 초기화"""
        self.start_time = time.time()
        self.frame_count = 0
        self.processing_times.clear()
        self.current_fps = 0.0
        self.avg_processing_time_ms = 0.0
        
        logger.info("Performance statistics reset")

    def set_targets(self, target_fps: float = None, max_processing_time_ms: float = None, memory_threshold_mb: float = None):
        """성능 목표 설정"""
        if target_fps is not None:
            self.target_fps = target_fps
        
        if max_processing_time_ms is not None:
            self.max_processing_time_ms = max_processing_time_ms
        
        if memory_threshold_mb is not None:
            self.memory_threshold_mb = memory_threshold_mb
        
        logger.info(f"Performance targets updated: FPS={self.target_fps}, Time={self.max_processing_time_ms}ms, Memory={self.memory_threshold_mb}MB")