"""
Performance Optimizer - Enhanced System
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
            
            # Calculate FPS
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                self.current_fps = self.frame_count / elapsed_time
            
            # Update system metrics periodically
            if self.frame_count % 30 == 0:  # Every 30 frames
                self._update_system_metrics()
                
        except Exception as e:
            logger.error(f"Error updating frame processing: {e}")

    def _update_system_metrics(self):
        """시스템 메트릭 업데이트"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.memory_usage_mb = memory_info.rss / 1024 / 1024
            self.cpu_usage_percent = process.cpu_percent()
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

    def get_performance_info(self) -> Dict[str, Any]:
        """성능 정보 반환"""
        return {
            'frame_count': self.frame_count,
            'current_fps': self.current_fps,
            'target_fps': self.target_fps,
            'avg_processing_time_ms': self.avg_processing_time_ms,
            'max_processing_time_ms': self.max_processing_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'memory_threshold_mb': self.memory_threshold_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'uptime_seconds': time.time() - self.start_time
        }

    def is_performance_acceptable(self) -> bool:
        """성능이 허용 가능한 수준인지 확인"""
        return (
            self.current_fps >= self.target_fps * 0.8 and
            self.avg_processing_time_ms <= self.max_processing_time_ms and
            self.memory_usage_mb <= self.memory_threshold_mb
        )

    def get_optimization_recommendations(self) -> list:
        """최적화 권장사항 반환"""
        recommendations = []
        
        if self.current_fps < self.target_fps * 0.8:
            recommendations.append("FPS가 목표치보다 낮습니다. 처리 파이프라인 최적화가 필요합니다.")
        
        if self.avg_processing_time_ms > self.max_processing_time_ms:
            recommendations.append("프레임 처리 시간이 임계값을 초과합니다. 알고리즘 최적화를 검토하세요.")
        
        if self.memory_usage_mb > self.memory_threshold_mb:
            recommendations.append("메모리 사용량이 높습니다. 메모리 누수를 확인하세요.")
        
        if self.cpu_usage_percent > 80:
            recommendations.append("CPU 사용량이 높습니다. 백그라운드 프로세스를 확인하세요.")
        
        if not recommendations:
            recommendations.append("성능이 양호합니다.")
        
        return recommendations

    def reset(self):
        """성능 통계 리셋"""
        self.start_time = time.time()
        self.frame_count = 0
        self.processing_times.clear()
        self.current_fps = 0.0
        self.avg_processing_time_ms = 0.0
        logger.info("Performance statistics reset")

    def set_targets(self, target_fps: Optional[float] = None, max_processing_time_ms: Optional[float] = None, memory_threshold_mb: Optional[float] = None):
        """성능 목표값 설정"""
        if target_fps is not None:
            self.target_fps = target_fps
        if max_processing_time_ms is not None:
            self.max_processing_time_ms = max_processing_time_ms
        if memory_threshold_mb is not None:
            self.memory_threshold_mb = memory_threshold_mb
        
        logger.info(f"Performance targets updated: FPS={self.target_fps}, "
                   f"Processing={self.max_processing_time_ms}ms, "
                   f"Memory={self.memory_threshold_mb}MB")

    # Legacy compatibility methods
    def log_performance(self, processing_time, fps):
        """Legacy compatibility method"""
        processing_time_ms = processing_time * 1000 if processing_time < 1 else processing_time
        self.update_frame_processing(processing_time_ms)

    def get_optimization_status(self):
        """Legacy compatibility method"""
        return {
            'optimization_active': not self.is_performance_acceptable(),
            'current_fps': self.current_fps,
            'target_fps': self.target_fps
        }
