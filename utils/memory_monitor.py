"""
Memory Monitor - Enhanced System
통합 시스템과 호환되는 메모리 모니터
기존 코드와 개선된 코드를 통합하여 호환성과 성능을 모두 확보
"""

import time
import logging
import psutil
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """통합 메모리 모니터 - 기존 코드와 개선된 코드의 통합"""

    def __init__(self):
        """메모리 모니터 초기화"""
        self.start_time = time.time()
        self.memory_history = []
        self.max_history = 100
        
        # Current state
        self.current_memory_mb = 0.0
        self.current_cpu_percent = 0.0
        self.peak_memory_mb = 0.0
        
        # Monitoring settings
        self.monitoring_enabled = True
        self.update_interval = 1.0  # seconds
        self.last_update_time = 0.0
        
        logger.info("Memory Monitor initialized")

    def update(self):
        """메모리 상태 업데이트"""
        try:
            if not self.monitoring_enabled:
                return
            
            current_time = time.time()
            if current_time - self.last_update_time < self.update_interval:
                return
            
            # Get current memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.current_memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            self.current_cpu_percent = process.cpu_percent()
            
            # Update peak memory
            if self.current_memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = self.current_memory_mb
            
            # Add to history
            self.memory_history.append({
                'timestamp': current_time,
                'memory_mb': self.current_memory_mb,
                'cpu_percent': self.current_cpu_percent
            })
            
            # Keep history size manageable
            if len(self.memory_history) > self.max_history:
                self.memory_history.pop(0)
            
            self.last_update_time = current_time
            
        except Exception as e:
            logger.error(f"Memory monitor update failed: {e}")

    def get_memory_usage(self) -> float:
        """현재 메모리 사용량 반환 (MB)"""
        self.update()
        return self.current_memory_mb

    def get_cpu_usage(self) -> float:
        """현재 CPU 사용량 반환 (%)"""
        self.update()
        return self.current_cpu_percent

    def get_memory_info(self) -> Dict[str, Any]:
        """상세 메모리 정보 반환"""
        self.update()
        
        return {
            'current_mb': self.current_memory_mb,
            'peak_mb': self.peak_memory_mb,
            'cpu_percent': self.current_cpu_percent,
            'uptime_seconds': time.time() - self.start_time,
            'history_size': len(self.memory_history)
        }

    def get_memory_trend(self) -> Dict[str, Any]:
        """메모리 사용 트렌드 분석"""
        if len(self.memory_history) < 2:
            return {
                'trend': 'insufficient_data',
                'rate_mb_per_sec': 0.0,
                'average_mb': self.current_memory_mb,
                'variance': 0.0
            }
        
        # Calculate trend over last 10 entries
        recent_history = self.memory_history[-10:]
        
        if len(recent_history) >= 2:
            start_entry = recent_history[0]
            end_entry = recent_history[-1]
            
            time_diff = end_entry['timestamp'] - start_entry['timestamp']
            memory_diff = end_entry['memory_mb'] - start_entry['memory_mb']
            
            rate = memory_diff / time_diff if time_diff > 0 else 0.0
            
            # Calculate average and variance
            memory_values = [entry['memory_mb'] for entry in recent_history]
            average = sum(memory_values) / len(memory_values)
            variance = sum((x - average) ** 2 for x in memory_values) / len(memory_values)
            
            # Determine trend
            if rate > 1.0:  # Growing more than 1MB/sec
                trend = 'increasing'
            elif rate < -1.0:  # Decreasing more than 1MB/sec
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            return {
                'trend': trend,
                'rate_mb_per_sec': rate,
                'average_mb': average,
                'variance': variance
            }
        
        return {
            'trend': 'stable',
            'rate_mb_per_sec': 0.0,
            'average_mb': self.current_memory_mb,
            'variance': 0.0
        }

    def is_memory_usage_high(self, threshold_mb: float = 500.0) -> bool:
        """메모리 사용량이 임계값을 초과하는지 확인"""
        return self.get_memory_usage() > threshold_mb

    def is_cpu_usage_high(self, threshold_percent: float = 80.0) -> bool:
        """CPU 사용량이 임계값을 초과하는지 확인"""
        return self.get_cpu_usage() > threshold_percent

    def get_system_info(self) -> Dict[str, Any]:
        """시스템 전체 정보 반환"""
        try:
            virtual_memory = psutil.virtual_memory()
            cpu_info = psutil.cpu_percent(interval=None)
            
            return {
                'system_memory_total_gb': virtual_memory.total / (1024**3),
                'system_memory_available_gb': virtual_memory.available / (1024**3),
                'system_memory_percent': virtual_memory.percent,
                'system_cpu_percent': cpu_info,
                'process_memory_mb': self.current_memory_mb,
                'process_cpu_percent': self.current_cpu_percent
            }
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {}

    def get_recommendations(self) -> list:
        """메모리 사용량 기반 권장사항 제공"""
        recommendations = []
        
        if self.is_memory_usage_high():
            recommendations.append("High memory usage detected. Consider reducing buffer sizes or enabling memory optimization.")
        
        if self.is_cpu_usage_high():
            recommendations.append("High CPU usage detected. Consider reducing processing frequency or enabling performance mode.")
        
        trend_info = self.get_memory_trend()
        if trend_info['trend'] == 'increasing' and trend_info['rate_mb_per_sec'] > 2.0:
            recommendations.append("Memory usage is increasing rapidly. Check for memory leaks.")
        
        if not recommendations:
            recommendations.append("Memory usage is within normal range.")
        
        return recommendations

    def reset(self):
        """메모리 모니터 리셋"""
        self.start_time = time.time()
        self.memory_history.clear()
        self.peak_memory_mb = 0.0
        self.last_update_time = 0.0
        logger.info("Memory Monitor reset")

    def set_update_interval(self, interval: float):
        """업데이트 간격 설정"""
        self.update_interval = max(0.1, interval)  # Minimum 100ms
        logger.info(f"Memory monitor update interval set to {self.update_interval}s")

    def enable_monitoring(self, enabled: bool):
        """모니터링 활성화/비활성화"""
        self.monitoring_enabled = enabled
        logger.info(f"Memory monitoring {'enabled' if enabled else 'disabled'}")

    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        if not self.memory_history:
            return {}
        
        memory_values = [entry['memory_mb'] for entry in self.memory_history]
        cpu_values = [entry['cpu_percent'] for entry in self.memory_history]
        
        return {
            'memory_min_mb': min(memory_values),
            'memory_max_mb': max(memory_values),
            'memory_avg_mb': sum(memory_values) / len(memory_values),
            'cpu_min_percent': min(cpu_values),
            'cpu_max_percent': max(cpu_values),
            'cpu_avg_percent': sum(cpu_values) / len(cpu_values),
            'sample_count': len(self.memory_history),
            'uptime_seconds': time.time() - self.start_time
        }


# Legacy compatibility function
def log_memory_usage(context: str = ""):
    """Legacy function for backward compatibility"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        if context:
            logger.info(f"Memory usage ({context}): {memory_mb:.2f} MB")
        else:
            logger.info(f"Memory usage: {memory_mb:.2f} MB")
            
    except Exception as e:
        logger.error(f"Failed to log memory usage: {e}")


# Global memory monitor instance
_global_monitor = None

def get_global_memory_monitor() -> MemoryMonitor:
    """전역 메모리 모니터 인스턴스 반환"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
    return _global_monitor