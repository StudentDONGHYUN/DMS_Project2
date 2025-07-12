"""
Memory Monitor v2
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
            self.current_cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Update peak memory
            if self.current_memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = self.current_memory_mb
            
            # Add to history
            history_entry = {
                'timestamp': current_time,
                'memory_mb': self.current_memory_mb,
                'cpu_percent': self.current_cpu_percent
            }
            
            self.memory_history.append(history_entry)
            
            # Keep history size manageable
            if len(self.memory_history) > self.max_history:
                self.memory_history = self.memory_history[-self.max_history:]
            
            self.last_update_time = current_time
            
        except Exception as e:
            logger.error(f"Error updating memory monitor: {e}")

    def get_memory_usage(self) -> float:
        """메모리 사용량 반환 (MB)"""
        self.update()
        return self.current_memory_mb

    def get_cpu_usage(self) -> float:
        """CPU 사용량 반환 (%)"""
        self.update()
        return self.current_cpu_percent

    def get_memory_info(self) -> Dict[str, Any]:
        """메모리 정보 반환"""
        self.update()
        
        return {
            'current_memory_mb': self.current_memory_mb,
            'peak_memory_mb': self.peak_memory_mb,
            'current_cpu_percent': self.current_cpu_percent,
            'memory_history_size': len(self.memory_history),
            'uptime': time.time() - self.start_time
        }

    def get_memory_trend(self) -> Dict[str, Any]:
        """메모리 사용량 트렌드 반환"""
        try:
            if len(self.memory_history) < 2:
                return {'trend': 'stable', 'change_mb': 0.0}
            
            # Get recent memory values
            recent_memory = [entry['memory_mb'] for entry in self.memory_history[-10:]]
            
            if len(recent_memory) < 2:
                return {'trend': 'stable', 'change_mb': 0.0}
            
            # Calculate trend
            first_avg = sum(recent_memory[:len(recent_memory)//2]) / (len(recent_memory)//2)
            second_avg = sum(recent_memory[len(recent_memory)//2:]) / (len(recent_memory)//2)
            
            change_mb = second_avg - first_avg
            
            if change_mb > 10:
                trend = 'increasing'
            elif change_mb < -10:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            return {
                'trend': trend,
                'change_mb': change_mb,
                'first_avg_mb': first_avg,
                'second_avg_mb': second_avg
            }
            
        except Exception as e:
            logger.error(f"Error calculating memory trend: {e}")
            return {'trend': 'unknown', 'change_mb': 0.0}

    def is_memory_usage_high(self, threshold_mb: float = 500.0) -> bool:
        """메모리 사용량이 높은지 확인"""
        return self.current_memory_mb > threshold_mb

    def is_cpu_usage_high(self, threshold_percent: float = 80.0) -> bool:
        """CPU 사용량이 높은지 확인"""
        return self.current_cpu_percent > threshold_percent

    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 반환"""
        try:
            # Get system memory info
            system_memory = psutil.virtual_memory()
            
            return {
                'total_memory_mb': system_memory.total / 1024 / 1024,
                'available_memory_mb': system_memory.available / 1024 / 1024,
                'memory_percent': system_memory.percent,
                'cpu_count': psutil.cpu_count(),
                'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}

    def get_recommendations(self) -> list:
        """메모리 사용량 기반 권장사항 반환"""
        recommendations = []
        
        if self.is_memory_usage_high():
            recommendations.append("Memory usage is high - consider reducing buffer sizes or cleaning up unused data")
        
        if self.is_cpu_usage_high():
            recommendations.append("CPU usage is high - consider reducing processing complexity")
        
        memory_trend = self.get_memory_trend()
        if memory_trend['trend'] == 'increasing':
            recommendations.append("Memory usage is increasing - check for memory leaks")
        
        return recommendations

    def reset(self):
        """메모리 모니터 초기화"""
        try:
            self.memory_history.clear()
            self.peak_memory_mb = 0.0
            self.start_time = time.time()
            
            logger.info("Memory Monitor reset")
            
        except Exception as e:
            logger.error(f"Error resetting Memory Monitor: {e}")

    def set_update_interval(self, interval: float):
        """업데이트 간격 설정"""
        self.update_interval = max(0.1, interval)  # Minimum 0.1 seconds
        logger.info(f"Memory monitor update interval set to {self.update_interval} seconds")

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
            'avg_memory_mb': sum(memory_values) / len(memory_values),
            'min_memory_mb': min(memory_values),
            'max_memory_mb': max(memory_values),
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'min_cpu_percent': min(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'peak_memory_mb': self.peak_memory_mb,
            'history_size': len(self.memory_history)
        }