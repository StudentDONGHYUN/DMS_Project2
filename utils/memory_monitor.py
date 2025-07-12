"""
메모리 사용량 모니터링 및 관리 유틸리티

이 모듈은 DMS 시스템의 메모리 사용량을 모니터링하고
필요시 경고를 발생시키거나 정리 작업을 수행합니다.
"""

import psutil
import gc
import logging
import time
from typing import Dict, Optional, Callable
from threading import Thread, Event
from collections import deque
import asyncio

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """메모리 사용량 모니터링 클래스 - 성능 최적화"""
    
    def __init__(self, 
                 warning_threshold_mb: float = 800,
                 critical_threshold_mb: float = 1200,
                 cleanup_callback: Optional[Callable] = None):
        """
        Args:
            warning_threshold_mb: 경고 임계값 (MB)
            critical_threshold_mb: 위험 임계값 (MB)
            cleanup_callback: 정리 작업 콜백 함수
        """
        self.warning_threshold = warning_threshold_mb
        self.critical_threshold = critical_threshold_mb
        self.cleanup_callback = cleanup_callback
        
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        self.stop_event = Event()
        
        self.last_warning_time = 0
        self.last_cleanup_time = 0
        self.warning_interval = 30  # 30초마다 경고
        self.cleanup_interval = 60  # 60초마다 정리
        
        # 성능 최적화: list 대신 deque 사용 (O(1) append/popleft)
        self.memory_history = deque(maxlen=100)
        
        logger.info(f"MemoryMonitor 초기화 - 경고: {warning_threshold_mb}MB, 위험: {critical_threshold_mb}MB")

    def get_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량 반환"""
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            usage = {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'percent': memory_percent,
                'available_mb': psutil.virtual_memory().available / 1024 / 1024,
                'timestamp': time.time()
            }
            
            # 히스토리에 추가 (deque는 자동으로 maxlen 관리)
            self.memory_history.append(usage)
            
            return usage
            
        except Exception as e:
            logger.error(f"메모리 사용량 조회 실패: {e}")
            return {
                'rss_mb': 0,
                'vms_mb': 0,
                'percent': 0,
                'available_mb': 0,
                'timestamp': time.time()
            }

    def check_memory_status(self) -> tuple[str, dict]:
        """메모리 상태 확인 및 처리 - 성능 최적화된 버전"""
        usage = self.get_memory_usage()
        memory_mb = usage['rss_mb']
        current_time = time.time()
        
        if memory_mb >= self.critical_threshold:
            # 위험 수준 - 즉시 정리 작업
            if current_time - self.last_cleanup_time > 5:  # 5초 간격
                logger.critical(f"메모리 사용량 위험 수준: {memory_mb:.1f}MB")
                self._perform_emergency_cleanup()
                self.last_cleanup_time = current_time
            return "critical", usage
            
        elif memory_mb >= self.warning_threshold:
            # 경고 수준
            if current_time - self.last_warning_time > self.warning_interval:
                logger.warning(f"메모리 사용량 경고: {memory_mb:.1f}MB")
                self.last_warning_time = current_time
                
                # 정리 작업 수행 (간격 확인)
                if current_time - self.last_cleanup_time > self.cleanup_interval:
                    self._perform_cleanup()
                    self.last_cleanup_time = current_time
                    
            return "warning", usage
        else:
            return "normal", usage

    def get_memory_status_simple(self) -> str:
        """
        Backward compatibility method that returns only the status
        Use check_memory_status() for better performance
        """
        status, _ = self.check_memory_status()
        return status

    def _perform_cleanup(self):
        """일반적인 정리 작업"""
        logger.info("메모리 정리 작업 시작...")
        
        try:
            # 가비지 컬렉션 실행
            collected = gc.collect()
            logger.info(f"가비지 컬렉션 완료: {collected}개 객체 정리")
            
            # 사용자 정의 정리 콜백 실행
            if self.cleanup_callback:
                try:
                    self.cleanup_callback()
                    logger.info("사용자 정의 정리 작업 완료")
                except Exception as e:
                    logger.error(f"사용자 정의 정리 작업 실패: {e}")
                    
        except Exception as e:
            logger.error(f"메모리 정리 작업 실패: {e}")

    def _perform_emergency_cleanup(self):
        """긴급 정리 작업"""
        logger.critical("긴급 메모리 정리 작업 시작...")
        
        try:
            # 강제 가비지 컬렉션 (모든 세대)
            for i in range(3):
                collected = gc.collect()
                logger.info(f"긴급 GC {i+1}/3: {collected}개 객체 정리")
            
            # 사용자 정의 긴급 정리
            if hasattr(self.cleanup_callback, '__self__'):
                obj = self.cleanup_callback.__self__
                if hasattr(obj, 'emergency_cleanup'):
                    try:
                        obj.emergency_cleanup()
                        logger.info("긴급 정리 작업 완료")
                    except Exception as e:
                        logger.error(f"긴급 정리 작업 실패: {e}")
                        
        except Exception as e:
            logger.error(f"긴급 메모리 정리 실패: {e}")

    def start_monitoring(self, interval: float = 10.0):
        """메모리 모니터링 시작"""
        if self.monitoring:
            logger.warning("메모리 모니터링이 이미 실행 중입니다")
            return
        
        self.monitoring = True
        self.stop_event.clear()
        
        def monitor_loop():
            logger.info(f"메모리 모니터링 시작 (간격: {interval}초)")
            
            while not self.stop_event.wait(interval):
                try:
                    status, usage = self.check_memory_status()
                    
                    # 상태별 로깅 (normal은 debug 레벨) - 이미 usage 정보를 가지고 있음
                    if status == "normal":
                        logger.debug(f"메모리 상태: {status} ({usage['rss_mb']:.1f}MB)")
                    else:
                        logger.info(f"메모리 상태: {status} ({usage['rss_mb']:.1f}MB)")
                        
                except Exception as e:
                    logger.error(f"메모리 모니터링 오류: {e}")
            
            logger.info("메모리 모니터링 종료")
        
        self.monitor_thread = Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """메모리 모니터링 중지"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
            if self.monitor_thread.is_alive():
                logger.warning("메모리 모니터링 스레드가 2초 내에 종료되지 않음")

    def get_memory_report(self) -> Dict:
        """메모리 사용량 리포트 생성 - 안전한 division by zero 방지"""
        if not self.memory_history:
            return {"error": "메모리 히스토리 없음"}
        
        recent_usage = [entry['rss_mb'] for entry in self.memory_history]
        
        # Division by zero 방지
        if not recent_usage:
            return {"error": "메모리 데이터 없음"}
        
        report = {
            'current_mb': self.memory_history[-1]['rss_mb'],
            'peak_mb': max(entry['rss_mb'] for entry in self.memory_history),
            'average_mb': sum(recent_usage) / len(recent_usage),
            'warning_threshold_mb': self.warning_threshold,
            'critical_threshold_mb': self.critical_threshold,
            'history_count': len(self.memory_history),
            'monitoring_active': self.monitoring
        }
        
        return report

    def __enter__(self):
        """Context Manager 진입"""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager 종료"""
        self.stop_monitoring()


# 전역 메모리 모니터 인스턴스
_global_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor() -> MemoryMonitor:
    """전역 메모리 모니터 인스턴스 반환"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
    return _global_monitor


def start_global_monitoring(cleanup_callback: Optional[Callable] = None):
    """전역 메모리 모니터링 시작"""
    monitor = get_memory_monitor()
    if cleanup_callback:
        monitor.cleanup_callback = cleanup_callback
    monitor.start_monitoring()


def stop_global_monitoring():
    """전역 메모리 모니터링 중지"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()


def get_current_memory_usage() -> float:
    """현재 메모리 사용량 (MB) 반환"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except Exception as e:
        logger.error(f"메모리 사용량 조회 실패: {e}")
        return 0.0


def log_memory_usage(prefix: str = ""):
    """현재 메모리 사용량 로깅"""
    usage_mb = get_current_memory_usage()
    logger.info(f"{prefix}메모리 사용량: {usage_mb:.1f}MB")


# 데코레이터 함수들
def monitor_memory_usage(func):
    """함수 실행 전후 메모리 사용량을 모니터링하는 데코레이터"""
    def wrapper(*args, **kwargs):
        before = get_current_memory_usage()
        logger.debug(f"{func.__name__} 실행 전 메모리: {before:.1f}MB")
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            after = get_current_memory_usage()
            diff = after - before
            logger.debug(f"{func.__name__} 실행 후 메모리: {after:.1f}MB (변화: {diff:+.1f}MB)")
    
    return wrapper


def async_monitor_memory_usage(func):
    """비동기 함수용 메모리 모니터링 데코레이터"""
    async def wrapper(*args, **kwargs):
        before = get_current_memory_usage()
        logger.debug(f"{func.__name__} 실행 전 메모리: {before:.1f}MB")
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            after = get_current_memory_usage()
            diff = after - before
            logger.debug(f"{func.__name__} 실행 후 메모리: {after:.1f}MB (변화: {diff:+.1f}MB)")
    
    return wrapper


if __name__ == "__main__":
    # 테스트 코드
    import asyncio
    
    def test_cleanup():
        print("정리 작업 실행됨")
    
    async def run_test():
        """비동기 테스트 함수"""
        # 메모리 모니터 테스트
        with MemoryMonitor(warning_threshold_mb=100, cleanup_callback=test_cleanup) as monitor:
            print("메모리 모니터링 테스트 시작...")
            
            for i in range(5):
                usage = monitor.get_memory_usage()
                print(f"메모리 사용량: {usage['rss_mb']:.1f}MB")
                await asyncio.sleep(0.5)  # 비동기 슬립으로 변경
            
            print("메모리 리포트:")
            report = monitor.get_memory_report()
            for key, value in report.items():
                print(f"  {key}: {value}")
    
    # 비동기 테스트 실행
    asyncio.run(run_test())