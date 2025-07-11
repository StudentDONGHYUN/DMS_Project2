import gc
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)

class MemoryManager:
    """메모리 관리자"""

    def __init__(self):
        self.cleanup_interval = 300
        self.last_cleanup = time.time()
        self.memory_usage_history = deque(maxlen=100)

    def check_and_cleanup(self):
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            collected = gc.collect()
            self.last_cleanup = current_time
            if collected > 0:
                logger.info(f"메모리 정리 완료: {collected}개 객체 정리")

    def log_memory_usage(self):
        """메모리 사용량 추적"""
        try:
            import psutil

            memory_percent = psutil.virtual_memory().percent
            self.memory_usage_history.append(memory_percent)

            if memory_percent > 85:
                logger.warning(f"높은 메모리 사용량 감지: {memory_percent:.1f}%")
                self.check_and_cleanup()
        except ImportError:
            pass
