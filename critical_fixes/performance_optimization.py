"""
Critical Fix #2: Performance Optimization Improvements
이 파일은 DMS 시스템의 성능 병목 현상을 해결하기 위한 최적화 기법들을 제시합니다.
"""

import asyncio
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List, Callable
from collections import deque
import psutil
import gc
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================================
# 1. 메모리 관리 최적화
# ============================================================================

@dataclass
class MemoryMetrics:
    """메모리 사용량 메트릭"""
    total_memory_mb: float
    used_memory_mb: float
    buffer_size: int
    cache_size: int
    timestamp: float

class MemoryOptimizer:
    """지능형 메모리 관리 시스템"""
    
    def __init__(self, max_memory_mb: float = 500.0):
        self.max_memory_mb = max_memory_mb
        self.warning_threshold = max_memory_mb * 0.8
        self.critical_threshold = max_memory_mb * 0.95
        self.metrics_history = deque(maxlen=100)
        self.cleanup_callbacks = []
        
    def register_cleanup_callback(self, callback: Callable[[], None]):
        """메모리 정리 콜백 등록"""
        self.cleanup_callbacks.append(callback)
    
    def get_current_memory_usage(self) -> MemoryMetrics:
        """현재 메모리 사용량 조회"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        metrics = MemoryMetrics(
            total_memory_mb=memory_info.rss / 1024 / 1024,
            used_memory_mb=memory_info.rss / 1024 / 1024,
            buffer_size=len(getattr(self, 'buffer_sizes', [])),
            cache_size=0,  # 실제 구현에서는 캐시 크기 계산
            timestamp=time.time()
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def check_and_optimize(self) -> bool:
        """메모리 상태 확인 및 최적화 실행"""
        metrics = self.get_current_memory_usage()
        
        if metrics.used_memory_mb > self.critical_threshold:
            logger.critical(f"Critical memory usage: {metrics.used_memory_mb:.1f}MB")
            self._emergency_cleanup()
            return True
            
        elif metrics.used_memory_mb > self.warning_threshold:
            logger.warning(f"High memory usage: {metrics.used_memory_mb:.1f}MB")
            self._gentle_cleanup()
            return True
            
        return False
    
    def _emergency_cleanup(self):
        """긴급 메모리 정리"""
        logger.info("긴급 메모리 정리 실행")
        
        # 등록된 정리 콜백 실행
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"정리 콜백 실행 실패: {e}")
        
        # 강제 가비지 컬렉션
        gc.collect()
        
        # 메트릭 히스토리 축소
        if len(self.metrics_history) > 50:
            self.metrics_history = deque(
                list(self.metrics_history)[-50:], 
                maxlen=100
            )
    
    def _gentle_cleanup(self):
        """점진적 메모리 정리"""
        logger.info("점진적 메모리 정리 실행")
        gc.collect()

# ============================================================================
# 2. CPU 최적화 및 병렬 처리
# ============================================================================

class ProcessingStrategy(Enum):
    """처리 전략 타입"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"

class CPUOptimizer:
    """CPU 사용량 최적화"""
    
    def __init__(self, max_workers: Optional[int] = None):
        cpu_count = psutil.cpu_count() or 4
        self.max_workers = max_workers or min(4, cpu_count)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.cpu_usage_history = deque(maxlen=50)
        self.processing_times = {}
        
    def get_optimal_strategy(self, task_complexity: float) -> ProcessingStrategy:
        """최적 처리 전략 결정"""
        cpu_usage = psutil.cpu_percent(interval=0.1)
        self.cpu_usage_history.append(cpu_usage)
        
        avg_cpu = sum(self.cpu_usage_history) / len(self.cpu_usage_history)
        
        if avg_cpu > 80 or task_complexity < 0.3:
            return ProcessingStrategy.SEQUENTIAL
        elif avg_cpu < 50 and task_complexity > 0.7:
            return ProcessingStrategy.PARALLEL
        else:
            return ProcessingStrategy.ADAPTIVE
    
    async def process_batch_optimized(self, 
                                    tasks: List[Callable], 
                                    strategy: Optional[ProcessingStrategy] = None) -> List[Any]:
        """최적화된 배치 처리"""
        if not tasks:
            return []
        
        if strategy is None:
            strategy = self.get_optimal_strategy(len(tasks) / 10.0)
        
        start_time = time.time()
        
        try:
            if strategy == ProcessingStrategy.SEQUENTIAL:
                results = await self._process_sequential(tasks)
            elif strategy == ProcessingStrategy.PARALLEL:
                results = await self._process_parallel(tasks)
            else:  # ADAPTIVE
                results = await self._process_adaptive(tasks)
            
            processing_time = time.time() - start_time
            self.processing_times[strategy.value] = processing_time
            
            logger.debug(f"배치 처리 완료: {strategy.value}, {processing_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"배치 처리 실패 ({strategy.value}): {e}")
            raise
    
    async def _process_sequential(self, tasks: List[Callable]) -> List[Any]:
        """순차 처리"""
        results = []
        for task in tasks:
            try:
                if asyncio.iscoroutinefunction(task):
                    result = await task()
                else:
                    result = task()
                results.append(result)
            except Exception as e:
                logger.warning(f"순차 처리 중 작업 실패: {e}")
                results.append(None)
        return results
    
    async def _process_parallel(self, tasks: List[Callable]) -> List[Any]:
        """병렬 처리"""
        loop = asyncio.get_event_loop()
        futures = []
        
        for task in tasks:
            if asyncio.iscoroutinefunction(task):
                futures.append(task())
            else:
                futures.append(loop.run_in_executor(self.thread_pool, task))
        
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # 예외를 None으로 변환
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"병렬 처리 중 작업 실패: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_adaptive(self, tasks: List[Callable]) -> List[Any]:
        """적응형 처리 (중요도 기반)"""
        # 작업을 중요도에 따라 분류
        high_priority = tasks[:len(tasks)//2]
        low_priority = tasks[len(tasks)//2:]
        
        # 고우선순위 작업은 순차 처리
        high_results = await self._process_sequential(high_priority)
        
        # 저우선순위 작업은 병렬 처리
        low_results = await self._process_parallel(low_priority)
        
        return high_results + low_results

# ============================================================================
# 3. 버퍼 및 캐시 최적화
# ============================================================================

class SmartBuffer:
    """지능형 버퍼 관리"""
    
    def __init__(self, max_size: int = 100, auto_cleanup: bool = True):
        self.max_size = max_size
        self.auto_cleanup = auto_cleanup
        self.buffer = deque(maxlen=max_size)
        self.access_times = {}
        self.memory_optimizer = MemoryOptimizer()
        
    def add_item(self, key: Any, item: Any, priority: float = 1.0):
        """아이템 추가 (우선순위 기반)"""
        timestamp = time.time()
        
        # 메모리 체크
        if self.auto_cleanup and self.memory_optimizer.check_and_optimize():
            self._cleanup_low_priority_items()
        
        # 버퍼가 가득 찬 경우 오래된 아이템 제거
        if len(self.buffer) >= self.max_size:
            self._remove_oldest_item()
        
        self.buffer.append({
            'key': key,
            'item': item,
            'priority': priority,
            'timestamp': timestamp,
            'access_count': 0
        })
        
        self.access_times[key] = timestamp
    
    def get_item(self, key: Any) -> Optional[Any]:
        """아이템 조회"""
        for item_data in self.buffer:
            if item_data['key'] == key:
                # 액세스 카운트 및 시간 업데이트
                item_data['access_count'] += 1
                self.access_times[key] = time.time()
                return item_data['item']
        return None
    
    def _cleanup_low_priority_items(self):
        """낮은 우선순위 아이템 정리"""
        if len(self.buffer) < 10:
            return
        
        # 우선순위와 액세스 빈도 기준으로 정렬
        sorted_items = sorted(
            self.buffer, 
            key=lambda x: (x['priority'], x['access_count']), 
            reverse=True
        )
        
        # 하위 30% 제거
        keep_count = int(len(sorted_items) * 0.7)
        self.buffer = deque(sorted_items[:keep_count], maxlen=self.max_size)
        
        logger.debug(f"버퍼 정리: {len(sorted_items)} → {keep_count}")
    
    def _remove_oldest_item(self):
        """가장 오래된 아이템 제거"""
        if self.buffer:
            removed = self.buffer.popleft()
            self.access_times.pop(removed['key'], None)

# ============================================================================
# 4. 비동기 처리 최적화
# ============================================================================

class AsyncOptimizer:
    """비동기 처리 최적화"""
    
    def __init__(self):
        self.semaphores = {}
        self.locks = {}
        self.processing_queue = asyncio.Queue(maxsize=1000)
        self.worker_tasks = []
        
    async def get_semaphore(self, name: str, limit: int = 10) -> asyncio.Semaphore:
        """세마포어 관리"""
        if name not in self.semaphores:
            self.semaphores[name] = asyncio.Semaphore(limit)
        return self.semaphores[name]
    
    async def get_lock(self, name: str) -> asyncio.Lock:
        """락 관리"""
        if name not in self.locks:
            self.locks[name] = asyncio.Lock()
        return self.locks[name]
    
    async def safe_concurrent_execution(self, 
                                      tasks: List[Callable], 
                                      max_concurrent: int = 5,
                                      timeout: float = 30.0) -> List[Any]:
        """안전한 동시 실행"""
        semaphore = await self.get_semaphore(f"concurrent_{max_concurrent}", max_concurrent)
        
        async def execute_with_semaphore(task):
            async with semaphore:
                try:
                    return await asyncio.wait_for(task(), timeout=timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"작업 타임아웃: {task.__name__}")
                    return None
                except Exception as e:
                    logger.error(f"작업 실행 실패: {e}")
                    return None
        
        # 모든 작업을 동시에 시작하되 세마포어로 제한
        results = await asyncio.gather(
            *[execute_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        return results

# ============================================================================
# 5. 통합 성능 모니터링
# ============================================================================

@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    processing_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_fps: float
    error_rate: float
    timestamp: float

class PerformanceMonitor:
    """통합 성능 모니터링"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.memory_optimizer = MemoryOptimizer()
        self.cpu_optimizer = CPUOptimizer()
        self.async_optimizer = AsyncOptimizer()
        
    async def measure_performance(self, operation: Callable) -> PerformanceMetrics:
        """성능 측정"""
        start_time = time.time()
        start_memory = self.memory_optimizer.get_current_memory_usage()
        
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation()
            else:
                result = operation()
            
            success = True
            
        except Exception as e:
            logger.error(f"성능 측정 중 오류: {e}")
            success = False
            result = None
        
        end_time = time.time()
        end_memory = self.memory_optimizer.get_current_memory_usage()
        
        processing_time = (end_time - start_time) * 1000  # ms
        self.processing_times.append(processing_time)
        
        # FPS 계산 (최근 측정값 기반)
        avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        throughput_fps = 1000 / avg_processing_time if avg_processing_time > 0 else 0
        
        metrics = PerformanceMetrics(
            processing_time_ms=processing_time,
            memory_usage_mb=end_memory.used_memory_mb,
            cpu_usage_percent=psutil.cpu_percent(),
            throughput_fps=throughput_fps,
            error_rate=0.0 if success else 1.0,
            timestamp=end_time
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 통계"""
        if not self.metrics_history:
            return {}
        
        processing_times = [m.processing_time_ms for m in self.metrics_history]
        memory_usage = [m.memory_usage_mb for m in self.metrics_history]
        cpu_usage = [m.cpu_usage_percent for m in self.metrics_history]
        throughput = [m.throughput_fps for m in self.metrics_history]
        error_rates = [m.error_rate for m in self.metrics_history]
        
        return {
            'processing_time': {
                'avg': sum(processing_times) / len(processing_times),
                'min': min(processing_times),
                'max': max(processing_times),
                'p95': sorted(processing_times)[int(len(processing_times) * 0.95)]
            },
            'memory_usage': {
                'avg': sum(memory_usage) / len(memory_usage),
                'max': max(memory_usage),
                'current': memory_usage[-1]
            },
            'cpu_usage': {
                'avg': sum(cpu_usage) / len(cpu_usage),
                'max': max(cpu_usage),
                'current': cpu_usage[-1]
            },
            'throughput': {
                'avg': sum(throughput) / len(throughput),
                'max': max(throughput),
                'current': throughput[-1]
            },
            'error_rate': sum(error_rates) / len(error_rates),
            'total_samples': len(self.metrics_history)
        }

# ============================================================================
# 6. 사용 예시
# ============================================================================

class OptimizedDMSProcessor:
    """최적화된 DMS 프로세서 예시"""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.buffer = SmartBuffer(max_size=50)
        self.memory_optimizer = MemoryOptimizer(max_memory_mb=300)
        self.cpu_optimizer = CPUOptimizer()
        self.async_optimizer = AsyncOptimizer()
        
        # 메모리 정리 콜백 등록
        self.memory_optimizer.register_cleanup_callback(self._cleanup_buffers)
    
    def _cleanup_buffers(self):
        """버퍼 정리"""
        self.buffer._cleanup_low_priority_items()
        logger.info("버퍼 정리 완료")
    
    async def process_frame_optimized(self, frame_data: Any) -> Optional[Dict[str, Any]]:
        """최적화된 프레임 처리"""
        
        # 성능 측정 시작
        async def process_operation():
            # 캐시에서 유사한 프레임 확인
            cached_result = self.buffer.get_item(f"frame_{hash(str(frame_data))}")
            if cached_result:
                logger.debug("캐시된 결과 사용")
                return cached_result
            
            # 실제 처리 로직
            tasks = [
                self._process_face_async,
                self._process_pose_async,
                self._process_hands_async
            ]
            
            # 최적화된 병렬 처리
            results = await self.cpu_optimizer.process_batch_optimized(
                tasks, 
                ProcessingStrategy.ADAPTIVE
            )
            
            # 결과 융합
            fused_result = self._fuse_results(results)
            
            # 결과 캐싱
            self.buffer.add_item(
                f"frame_{hash(str(frame_data))}", 
                fused_result, 
                priority=0.8
            )
            
            return fused_result
        
        # 성능 측정과 함께 실행
        metrics = await self.performance_monitor.measure_performance(process_operation)
        
        logger.debug(f"프레임 처리 완료: {metrics.processing_time_ms:.1f}ms")
        
        return await process_operation()
    
    async def _process_face_async(self):
        """비동기 얼굴 처리"""
        await asyncio.sleep(0.01)  # 시뮬레이션
        return {"face": "processed"}
    
    async def _process_pose_async(self):
        """비동기 자세 처리"""
        await asyncio.sleep(0.008)  # 시뮬레이션
        return {"pose": "processed"}
    
    async def _process_hands_async(self):
        """비동기 손 처리"""
        await asyncio.sleep(0.005)  # 시뮬레이션
        return {"hands": "processed"}
    
    def _fuse_results(self, results: List[Any]) -> Dict[str, Any]:
        """결과 융합"""
        fused = {}
        for result in results:
            if result:
                fused.update(result)
        return fused
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """최적화 상태 조회"""
        performance_summary = self.performance_monitor.get_performance_summary()
        memory_metrics = self.memory_optimizer.get_current_memory_usage()
        
        return {
            'performance_score': min(1.0, 100 / performance_summary.get('processing_time', {}).get('avg', 100)),
            'memory_efficiency': max(0.0, 1.0 - memory_metrics.used_memory_mb / 300),
            'throughput_fps': performance_summary.get('throughput', {}).get('current', 0),
            'error_rate': performance_summary.get('error_rate', 0),
            'optimization_active': True
        }

if __name__ == "__main__":
    # 사용 예시
    async def main():
        processor = OptimizedDMSProcessor()
        
        # 샘플 프레임 처리
        for i in range(10):
            frame_data = f"frame_{i}"
            result = await processor.process_frame_optimized(frame_data)
            print(f"프레임 {i} 처리 완료: {result}")
        
        # 최적화 상태 출력
        status = processor.get_optimization_status()
        print(f"최적화 상태: {status}")
    
    asyncio.run(main())