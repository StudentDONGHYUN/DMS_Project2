"""
Analysis Orchestrator (S-Class): 디지털 시스템 지휘자
- [S-Class] 적응형 파이프라인 관리 (성능에 따른 동적 우선순위 조정)
- [S-Class] 장애 허용 시스템 (Fault Tolerance) - 부분적 실패 시에도 지속 동작
- [S-Class] 실시간 성능 모니터링 및 자동 최적화
- [S-Class] 예측적 리소스 관리 (다음 프레임 처리 요구사항 예측)
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import inspect

from core.interfaces import (
    IMetricsUpdater, IFaceDataProcessor, IPoseDataProcessor, 
    IHandDataProcessor, IObjectDataProcessor, IMultiModalAnalyzer
)

logger = logging.getLogger(__name__)


class ProcessorStatus(Enum):
    """프로세서 상태"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # 성능 저하
    FAILING = "failing"    # 간헐적 실패
    FAILED = "failed"      # 완전 실패


class PipelineMode(Enum):
    """파이프라인 실행 모드"""
    FULL_PARALLEL = "full_parallel"      # 모든 프로세서 병렬 실행
    SELECTIVE_PARALLEL = "selective"     # 중요한 프로세서만 병렬
    SEQUENTIAL_SAFE = "sequential"       # 순차 실행 (안전 모드)
    EMERGENCY_MINIMAL = "emergency"      # 최소한의 핵심 기능만


@dataclass
class ProcessorPerformance:
    """프로세서 성능 메트릭"""
    avg_execution_time: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    last_success_time: float = 0.0
    consecutive_failures: int = 0
    status: ProcessorStatus = ProcessorStatus.HEALTHY


@dataclass
class PipelineExecution:
    """파이프라인 실행 결과"""
    total_time: float
    successful_processors: List[str]
    failed_processors: List[str]
    degraded_performance: bool
    confidence_score: float


class AnalysisOrchestrator:
    """
    Analysis Orchestrator (S-Class)
    
    이 클래스는 마치 숙련된 오케스트라 지휘자처럼 각 분석 모듈의
    성능을 실시간으로 모니터링하고, 상황에 맞게 최적의 실행 전략을
    동적으로 선택합니다.
    """
    
    def __init__(
        self,
        metrics_updater: IMetricsUpdater,
        face_processor: IFaceDataProcessor,
        pose_processor: IPoseDataProcessor,
        hand_processor: IHandDataProcessor,
        object_processor: IObjectDataProcessor,
        fusion_engine: IMultiModalAnalyzer,
    ):
        self.metrics_updater = metrics_updater
        self.processors = {
            'face': face_processor,
            'pose': pose_processor,
            'hand': hand_processor,
            'object': object_processor
        }
        self.fusion_engine = fusion_engine
        
        # --- S-Class 고도화: 지능형 관리 시스템 ---
        self.processor_performance = {
            name: ProcessorPerformance() for name in self.processors.keys()
        }
        self.current_pipeline_mode = PipelineMode.FULL_PARALLEL
        self.performance_history = []
        self.resource_predictor = ResourcePredictor()
        
        # 적응형 타임아웃 설정
        self.adaptive_timeouts = {
            'face': 0.1,
            'pose': 0.08,
            'hand': 0.06,
            'object': 0.12
        }
        
        logger.info("AnalysisOrchestrator (S-Class) 초기화 완료 - 지능형 지휘자 준비됨")

    async def process_frame_data(
        self, mediapipe_results: Dict[str, Any], timestamp: float
    ) -> Dict[str, Any]:
        """[S-Class] 적응형 프레임 처리 파이프라인"""
        
        # 1. 실행 전 시스템 상태 평가
        system_health = self._assess_system_health()
        optimal_mode = self._determine_optimal_pipeline_mode(system_health)
        
        # 2. 파이프라인 모드 변경이 필요한 경우 적응
        if optimal_mode != self.current_pipeline_mode:
            logger.info(f"파이프라인 모드 변경: {self.current_pipeline_mode.value} → {optimal_mode.value}")
            self.current_pipeline_mode = optimal_mode
        
        # 3. 다음 프레임 리소스 요구사항 예측
        predicted_load = self.resource_predictor.predict_next_frame_load(
            mediapipe_results, self.processor_performance
        )
        
        # 4. 예측된 부하에 따른 타임아웃 조정
        self._adjust_adaptive_timeouts(predicted_load)
        
        # 5. 선택된 모드에 따른 실행
        execution_start = time.time()
        
        if optimal_mode == PipelineMode.FULL_PARALLEL:
            results = await self._execute_full_parallel_pipeline(mediapipe_results, timestamp)
        elif optimal_mode == PipelineMode.SELECTIVE_PARALLEL:
            results = await self._execute_selective_pipeline(mediapipe_results, timestamp)
        elif optimal_mode == PipelineMode.SEQUENTIAL_SAFE:
            results = await self._execute_sequential_pipeline(mediapipe_results, timestamp)
        else:  # EMERGENCY_MINIMAL
            results = await self._execute_emergency_pipeline(mediapipe_results, timestamp)
        
        execution_time = time.time() - execution_start
        
        # 6. 실행 결과 평가 및 성능 메트릭 업데이트
        execution_report = self._evaluate_execution_performance(results, execution_time)
        self._update_processor_performance_metrics(execution_report)
        
        # 7. 융합 엔진으로 최종 결과 계산
        final_results = await self._perform_intelligent_fusion(results, execution_report)
        
        # 8. 성능 이력 업데이트 및 예측 모델 학습
        self._update_performance_history(execution_report, predicted_load)
        self.resource_predictor.update_model(execution_report)
        
        return final_results

    async def _execute_full_parallel_pipeline(
        self, mediapipe_results: Dict[str, Any], timestamp: float
    ) -> Dict[str, Any]:
        """모든 프로세서를 병렬로 실행 (최고 성능 모드)"""
        
        # 독립적인 프로세서들을 비동기 태스크로 생성
        tasks = {}
        for name, processor in self.processors.items():
            if name == 'object':
                continue  # object는 hand 결과가 필요하므로 나중에 처리
            
            task = asyncio.create_task(
                self._execute_processor_with_monitoring(
                    name, processor, mediapipe_results.get(name), timestamp, mediapipe_results
                )
            )
            tasks[name] = task
        
        # Hand 프로세서 먼저 실행 (Object가 의존성을 가지므로)
        hand_result = await tasks['hand']
        
        # Object 프로세서 실행 (Hand 결과를 전달)
        object_data_enriched = self._enrich_object_data(
            mediapipe_results.get('object'), hand_result[1] if hand_result[0] else {}
        )
        
        object_task = asyncio.create_task(
            self._execute_processor_with_monitoring(
                'object', self.processors['object'], object_data_enriched, timestamp, mediapipe_results
            )
        )
        
        # 나머지 결과들 수집
        face_result = await tasks['face']
        pose_result = await tasks['pose']
        object_result = await object_task
        
        return {
            'face': face_result,
            'pose': pose_result,
            'hand': hand_result,
            'object': object_result
        }

    async def _execute_selective_pipeline(
        self, mediapipe_results: Dict[str, Any], timestamp: float
    ) -> Dict[str, Any]:
        """선별적 병렬 실행 (성능 저하 시)"""
        
        # 가장 중요하고 안정적인 프로세서들만 병렬 실행
        critical_processors = self._identify_critical_processors()
        
        # 중요 프로세서들 병렬 실행
        critical_tasks = {}
        for name in critical_processors:
            if name in self.processors:
                processor = self.processors[name]
                task = asyncio.create_task(
                    self._execute_processor_with_monitoring(
                        name, processor, mediapipe_results.get(name), timestamp, mediapipe_results
                    )
                )
                critical_tasks[name] = task
        
        critical_results = {}
        for name, task in critical_tasks.items():
            critical_results[name] = await task
        
        # 나머지 프로세서들 순차 실행
        remaining_results = {}
        for name, processor in self.processors.items():
            if name not in critical_processors:
                result = await self._execute_processor_with_monitoring(
                    name, processor, mediapipe_results.get(name), timestamp, mediapipe_results
                )
                remaining_results[name] = result
        
        # 결과 통합
        all_results = {**critical_results, **remaining_results}
        return all_results

    async def _execute_sequential_pipeline(
        self, mediapipe_results: Dict[str, Any], timestamp: float
    ) -> Dict[str, Any]:
        """순차 실행 (안전 모드)"""
        
        results = {}
        execution_order = ['face', 'pose', 'hand', 'object']  # 의존성 순서 고려
        
        for name in execution_order:
            if name in self.processors:
                processor = self.processors[name]
                
                # Object 프로세서는 Hand 결과 필요
                if name == 'object' and 'hand' in results:
                    data = self._enrich_object_data(
                        mediapipe_results.get(name), results['hand'][1] if results['hand'][0] else {}
                    )
                else:
                    data = mediapipe_results.get(name)
                
                result = await self._execute_processor_with_monitoring(
                    name, processor, data, timestamp, mediapipe_results
                )
                results[name] = result
        
        return results

    async def _execute_emergency_pipeline(
        self, mediapipe_results: Dict[str, Any], timestamp: float
    ) -> Dict[str, Any]:
        """응급 모드 (최소한의 핵심 기능만)"""
        
        # 가장 안정적이고 중요한 프로세서만 실행 (보통 Face)
        essential_processor = self._identify_most_reliable_processor()
        
        if essential_processor in self.processors:
            processor = self.processors[essential_processor]
            result = await self._execute_processor_with_monitoring(
                essential_processor, processor, 
                mediapipe_results.get(essential_processor), timestamp, mediapipe_results
            )
            
            return {essential_processor: result}
        
        # 모든 프로세서가 실패한 경우
        return {'emergency_mode': (False, {})}

    async def _execute_processor_with_monitoring(
        self, name: str, processor: Any, data: Any, timestamp: float, mediapipe_results: dict = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """[S-Class] 프로세서 실행 및 성능 모니터링 (수정됨: 인자 불일치 및 비동기 처리 문제 해결)"""
        
        start_time = time.time()
        timeout = self.adaptive_timeouts.get(name, 0.1)
        
        try:
            process_method = processor.process_data
            args = []
            params = inspect.signature(process_method).parameters
            if 'data' in params: args.append(data)
            if 'image' in params:
                image = mediapipe_results.get('image') if mediapipe_results else None
                args.append(image)
            if 'timestamp' in params: args.append(timestamp)
            
            if inspect.iscoroutinefunction(process_method):
                result = await asyncio.wait_for(process_method(*args), timeout=timeout)
            else:
                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(loop.run_in_executor(None, lambda: process_method(*args)), timeout=timeout)
            
            execution_time = time.time() - start_time
            
            # 성공 메트릭 업데이트
            perf = self.processor_performance[name]
            perf.avg_execution_time = (perf.avg_execution_time * 0.8 + execution_time * 0.2)
            perf.last_success_time = timestamp
            perf.consecutive_failures = 0
            
            if perf.status == ProcessorStatus.FAILED:
                perf.status = ProcessorStatus.DEGRADED  # 복구 중
                logger.info(f"{name} 프로세서 복구 감지")
            
            return True, result
            
        except asyncio.TimeoutError:
            logger.warning(f"{name} 프로세서 타임아웃 (>{timeout:.3f}s)")
            self._handle_processor_failure(name, "timeout")
            return False, {}
            
        except Exception as e:
            logger.error(f"{name} 프로세서 실행 중 오류: {e}")
            self._handle_processor_failure(name, "exception")
            return False, {}

    def _handle_processor_failure(self, processor_name: str, failure_type: str):
        """프로세서 실패 처리"""
        perf = self.processor_performance[processor_name]
        perf.error_count += 1
        perf.consecutive_failures += 1
        
        # 상태 업데이트
        if perf.consecutive_failures >= 3:
            perf.status = ProcessorStatus.FAILED
        elif perf.consecutive_failures >= 1:
            perf.status = ProcessorStatus.FAILING
        
        # 적응형 타임아웃 증가 (너무 짧을 수 있음)
        current_timeout = self.adaptive_timeouts.get(processor_name, 0.1)
        self.adaptive_timeouts[processor_name] = min(0.3, current_timeout * 1.2)

    def _assess_system_health(self) -> Dict[str, float]:
        """시스템 전체 건강도 평가"""
        health_scores = {}
        
        for name, perf in self.processor_performance.items():
            if perf.status == ProcessorStatus.HEALTHY:
                health_scores[name] = 1.0
            elif perf.status == ProcessorStatus.DEGRADED:
                health_scores[name] = 0.7
            elif perf.status == ProcessorStatus.FAILING:
                health_scores[name] = 0.3
            else:  # FAILED
                health_scores[name] = 0.0
        
        health_scores['overall'] = sum(health_scores.values()) / len(health_scores)
        return health_scores

    def _determine_optimal_pipeline_mode(self, health_scores: Dict[str, float]) -> PipelineMode:
        """최적 파이프라인 모드 결정"""
        overall_health = health_scores.get('overall', 0.5)
        
        if overall_health >= 0.8:
            return PipelineMode.FULL_PARALLEL
        elif overall_health >= 0.6:
            return PipelineMode.SELECTIVE_PARALLEL
        elif overall_health >= 0.3:
            return PipelineMode.SEQUENTIAL_SAFE
        else:
            return PipelineMode.EMERGENCY_MINIMAL

    def _identify_critical_processors(self) -> List[str]:
        """중요한 프로세서들 식별 (선별적 실행용)"""
        # 성능과 중요도를 종합하여 중요 프로세서 선별
        processor_priorities = {
            'face': 0.9,    # 가장 중요 (졸음, 감정, 시선)
            'object': 0.8,  # 주의산만 감지
            'hand': 0.7,    # 핸들 그립
            'pose': 0.6     # 자세 분석
        }
        
        critical_processors = []
        for name, priority in processor_priorities.items():
            perf = self.processor_performance[name]
            if perf.status in [ProcessorStatus.HEALTHY, ProcessorStatus.DEGRADED] and priority >= 0.7:
                critical_processors.append(name)
        
        return critical_processors

    def _identify_most_reliable_processor(self) -> str:
        """가장 안정적인 프로세서 식별 (응급 모드용)"""
        best_processor = 'face'  # 기본값
        best_score = -1.0
        
        for name, perf in self.processor_performance.items():
            # 안정성 점수 = 성공률 * (1 - 평균실행시간/0.2) * 상태점수
            status_score = 1.0 if perf.status == ProcessorStatus.HEALTHY else 0.0
            time_score = max(0.0, 1.0 - perf.avg_execution_time / 0.2)
            stability_score = perf.success_rate * time_score * status_score
            
            if stability_score > best_score:
                best_score = stability_score
                best_processor = name
        
        return best_processor

    async def _perform_intelligent_fusion(
        self, processor_results: Dict[str, Any], execution_report: PipelineExecution
    ) -> Dict[str, Any]:
        """[S-Class] 지능형 융합 (실행 품질 반영)"""
        
        # 성공적으로 실행된 프로세서들의 결과만 추출
        successful_results = {}
        for name in execution_report.successful_processors:
            if name in processor_results and processor_results[name][0]:  # (성공여부, 결과)
                successful_results[name] = processor_results[name][1]
            else:
                successful_results[name] = {'available': False}
        
        # 실패한 프로세서들에 대한 기본값 설정
        for name in ['face', 'pose', 'hand', 'object']:
            if name not in successful_results:
                successful_results[name] = {'available': False}
        
        # 융합 엔진에 전달
        if 'face' in successful_results and 'pose' in successful_results and 'hand' in successful_results:
            fatigue_risk = self.fusion_engine.fuse_drowsiness_signals(
                successful_results['face'], 
                successful_results['pose'], 
                successful_results['hand']
            )
        else:
            fatigue_risk = 0.0
        
        if all(name in successful_results for name in ['face', 'pose', 'hand', 'object']):
            distraction_risk = self.fusion_engine.fuse_distraction_signals(
                successful_results['face'], 
                successful_results['pose'], 
                successful_results['hand'], 
                successful_results['object'],
                successful_results['face'].get('emotion', {})
            )
        else:
            distraction_risk = 0.0
        
        # 최종 결과 구성
        return {
            'timestamp': time.time(),
            'processed_data': successful_results,
            'fused_risks': {
                'fatigue_score': fatigue_risk,
                'distraction_score': distraction_risk,
            },
            'execution_quality': {
                'confidence_score': execution_report.confidence_score,
                'degraded_performance': execution_report.degraded_performance,
                'pipeline_mode': self.current_pipeline_mode.value
            }
        }

    # --- 성능 관리 및 예측 메서드들 ---
    
    def _adjust_adaptive_timeouts(self, predicted_load: Dict[str, float]):
        """예측된 부하에 따른 타임아웃 조정"""
        for name, load in predicted_load.items():
            if name in self.adaptive_timeouts:
                base_timeout = 0.1
                adjusted_timeout = base_timeout * (1.0 + load * 0.5)
                self.adaptive_timeouts[name] = min(0.3, adjusted_timeout)

    def _evaluate_execution_performance(
        self, results: Dict[str, Any], execution_time: float
    ) -> PipelineExecution:
        """실행 성능 평가"""
        successful = []
        failed = []
        
        for name, result in results.items():
            if isinstance(result, tuple) and result[0]:  # (성공여부, 데이터)
                successful.append(name)
            else:
                failed.append(name)
        
        # 신뢰도 점수 계산
        success_ratio = len(successful) / len(results) if results else 0.0
        time_penalty = max(0.0, 1.0 - (execution_time - 0.2) / 0.3)  # 0.2초 기준
        confidence = success_ratio * 0.7 + time_penalty * 0.3
        
        return PipelineExecution(
            total_time=execution_time,
            successful_processors=successful,
            failed_processors=failed,
            degraded_performance=len(failed) > 0,
            confidence_score=confidence
        )

    def _update_processor_performance_metrics(self, execution_report: PipelineExecution):
        """프로세서 성능 메트릭 업데이트"""
        for name in self.processors.keys():
            perf = self.processor_performance[name]
            
            if name in execution_report.successful_processors:
                perf.success_rate = perf.success_rate * 0.9 + 0.1  # 성공률 상승
            else:
                perf.success_rate = perf.success_rate * 0.9  # 성공률 하락

    def _update_performance_history(self, execution_report: PipelineExecution, predicted_load: Dict):
        """성능 이력 업데이트"""
        self.performance_history.append({
            'timestamp': time.time(),
            'execution_time': execution_report.total_time,
            'confidence': execution_report.confidence_score,
            'predicted_load': predicted_load,
            'pipeline_mode': self.current_pipeline_mode.value
        })
        
        # 최근 1000개 항목만 유지
        if len(self.performance_history) > 1000:
            self.performance_history.pop(0)

    def _enrich_object_data(self, object_data: Optional[Any], hand_results: Dict[str, Any]) -> Optional[Any]:
        """객체 데이터에 손 위치 정보를 보강"""
        if object_data and hand_results:
            hand_positions = hand_results.get('hand_positions', [])
            setattr(object_data, 'hand_positions', hand_positions)
        return object_data


class ResourcePredictor:
    """리소스 요구사항 예측기"""
    
    def __init__(self):
        self.load_history = []
    
    def predict_next_frame_load(
        self, mediapipe_results: Dict[str, Any], 
        processor_performance: Dict[str, ProcessorPerformance]
    ) -> Dict[str, float]:
        """다음 프레임의 처리 부하 예측"""
        
        predicted_load = {}
        
        for name in ['face', 'pose', 'hand', 'object']:
            # 데이터 크기와 복잡도 기반 예측
            data = mediapipe_results.get(name)
            base_load = 0.5  # 기본 부하
            
            if data:
                # 데이터 복잡도 추정 (실제로는 더 정교한 분석 필요)
                if hasattr(data, 'landmarks') and data.landmarks:
                    base_load += len(data.landmarks) * 0.01
                if hasattr(data, 'detections') and data.detections:
                    base_load += len(data.detections) * 0.02
            
            # 프로세서 상태 반영
            perf = processor_performance.get(name)
            if perf and perf.status == ProcessorStatus.DEGRADED:
                base_load *= 1.3
            elif perf and perf.status == ProcessorStatus.FAILING:
                base_load *= 1.6
            
            predicted_load[name] = min(1.0, base_load)
        
        return predicted_load
    
    def update_model(self, execution_report: PipelineExecution):
        """예측 모델 업데이트 (실제 결과 학습)"""
        self.load_history.append({
            'timestamp': time.time(),
            'actual_time': execution_report.total_time,
            'confidence': execution_report.confidence_score
        })
        
        # 최근 500개 기록만 유지
        if len(self.load_history) > 500:
            self.load_history.pop(0)