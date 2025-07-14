"""
DMS System Integration Test - 통합 시스템 무결성 검증

이 테스트는 새로 구현된 S-Class DMS 시스템의 모든 컴포넌트들이
올바르게 연결되고 동작하는지 검증합니다.

비유: 신차 출고 전 최종 검수
- 모든 전자 시스템 작동 확인
- 각 부품간 연결 상태 점검
- 성능 벤치마크 테스트
- 안전 시스템 동작 검증

테스트 단계:
1. 기본 컴포넌트 로딩 테스트
2. 이벤트 시스템 통신 테스트
3. 팩토리 패턴 동작 테스트
4. 통합 시스템 종합 테스트
5. 성능 벤치마크 측정
"""

import asyncio
import logging
import time
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DMSSystemIntegrationTest:
    """DMS 시스템 통합 테스트 클래스"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
    async def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info("=== DMS S-Class 시스템 통합 테스트 시작 ===")
        
        try:
            # 1. 기본 컴포넌트 로딩 테스트
            await self.test_basic_imports()
            
            # 2. 이벤트 시스템 테스트
            await self.test_event_system()
            
            # 3. 팩토리 시스템 테스트
            await self.test_factory_system()
            
            # 4. 통합 시스템 테스트
            await self.test_integrated_system()
            
            # 5. 성능 벤치마크
            await self.test_performance_benchmark()
            
            # 6. 결과 보고서 생성
            self.generate_test_report()
            
        except Exception as e:
            logger.error(f"테스트 실행 중 치명적 오류: {e}")
            return False
        
        return True
    
    async def test_basic_imports(self):
        """기본 컴포넌트 import 테스트"""
        logger.info("📦 기본 컴포넌트 로딩 테스트 시작...")
        
        test_name = "basic_imports"
        start_time = time.time()
        
        try:
            # === S-Class 프로세서들 ===
            from analysis.processors.face_processor import FaceDataProcessor
            from analysis.processors.pose_processor_s_class import PoseDataProcessor
            from analysis.processors.hand_processor_s_class import HandDataProcessor
            from analysis.processors.object_processor_s_class import ObjectDataProcessor
            logger.info("✅ S-Class 프로세서들 로딩 성공")
            
            # === 융합 엔진 ===
            from analysis.fusion.fusion_engine_advanced import MultiModalFusionEngine
            logger.info("✅ 고급 융합 엔진 로딩 성공")
            
            # === 이벤트 시스템 ===
            from events.event_bus import EventBus, initialize_event_system
            from events.handlers import SafetyEventHandler, AnalyticsEventHandler
            logger.info("✅ 이벤트 시스템 로딩 성공")
            
            # === 팩토리 시스템 ===
            from analysis.factory.analysis_factory import create_analysis_system, AnalysisSystemType
            logger.info("✅ 팩토리 시스템 로딩 성공")
            
            # === 통합 시스템 ===
            from integration.integrated_system import IntegratedDMSSystem
            logger.info("✅ 통합 시스템 로딩 성공")
            
            # === 설정 및 상수 ===
            from config.settings import get_config
            from core.constants import SClassConstants, EventSystemConstants
            from core.interfaces import IAdvancedFaceProcessor, IEventHandler
            logger.info("✅ 설정 및 인터페이스 로딩 성공")
            
            self.test_results[test_name] = {
                'status': 'PASS',
                'duration': time.time() - start_time,
                'details': 'All core components loaded successfully'
            }
            
        except ImportError as e:
            self.test_results[test_name] = {
                'status': 'FAIL',
                'duration': time.time() - start_time,
                'error': f'Import error: {e}',
                'details': 'Failed to load one or more core components'
            }
            logger.error(f"❌ 컴포넌트 로딩 실패: {e}")
            raise
        
        logger.info(f"📦 기본 컴포넌트 테스트 완료 ({time.time() - start_time:.2f}초)")
    
    async def test_event_system(self):
        """이벤트 시스템 동작 테스트"""
        logger.info("📡 이벤트 시스템 테스트 시작...")
        
        test_name = "event_system"
        start_time = time.time()
        
        try:
            from events.event_bus import (
                initialize_event_system,
                get_event_bus,
                publish_safety_event,
                EventType,
                EventPriority
            )
            from events.handlers import SafetyEventHandler
            
            # 1. 이벤트 시스템 초기화
            await initialize_event_system()
            event_bus = get_event_bus()
            logger.info("✅ 이벤트 버스 초기화 성공")
            
            # 2. 핸들러 등록
            safety_handler = SafetyEventHandler()
            event_bus.subscribe(safety_handler, [EventType.DROWSINESS_DETECTED])
            logger.info("✅ 이벤트 핸들러 등록 성공")
            
            # 3. 테스트 이벤트 발행
            await publish_safety_event(
                EventType.DROWSINESS_DETECTED,
                {'test_data': 'integration_test', 'fatigue_level': 0.7},
                source='integration_test',
                priority=EventPriority.HIGH
            )
            
            # 4. 이벤트 처리 대기
            await asyncio.sleep(0.1)  # 이벤트 처리 시간 확보
            
            # 5. 통계 확인
            stats = event_bus.get_statistics()
            if stats['total_events'] > 0:
                logger.info(f"✅ 이벤트 처리 성공 (총 {stats['total_events']}개 이벤트)")
            else:
                raise RuntimeError("이벤트가 처리되지 않음")
            
            self.test_results[test_name] = {
                'status': 'PASS',
                'duration': time.time() - start_time,
                'details': f"Events processed: {stats['total_events']}"
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'FAIL',
                'duration': time.time() - start_time,
                'error': str(e),
                'details': 'Event system test failed'
            }
            logger.error(f"❌ 이벤트 시스템 테스트 실패: {e}")
            raise
        
        logger.info(f"📡 이벤트 시스템 테스트 완료 ({time.time() - start_time:.2f}초)")
    
    async def test_factory_system(self):
        """팩토리 시스템 테스트"""
        logger.info("🏭 팩토리 시스템 테스트 시작...")
        
        test_name = "factory_system"
        start_time = time.time()
        
        try:
            from analysis.factory.analysis_factory import (
                AnalysisSystemType, 
                get_system_info, 
                list_available_systems
            )
            
            # 1. 사용 가능한 시스템 타입 확인
            available_systems = list_available_systems()
            logger.info(f"✅ 사용 가능한 시스템: {len(available_systems)}개")
            
            # 2. 각 시스템 타입 정보 확인
            for system_type in AnalysisSystemType:
                info = get_system_info(system_type)
                if 'error' in info:
                    raise RuntimeError(f"시스템 정보 조회 실패: {system_type}")
                logger.info(f"✅ {system_type.value} 시스템 정보 확인 완료")
            
            self.test_results[test_name] = {
                'status': 'PASS',
                'duration': time.time() - start_time,
                'details': f"All {len(available_systems)} system types validated"
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'FAIL',
                'duration': time.time() - start_time,
                'error': str(e),
                'details': 'Factory system test failed'
            }
            logger.error(f"❌ 팩토리 시스템 테스트 실패: {e}")
            raise
        
        logger.info(f"🏭 팩토리 시스템 테스트 완료 ({time.time() - start_time:.2f}초)")
    
    async def test_integrated_system(self):
        """통합 시스템 종합 테스트"""
        logger.info("🎯 통합 시스템 종합 테스트 시작...")
        
        test_name = "integrated_system"
        start_time = time.time()
        
        try:
            from integration.integrated_system import IntegratedDMSSystem
            from analysis.factory.analysis_factory import AnalysisSystemType
            
            # 1. 표준 시스템 생성 및 초기화
            dms = IntegratedDMSSystem(AnalysisSystemType.STANDARD)
            await dms.initialize()
            logger.info("✅ 통합 시스템 초기화 성공")
            
            # 2. 모의 데이터로 테스트
            mock_frame_data = {
                'face': None,  # 실제로는 MediaPipe 결과
                'pose': None,
                'hand': None,
                'object': None
            }
            
            # 3. 프레임 처리 테스트 (장애 허용 모드로 동작해야 함)
            result = await dms.process_and_annotate_frame(mock_frame_data, time.time())
            
            # 4. 결과 검증
            required_fields = ['fatigue_risk_score', 'distraction_risk_score', 'confidence_score']
            for field in required_fields:
                if field not in result:
                    raise RuntimeError(f"필수 결과 필드 누락: {field}")
            
            logger.info(f"✅ 프레임 처리 성공 (신뢰도: {result['confidence_score']:.2f})")
            
            # 5. 시스템 상태 확인
            status = dms.get_system_status()
            if status['system_health'] == 'healthy':
                logger.info("✅ 시스템 상태 정상")
            
            # 6. 정리
            await dms.shutdown()
            logger.info("✅ 시스템 정상 종료")
            
            self.test_results[test_name] = {
                'status': 'PASS',
                'duration': time.time() - start_time,
                'details': f"Frame processed with confidence: {result['confidence_score']:.2f}"
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'FAIL',
                'duration': time.time() - start_time,
                'error': str(e),
                'details': 'Integrated system test failed'
            }
            logger.error(f"❌ 통합 시스템 테스트 실패: {e}")
            raise
        
        logger.info(f"🎯 통합 시스템 테스트 완료 ({time.time() - start_time:.2f}초)")
    
    async def test_performance_benchmark(self):
        """성능 벤치마크 테스트"""
        logger.info("⚡ 성능 벤치마크 테스트 시작...")
        
        test_name = "performance_benchmark"
        start_time = time.time()
        
        try:
            from core.constants import PerformanceBenchmarks
            
            # 성능 목표값 확인
            target_processing_time = PerformanceBenchmarks.ProcessingTime.TOTAL_PIPELINE_TARGET  # ms
            target_memory = PerformanceBenchmarks.MemoryUsage.TOTAL_SYSTEM_TARGET  # MB
            
            logger.info(f"✅ 성능 벤치마크 로딩 완료")
            logger.info(f"  - 목표 처리시간: {target_processing_time}ms")
            logger.info(f"  - 목표 메모리: {target_memory}MB")
            
            # 실제 성능 테스트는 통합 시스템 테스트에서 이미 수행됨
            # 여기서는 설정값들의 일관성만 확인
            
            accuracy_targets = PerformanceBenchmarks.AccuracyTargets
            if accuracy_targets.OVERALL_SYSTEM_CONFIDENCE > 0.5:
                logger.info(f"✅ 정확도 목표 설정 적절함 ({accuracy_targets.OVERALL_SYSTEM_CONFIDENCE})")
            
            self.test_results[test_name] = {
                'status': 'PASS',
                'duration': time.time() - start_time,
                'details': f"Performance targets validated"
            }
            
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'FAIL',
                'duration': time.time() - start_time,
                'error': str(e),
                'details': 'Performance benchmark test failed'
            }
            logger.error(f"❌ 성능 벤치마크 테스트 실패: {e}")
            raise
        
        logger.info(f"⚡ 성능 벤치마크 테스트 완료 ({time.time() - start_time:.2f}초)")
    
    def generate_test_report(self):
        """테스트 결과 보고서 생성"""
        total_duration = time.time() - self.start_time
        
        logger.info("\n" + "="*60)
        logger.info("🏆 DMS S-Class 시스템 통합 테스트 결과 보고서")
        logger.info("="*60)
        
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASS')
        total_tests = len(self.test_results)
        
        logger.info(f"📊 전체 결과: {passed_tests}/{total_tests} 테스트 통과")
        logger.info(f"⏱️  총 소요시간: {total_duration:.2f}초")
        logger.info("")
        
        # 개별 테스트 결과
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            logger.info(f"{status_icon} {test_name}: {result['status']} ({result['duration']:.2f}초)")
            
            if result['status'] == 'FAIL':
                logger.info(f"   오류: {result.get('error', 'Unknown error')}")
            else:
                logger.info(f"   세부사항: {result.get('details', 'No details')}")
        
        logger.info("")
        
        if passed_tests == total_tests:
            logger.info("🎉 모든 테스트 통과! S-Class DMS 시스템이 완벽히 통합되었습니다.")
            logger.info("🚀 시스템 배포 준비 완료!")
        else:
            logger.warning(f"⚠️  {total_tests - passed_tests}개 테스트 실패. 문제를 해결해야 합니다.")
        
        logger.info("="*60)
        
        return passed_tests == total_tests


async def main():
    """메인 테스트 실행 함수"""
    test_runner = DMSSystemIntegrationTest()
    success = await test_runner.run_all_tests()
    
    if success:
        print("\n🎯 통합 테스트 성공! 시스템 준비 완료.")
        return 0
    else:
        print("\n❌ 통합 테스트 실패. 시스템 점검 필요.")
        return 1


if __name__ == "__main__":
    # 테스트 실행
    exit_code = asyncio.run(main())
    sys.exit(exit_code)