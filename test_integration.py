#!/usr/bin/env python3
"""
🧪 S-Class DMS v19.0 - 통합 테스트 스크립트
모든 시스템 구성 요소의 정상 작동을 검증합니다.
"""

import asyncio
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import traceback

# 테스트 대상 모듈들
try:
    from s_class_dms_v19_main import SClassDMSv19
    from config.settings import get_config
    from config.environments import get_environment_config, get_system_config_for_environment
    from config.profile_manager import get_profile_manager, UserProfile
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"❌ 모듈 가져오기 실패: {e}")
    IMPORTS_SUCCESS = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResult:
    """테스트 결과"""
    
    def __init__(self, name: str):
        self.name = name
        self.success = False
        self.message = ""
        self.duration = 0.0
        self.details = {}
    
    def __str__(self):
        status = "✅ PASS" if self.success else "❌ FAIL"
        return f"{status} {self.name} ({self.duration:.2f}s) - {self.message}"


class SClassDMSIntegrationTest:
    """S-Class DMS v19 통합 테스트"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
    
    def add_result(self, result: TestResult):
        """테스트 결과 추가"""
        self.results.append(result)
        self.total_tests += 1
        
        if result.success:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
    
    async def run_all_tests(self) -> bool:
        """모든 테스트 실행"""
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║                🧪 S-Class DMS v19.0 통합 테스트 시작                  ║
╚══════════════════════════════════════════════════════════════════════╝
        """)
        
        # 전체 테스트 시작 시간
        total_start_time = time.time()
        
        # 1. 기본 검증 테스트
        await self.test_basic_imports()
        await self.test_file_structure()
        
        # 2. 설정 시스템 테스트
        await self.test_configuration_system()
        await self.test_environment_configs()
        
        # 3. 프로필 관리 시스템 테스트
        await self.test_profile_management()
        
        # 4. 핵심 시스템 테스트
        await self.test_sclass_system_initialization()
        await self.test_innovation_systems()
        
        # 5. 통합 실행 시스템 테스트
        await self.test_launcher_systems()
        
        # 6. 성능 및 안정성 테스트
        await self.test_system_performance()
        
        # 전체 테스트 시간
        total_duration = time.time() - total_start_time
        
        # 결과 출력
        self.print_test_summary(total_duration)
        
        return self.failed_tests == 0
    
    async def test_basic_imports(self):
        """기본 모듈 가져오기 테스트"""
        result = TestResult("기본 모듈 가져오기")
        start_time = time.time()
        
        try:
            if not IMPORTS_SUCCESS:
                raise ImportError("필수 모듈 가져오기 실패")
            
            # 추가 모듈 가져오기 테스트
            from systems.ai_driving_coach import AIDrivingCoach
            from systems.v2d_healthcare import V2DHealthcareSystem
            from systems.ar_hud_system import ARHUDSystem
            from systems.emotional_care_system import EmotionalCareSystem
            from systems.digital_twin_platform import DigitalTwinPlatform
            
            result.success = True
            result.message = "모든 핵심 모듈 가져오기 성공"
            
        except Exception as e:
            result.success = False
            result.message = f"모듈 가져오기 실패: {e}"
        
        result.duration = time.time() - start_time
        self.add_result(result)
    
    async def test_file_structure(self):
        """파일 구조 검증 테스트"""
        result = TestResult("파일 구조 검증")
        start_time = time.time()
        
        try:
            required_files = [
                "run_sclass_dms.py",
                "gui_launcher.py", 
                "app.py",
                "main.py",
                "s_class_dms_v19_main.py",
                "README.md",
                "requirements.txt"
            ]
            
            required_dirs = [
                "config",
                "systems", 
                "core",
                "models",
                "profiles",
                "legacy_backup"
            ]
            
            missing_files = []
            missing_dirs = []
            
            # 파일 확인
            for file_name in required_files:
                if not Path(file_name).exists():
                    missing_files.append(file_name)
            
            # 디렉토리 확인  
            for dir_name in required_dirs:
                if not Path(dir_name).exists():
                    missing_dirs.append(dir_name)
            
            if missing_files or missing_dirs:
                raise FileNotFoundError(
                    f"누락된 파일: {missing_files}, 누락된 디렉토리: {missing_dirs}"
                )
            
            result.success = True
            result.message = "모든 필수 파일 및 디렉토리 존재 확인"
            result.details = {
                "checked_files": len(required_files),
                "checked_dirs": len(required_dirs)
            }
            
        except Exception as e:
            result.success = False
            result.message = f"파일 구조 검증 실패: {e}"
        
        result.duration = time.time() - start_time
        self.add_result(result)
    
    async def test_configuration_system(self):
        """설정 시스템 테스트"""
        result = TestResult("설정 시스템")
        start_time = time.time()
        
        try:
            # 기본 설정 로드
            config = get_config()
            
            # 설정 검증
            if not config.validate():
                raise ValueError("설정 검증 실패")
            
            # 주요 설정값 확인
            assert config.performance.target_fps > 0
            assert config.performance.max_processing_time_ms > 0
            assert config.feature_flags is not None
            
            result.success = True
            result.message = "설정 시스템 정상 작동"
            result.details = {
                "target_fps": config.performance.target_fps,
                "edition": config.feature_flags.system_edition
            }
            
        except Exception as e:
            result.success = False
            result.message = f"설정 시스템 오류: {e}"
        
        result.duration = time.time() - start_time
        self.add_result(result)
    
    async def test_environment_configs(self):
        """환경별 설정 테스트"""
        result = TestResult("환경별 설정")
        start_time = time.time()
        
        try:
            environments = ["development", "testing", "production", "demo"]
            
            for env_name in environments:
                env_config = get_environment_config(env_name)
                system_config = get_system_config_for_environment(env_name)
                
                # 환경 설정 검증
                assert env_config.name == env_name
                assert system_config.validate()
            
            result.success = True
            result.message = f"{len(environments)}개 환경 설정 모두 정상"
            result.details = {"environments": environments}
            
        except Exception as e:
            result.success = False
            result.message = f"환경 설정 오류: {e}"
        
        result.duration = time.time() - start_time
        self.add_result(result)
    
    async def test_profile_management(self):
        """프로필 관리 시스템 테스트"""
        result = TestResult("프로필 관리 시스템")
        start_time = time.time()
        
        try:
            pm = get_profile_manager()
            
            # 테스트 프로필 생성
            test_user_id = "integration_test_user"
            profile = pm.create_profile(test_user_id, display_name="통합테스트사용자")
            
            # 프로필 검증
            assert profile.user_id == test_user_id
            assert pm.profile_exists(test_user_id)
            
            # 프로필 업데이트 테스트
            success = pm.update_biometric_baseline(test_user_id, avg_heart_rate=80.0)
            assert success
            
            # 프로필 요약 테스트
            summary = pm.get_profile_summary(test_user_id)
            assert "user_id" in summary
            
            # 정리
            pm.delete_profile(test_user_id)
            
            result.success = True
            result.message = "프로필 CRUD 작업 모두 성공"
            
        except Exception as e:
            result.success = False
            result.message = f"프로필 관리 오류: {e}"
        
        result.duration = time.time() - start_time
        self.add_result(result)
    
    async def test_sclass_system_initialization(self):
        """S-Class 시스템 초기화 테스트"""
        result = TestResult("S-Class 시스템 초기화")
        start_time = time.time()
        
        try:
            # S-Class DMS v19 시스템 생성
            dms_system = SClassDMSv19(
                user_id="test_init_user",
                edition="RESEARCH"
            )
            
            # 기본 속성 확인
            assert dms_system.user_id == "test_init_user"
            assert dms_system.edition == "RESEARCH"
            assert dms_system.innovation_systems is not None
            
            # 혁신 시스템들 확인
            expected_systems = ["ai_coach", "healthcare", "ar_hud", "emotional_care", "digital_twin"]
            for system_name in expected_systems:
                assert system_name in dms_system.innovation_systems
            
            result.success = True
            result.message = f"시스템 초기화 성공 ({len(expected_systems)}개 혁신 시스템)"
            result.details = {"innovation_systems": list(dms_system.innovation_systems.keys())}
            
        except Exception as e:
            result.success = False
            result.message = f"시스템 초기화 실패: {e}"
        
        result.duration = time.time() - start_time
        self.add_result(result)
    
    async def test_innovation_systems(self):
        """5대 혁신 시스템 개별 테스트"""
        result = TestResult("5대 혁신 시스템")
        start_time = time.time()
        
        try:
            from systems.ai_driving_coach import AIDrivingCoach
            from systems.v2d_healthcare import V2DHealthcareSystem
            from systems.ar_hud_system import ARHUDSystem
            from systems.emotional_care_system import EmotionalCareSystem
            from systems.digital_twin_platform import DigitalTwinPlatform
            
            systems_tested = []
            
            # 1. AI 드라이빙 코치
            try:
                ai_coach = AIDrivingCoach("test_user")
                systems_tested.append("AI 드라이빙 코치")
            except Exception as e:
                logger.warning(f"AI 코치 테스트 실패: {e}")
            
            # 2. V2D 헬스케어
            try:
                healthcare = V2DHealthcareSystem("test_user")
                systems_tested.append("V2D 헬스케어")
            except Exception as e:
                logger.warning(f"헬스케어 테스트 실패: {e}")
            
            # 3. AR HUD
            try:
                ar_hud = ARHUDSystem()
                systems_tested.append("AR HUD")
            except Exception as e:
                logger.warning(f"AR HUD 테스트 실패: {e}")
            
            # 4. 감성 케어
            try:
                emotional_care = EmotionalCareSystem("test_user")
                systems_tested.append("감성 케어")
            except Exception as e:
                logger.warning(f"감성 케어 테스트 실패: {e}")
            
            # 5. 디지털 트윈
            try:
                digital_twin = DigitalTwinPlatform()
                systems_tested.append("디지털 트윈")
            except Exception as e:
                logger.warning(f"디지털 트윈 테스트 실패: {e}")
            
            if len(systems_tested) >= 3:  # 최소 3개 시스템이 작동하면 성공
                result.success = True
                result.message = f"{len(systems_tested)}/5 혁신 시스템 초기화 성공"
            else:
                result.success = False
                result.message = f"혁신 시스템 초기화 부족: {len(systems_tested)}/5"
            
            result.details = {"working_systems": systems_tested}
            
        except Exception as e:
            result.success = False
            result.message = f"혁신 시스템 테스트 오류: {e}"
        
        result.duration = time.time() - start_time
        self.add_result(result)
    
    async def test_launcher_systems(self):
        """실행 시스템 테스트"""
        result = TestResult("실행 시스템")
        start_time = time.time()
        
        try:
            launchers_tested = []
            
            # 1. CLI 런처 모듈 테스트
            try:
                import run_sclass_dms
                launchers_tested.append("CLI 런처")
            except Exception as e:
                logger.warning(f"CLI 런처 테스트 실패: {e}")
            
            # 2. GUI 런처 모듈 테스트  
            try:
                import gui_launcher
                launchers_tested.append("GUI 런처")
            except Exception as e:
                logger.warning(f"GUI 런처 테스트 실패: {e}")
            
            # 3. 웹 대시보드 모듈 테스트
            try:
                import app
                launchers_tested.append("웹 대시보드")
            except Exception as e:
                logger.warning(f"웹 대시보드 테스트 실패: {e}")
            
            # 4. 메인 런처 모듈 테스트
            try:
                import main
                launchers_tested.append("메인 런처")
            except Exception as e:
                logger.warning(f"메인 런처 테스트 실패: {e}")
            
            if len(launchers_tested) >= 3:
                result.success = True
                result.message = f"{len(launchers_tested)}/4 런처 시스템 정상"
            else:
                result.success = False
                result.message = f"런처 시스템 부족: {len(launchers_tested)}/4"
            
            result.details = {"working_launchers": launchers_tested}
            
        except Exception as e:
            result.success = False
            result.message = f"런처 시스템 테스트 오류: {e}"
        
        result.duration = time.time() - start_time
        self.add_result(result)
    
    async def test_system_performance(self):
        """시스템 성능 테스트"""
        result = TestResult("시스템 성능")
        start_time = time.time()
        
        try:
            # 메모리 사용량 체크
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # CPU 사용률 체크
            cpu_percent = process.cpu_percent()
            
            # 성능 기준 (느슨한 기준)
            memory_limit_mb = 1000  # 1GB
            cpu_limit_percent = 50   # 50%
            
            performance_ok = memory_mb < memory_limit_mb and cpu_percent < cpu_limit_percent
            
            if performance_ok:
                result.success = True
                result.message = "성능 기준 충족"
            else:
                result.success = False
                result.message = "성능 기준 초과"
            
            result.details = {
                "memory_mb": round(memory_mb, 2),
                "cpu_percent": round(cpu_percent, 2),
                "memory_limit_mb": memory_limit_mb,
                "cpu_limit_percent": cpu_limit_percent
            }
            
        except Exception as e:
            result.success = False
            result.message = f"성능 테스트 오류: {e}"
        
        result.duration = time.time() - start_time
        self.add_result(result)
    
    def print_test_summary(self, total_duration: float):
        """테스트 결과 요약 출력"""
        print("\n" + "=" * 80)
        print("📊 S-Class DMS v19.0 통합 테스트 결과")
        print("=" * 80)
        
        # 개별 테스트 결과
        for result in self.results:
            print(f"  {result}")
        
        print("\n" + "-" * 80)
        
        # 전체 요약
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print(f"📈 전체 요약:")
        print(f"  • 총 테스트: {self.total_tests}개")
        print(f"  • 성공: {self.passed_tests}개")
        print(f"  • 실패: {self.failed_tests}개")
        print(f"  • 성공률: {success_rate:.1f}%")
        print(f"  • 총 소요 시간: {total_duration:.2f}초")
        
        # 결과 판정
        if self.failed_tests == 0:
            print(f"\n🎉 모든 테스트 통과! S-Class DMS v19.0 시스템이 정상적으로 작동합니다.")
            print("✅ Phase 2 통합 실행 시스템 구축이 성공적으로 완료되었습니다!")
        else:
            print(f"\n⚠️ {self.failed_tests}개 테스트 실패. 시스템 점검이 필요합니다.")
            print("❌ 실패한 테스트들을 수정한 후 다시 실행해주세요.")
        
        print("=" * 80)
    
    def save_test_report(self, filepath: Path):
        """테스트 보고서 저장"""
        report = {
            "test_timestamp": time.time(),
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "success_rate": (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0,
            "results": []
        }
        
        for result in self.results:
            report["results"].append({
                "name": result.name,
                "success": result.success,
                "message": result.message,
                "duration": result.duration,
                "details": result.details
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 테스트 보고서 저장: {filepath}")


async def main():
    """메인 테스트 실행"""
    try:
        # 통합 테스트 실행
        tester = SClassDMSIntegrationTest()
        success = await tester.run_all_tests()
        
        # 테스트 보고서 저장
        report_path = Path("test_results") / f"integration_test_{int(time.time())}.json"
        report_path.parent.mkdir(exist_ok=True)
        tester.save_test_report(report_path)
        
        # 종료 코드 반환
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 치명적 오류: {e}")
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())