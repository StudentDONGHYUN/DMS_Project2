"""
S-Class DMS v19 - 환경별 설정 관리
개발, 테스트, 운영 환경별로 최적화된 설정을 제공합니다.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import os
from .settings import SystemConfig, PerformanceConfig, FeatureFlagConfig


@dataclass
class EnvironmentConfig:
    """환경별 설정 기본 클래스"""
    name: str
    debug_mode: bool
    log_level: str
    enable_profiling: bool
    save_session_data: bool
    enable_web_dashboard: bool
    max_users: int = 1


class DevelopmentConfig(EnvironmentConfig):
    """개발 환경 설정"""
    
    def __init__(self):
        super().__init__(
            name="development",
            debug_mode=True,
            log_level="DEBUG",
            enable_profiling=True,
            save_session_data=True,
            enable_web_dashboard=True,
            max_users=5
        )
    
    def get_system_config(self) -> SystemConfig:
        """개발 환경용 시스템 설정"""
        config = SystemConfig()
        
        # 개발 환경 최적화
        config.debug_mode = True
        config.log_level = "DEBUG"
        
        # 성능 설정 - 개발용으로 완화
        config.performance.target_fps = 15  # 개발 시 부하 감소
        config.performance.max_processing_time_ms = 500.0  # 여유롭게 설정
        
        # 모든 기능 활성화 (개발/테스트용)
        config.feature_flags = FeatureFlagConfig(system_edition="RESEARCH")
        config.feature_flags.enable_detailed_logging = True
        config.feature_flags.enable_experimental_algorithms = True
        
        return config


class TestingConfig(EnvironmentConfig):
    """테스트 환경 설정"""
    
    def __init__(self):
        super().__init__(
            name="testing",
            debug_mode=True,
            log_level="INFO",
            enable_profiling=True,
            save_session_data=False,  # 테스트 데이터는 저장하지 않음
            enable_web_dashboard=False,
            max_users=1
        )
    
    def get_system_config(self) -> SystemConfig:
        """테스트 환경용 시스템 설정"""
        config = SystemConfig()
        
        # 테스트 환경 최적화
        config.debug_mode = True
        config.log_level = "INFO"
        
        # 성능 설정 - 빠른 테스트를 위해 최적화
        config.performance.target_fps = 10  # 테스트 속도 향상
        config.performance.max_processing_time_ms = 100.0
        
        # 기본 기능만 활성화 (테스트 안정성)
        config.feature_flags = FeatureFlagConfig(system_edition="PRO")
        config.feature_flags.enable_detailed_logging = False
        
        # 테스트용 디렉토리
        config.logs_dir = Path("test_logs")
        config.profiles_dir = Path("test_profiles")
        
        return config


class ProductionConfig(EnvironmentConfig):
    """운영 환경 설정"""
    
    def __init__(self):
        super().__init__(
            name="production",
            debug_mode=False,
            log_level="WARNING",
            enable_profiling=False,
            save_session_data=True,
            enable_web_dashboard=True,
            max_users=100
        )
    
    def get_system_config(self) -> SystemConfig:
        """운영 환경용 시스템 설정"""
        config = SystemConfig()
        
        # 운영 환경 최적화
        config.debug_mode = False
        config.log_level = "WARNING"
        
        # 성능 설정 - 최적 성능
        config.performance.target_fps = 30
        config.performance.max_processing_time_ms = 80.0  # 엄격한 성능 요구사항
        
        # 안정적인 기능만 활성화
        config.feature_flags = FeatureFlagConfig(system_edition="ENTERPRISE")
        config.feature_flags.enable_detailed_logging = False
        config.feature_flags.enable_experimental_algorithms = False
        
        return config


class DemoConfig(EnvironmentConfig):
    """데모 환경 설정"""
    
    def __init__(self):
        super().__init__(
            name="demo",
            debug_mode=False,
            log_level="INFO",
            enable_profiling=False,
            save_session_data=False,
            enable_web_dashboard=True,
            max_users=10
        )
    
    def get_system_config(self) -> SystemConfig:
        """데모 환경용 시스템 설정"""
        config = SystemConfig()
        
        # 데모 환경 최적화
        config.debug_mode = False
        config.log_level = "INFO"
        
        # 성능 설정 - 시연용
        config.performance.target_fps = 20
        config.performance.max_processing_time_ms = 150.0
        
        # 모든 혁신 기능 활성화 (데모 효과)
        config.feature_flags = FeatureFlagConfig(system_edition="RESEARCH")
        config.feature_flags.enable_detailed_logging = False
        
        return config


# 환경별 설정 매핑
ENVIRONMENT_CONFIGS = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "demo": DemoConfig
}


def get_environment() -> str:
    """현재 환경을 감지합니다."""
    # 환경 변수에서 확인
    env = os.getenv("SCLASS_ENV", "").lower()
    if env in ENVIRONMENT_CONFIGS:
        return env
    
    # 파일 존재 여부로 추정
    if Path(".dev").exists():
        return "development"
    elif Path(".test").exists():
        return "testing"
    elif Path(".demo").exists():
        return "demo"
    else:
        return "production"  # 기본값


def get_environment_config(env_name: str = None) -> EnvironmentConfig:
    """지정된 환경의 설정을 반환합니다."""
    if env_name is None:
        env_name = get_environment()
    
    config_class = ENVIRONMENT_CONFIGS.get(env_name)
    if config_class is None:
        raise ValueError(f"알 수 없는 환경: {env_name}")
    
    return config_class()


def get_system_config_for_environment(env_name: str = None) -> SystemConfig:
    """지정된 환경에 최적화된 시스템 설정을 반환합니다."""
    env_config = get_environment_config(env_name)
    return env_config.get_system_config()


def create_environment_marker(env_name: str):
    """환경 마커 파일을 생성합니다."""
    marker_files = {
        "development": ".dev",
        "testing": ".test", 
        "production": ".prod",
        "demo": ".demo"
    }
    
    # 기존 마커 파일들 제거
    for marker in marker_files.values():
        marker_path = Path(marker)
        if marker_path.exists():
            marker_path.unlink()
    
    # 새 마커 파일 생성
    if env_name in marker_files:
        marker_path = Path(marker_files[env_name])
        marker_path.write_text(f"S-Class DMS v19 - {env_name} environment")
        print(f"✅ {env_name} 환경 마커 생성: {marker_path}")


def list_available_environments() -> Dict[str, str]:
    """사용 가능한 환경 목록을 반환합니다."""
    return {
        "development": "개발 환경 - 모든 기능 활성화, 상세 로깅",
        "testing": "테스트 환경 - 빠른 실행, 기본 기능만",
        "production": "운영 환경 - 최적 성능, 안정적 기능만",
        "demo": "데모 환경 - 시연용, 모든 혁신 기능 활성화"
    }


def show_current_environment():
    """현재 환경 정보를 출력합니다."""
    current_env = get_environment()
    env_config = get_environment_config(current_env)
    system_config = env_config.get_system_config()
    
    print(f"""
🌍 현재 환경: {current_env.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 환경 설정:
  • 디버그 모드: {'활성화' if env_config.debug_mode else '비활성화'}
  • 로그 레벨: {env_config.log_level}
  • 프로파일링: {'활성화' if env_config.enable_profiling else '비활성화'}
  • 세션 저장: {'활성화' if env_config.save_session_data else '비활성화'}
  • 웹 대시보드: {'활성화' if env_config.enable_web_dashboard else '비활성화'}
  • 최대 사용자: {env_config.max_users}명

⚡ 성능 설정:
  • 목표 FPS: {system_config.performance.target_fps}
  • 최대 처리 시간: {system_config.performance.max_processing_time_ms}ms
  • 시스템 에디션: {system_config.feature_flags.system_edition}

🎯 활성화된 주요 기능:
  • rPPG 심박수: {'✅' if system_config.feature_flags.enable_rppg_heart_rate else '❌'}
  • 감정 AI: {'✅' if system_config.feature_flags.enable_emotion_ai else '❌'}
  • 예측 안전: {'✅' if system_config.feature_flags.enable_predictive_safety else '❌'}
  • 디지털 트윈: {'✅' if system_config.feature_flags.enable_digital_twin_simulation else '❌'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)


if __name__ == "__main__":
    # 환경 설정 테스트
    print("🧪 환경 설정 테스트")
    print("=" * 50)
    
    # 현재 환경 표시
    show_current_environment()
    
    # 모든 환경 나열
    print("\n📋 사용 가능한 환경:")
    for env_name, description in list_available_environments().items():
        print(f"  • {env_name}: {description}")