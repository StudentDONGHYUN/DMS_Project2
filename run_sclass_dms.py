#!/usr/bin/env python3
"""
🚀 S-Class DMS v19.0 통합 실행 런처
사용법: 
    python run_sclass_dms.py                    # 기본 실행
    python run_sclass_dms.py --gui              # GUI 모드  
    python run_sclass_dms.py --edition=RESEARCH # 에디션 선택
    python run_sclass_dms.py --user=myuser      # 사용자 ID 지정
    python run_sclass_dms.py --config=custom    # 커스텀 설정
"""

import argparse
import asyncio
import sys
import logging
from pathlib import Path
from typing import Optional

# S-Class DMS v19 모듈
from s_class_dms_v19_main import SClassDMSv19
from config.settings import get_config, FeatureFlagConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SClassDMSLauncher:
    """S-Class DMS v19 통합 런처"""
    
    def __init__(self):
        self.parser = self._create_argument_parser()
        
    def _create_argument_parser(self):
        """명령행 인수 파서 생성"""
        parser = argparse.ArgumentParser(
            description="🚀 S-Class DMS v19.0 - 차세대 지능형 운전자 모니터링 시스템",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
🎯 에디션 설명:
  COMMUNITY   - 기본 전문가 시스템 (무료)
  PRO         - AI 코치 + 헬스케어 (유료)
  ENTERPRISE  - AR HUD + 감성 케어 (프리미엄)  
  RESEARCH    - 모든 기능 + 디지털 트윈 (연구용)

📖 사용 예시:
  python run_sclass_dms.py
  python run_sclass_dms.py --gui --user=driver1
  python run_sclass_dms.py --edition=RESEARCH --verbose
  python run_sclass_dms.py --config=production.json
            """
        )
        
        # 기본 옵션들
        parser.add_argument(
            '--edition', '-e',
            choices=['COMMUNITY', 'PRO', 'ENTERPRISE', 'RESEARCH'],
            default='RESEARCH',
            help='시스템 에디션 선택 (기본: RESEARCH)'
        )
        
        parser.add_argument(
            '--user', '-u',
            type=str,
            default='default',
            help='사용자 ID 설정 (기본: default)'
        )
        
        parser.add_argument(
            '--config', '-c',
            type=str,
            help='커스텀 설정 파일 경로'
        )
        
        parser.add_argument(
            '--gui', '-g',
            action='store_true',
            help='GUI 모드로 실행'
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='상세 로그 출력'
        )
        
        parser.add_argument(
            '--demo',
            action='store_true',
            help='데모 모드 실행 (테스트 데이터 사용)'
        )
        
        parser.add_argument(
            '--version',
            action='version',
            version='S-Class DMS v19.0 "The Next Chapter"'
        )
        
        # 시스템 옵션들
        system_group = parser.add_argument_group('시스템 옵션')
        system_group.add_argument(
            '--no-ai-coach',
            action='store_true',
            help='AI 드라이빙 코치 비활성화'
        )
        
        system_group.add_argument(
            '--no-healthcare',
            action='store_true',
            help='헬스케어 시스템 비활성화'
        )
        
        system_group.add_argument(
            '--no-ar-hud',
            action='store_true',
            help='AR HUD 시스템 비활성화'
        )
        
        system_group.add_argument(
            '--no-emotional-care',
            action='store_true',
            help='감성 케어 시스템 비활성화'
        )
        
        system_group.add_argument(
            '--no-digital-twin',
            action='store_true',
            help='디지털 트윈 플랫폼 비활성화'
        )
        
        return parser
    
    def display_welcome_banner(self, args):
        """환영 배너 출력"""
        banner = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                  🚀 S-Class DMS v19.0 "The Next Chapter"             ║
║              차세대 지능형 운전자 모니터링 시스템                        ║
╠══════════════════════════════════════════════════════════════════════╣
║ 👤 사용자: {args.user:<20} 📦 에디션: {args.edition:<15} ║
║ 🧠 5대 혁신 시스템 통합 실행                                          ║
╚══════════════════════════════════════════════════════════════════════╝

🎯 활성화된 혁신 시스템:
"""
        print(banner)
        
        # 에디션별 기능 표시
        features = self._get_features_by_edition(args.edition)
        for feature in features:
            status = "🔴 비활성화" if self._is_feature_disabled(feature, args) else "✅ 활성화"
            print(f"   {feature}: {status}")
        
        print("\n🚀 시스템 시작 중...")
        print("=" * 70)
    
    def _get_features_by_edition(self, edition: str) -> list:
        """에디션별 기능 목록 반환"""
        all_features = [
            "🎓 AI 드라이빙 코치",
            "🏥 V2D 헬스케어 플랫폼", 
            "🥽 상황인지형 AR HUD",
            "🎭 멀티모달 감성 케어",
            "🤖 디지털 트윈 플랫폼"
        ]
        
        if edition == "COMMUNITY":
            return []  # 기본 전문가 시스템만
        elif edition == "PRO":
            return all_features[:2]  # AI 코치 + 헬스케어
        elif edition == "ENTERPRISE":
            return all_features[:4]  # AR HUD + 감성 케어 추가
        else:  # RESEARCH
            return all_features  # 모든 기능
    
    def _is_feature_disabled(self, feature: str, args) -> bool:
        """기능이 비활성화되었는지 확인"""
        feature_flags = {
            "🎓 AI 드라이빙 코치": args.no_ai_coach,
            "🏥 V2D 헬스케어 플랫폼": args.no_healthcare,
            "🥽 상황인지형 AR HUD": args.no_ar_hud,
            "🎭 멀티모달 감성 케어": args.no_emotional_care,
            "🤖 디지털 트윈 플랫폼": args.no_digital_twin
        }
        return feature_flags.get(feature, False)
    
    def setup_logging(self, verbose: bool):
        """로깅 레벨 설정"""
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("상세 로그 모드 활성화")
        else:
            logging.getLogger().setLevel(logging.INFO)
    
    async def run_cli_mode(self, args):
        """CLI 모드 실행"""
        logger.info("CLI 모드로 S-Class DMS v19 시작")
        
        try:
            # S-Class DMS v19 시스템 초기화
            dms_system = SClassDMSv19(
                user_id=args.user,
                edition=args.edition
            )
            
            # 시스템 시작
            if await dms_system.start_system():
                logger.info("✅ 모든 시스템이 성공적으로 시작되었습니다")
                
                # 메인 루프 실행
                await dms_system.run_main_loop()
                
            else:
                logger.error("❌ 시스템 시작에 실패했습니다")
                return False
                
        except KeyboardInterrupt:
            logger.info("사용자에 의한 시스템 중단")
        except Exception as e:
            logger.error(f"시스템 실행 중 오류: {e}")
            return False
        
        logger.info("S-Class DMS v19 시스템이 정상적으로 종료되었습니다")
        return True
    
    def run_gui_mode(self, args):
        """GUI 모드 실행"""
        logger.info("GUI 모드로 S-Class DMS v19 시작")
        
        try:
            # 기존 main.py의 GUI를 S-Class v19 전용으로 실행
            from gui_launcher import SClassDMSGUI
            gui = SClassDMSGUI(
                user_id=args.user,
                edition=args.edition
            )
            gui.run()
            
        except ImportError:
            logger.warning("GUI 모듈을 찾을 수 없습니다. CLI 모드로 전환합니다.")
            return asyncio.run(self.run_cli_mode(args))
        except Exception as e:
            logger.error(f"GUI 모드 실행 중 오류: {e}")
            return False
    
    def run_demo_mode(self, args):
        """데모 모드 실행"""
        logger.info("🎬 데모 모드 시작 - 테스트 데이터로 실행")
        
        print("""
🎬 S-Class DMS v19 데모 모드
================================

이 모드에서는 실제 카메라 없이 테스트 데이터를 사용하여
모든 혁신 시스템의 기능을 시연합니다.

📊 데모 시나리오:
  1. 🎓 AI 코치가 운전 행동을 분석하고 피드백 제공
  2. 🏥 헬스케어 시스템이 생체 데이터 모니터링
  3. 🥽 AR HUD가 상황인지형 정보 표시
  4. 🎭 감성 케어가 운전자 감정 상태 관리
  5. 🤖 디지털 트윈이 시뮬레이션 실행

데모를 시작하려면 Enter를 누르세요...
        """)
        
        input()  # 사용자 입력 대기
        
        # 데모 모드로 실행 (테스트 데이터 사용)
        args.user = "demo_user"
        return asyncio.run(self.run_cli_mode(args))
    
    def run(self):
        """메인 실행 함수"""
        args = self.parser.parse_args()
        
        # 로깅 설정
        self.setup_logging(args.verbose)
        
        # 환영 배너 출력
        self.display_welcome_banner(args)
        
        # 설정 파일 로드
        if args.config:
            logger.info(f"커스텀 설정 파일 로드: {args.config}")
        
        # 실행 모드 결정
        if args.demo:
            return self.run_demo_mode(args)
        elif args.gui:
            return self.run_gui_mode(args)
        else:
            return asyncio.run(self.run_cli_mode(args))


def main():
    """메인 엔트리 포인트"""
    launcher = SClassDMSLauncher()
    
    try:
        success = launcher.run()
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        print("\n👋 사용자에 의한 프로그램 종료")
        exit_code = 0
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        exit_code = 1
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()