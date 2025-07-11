#!/usr/bin/env python3
"""
🚀 S-Class DMS v19.0 - 통합 런처
사용자가 원하는 실행 방식을 선택할 수 있는 메인 진입점
"""

import sys
import subprocess
import argparse
from pathlib import Path

# 로고 및 환영 메시지
WELCOME_BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║                  🚀 S-Class DMS v19.0 "The Next Chapter"             ║
║              차세대 지능형 운전자 모니터링 시스템                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  🎯 5대 혁신 시스템이 통합된 완전한 지능형 안전 플랫폼                    ║
║  ✅ 기존 시스템에서 완전히 개선된 단일 통합 시스템                        ║
║  🚀 상용화 준비 완료 • 세계 최초 통합 플랫폼                           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def print_execution_options():
    """실행 옵션 출력"""
    print("""
🎮 실행 방식을 선택하세요:

┌─────────────────────────────────────────────────────────────────────┐
│ 1. 🚀 CLI 모드 (권장)                                                │
│    python run_sclass_dms.py                                        │
│    → 명령줄에서 빠르고 간단하게 실행                                  │
│                                                                     │
│ 2. 📱 GUI 모드                                                      │
│    python gui_launcher.py                                          │
│    → 사용자 친화적인 그래픽 인터페이스                                 │
│                                                                     │
│ 3. 🌐 웹 대시보드                                                    │
│    python app.py                                                   │
│    → 웹 브라우저에서 실시간 모니터링                                   │
│                                                                     │
│ 4. 🎬 데모 모드                                                      │
│    python run_sclass_dms.py --demo                                 │
│    → 카메라 없이 모든 기능 체험                                       │
└─────────────────────────────────────────────────────────────────────┘

💡 빠른 시작: 숫자를 입력하거나 Enter를 누르세요 (기본: CLI 모드)
""")


def get_user_choice():
    """사용자 선택 입력"""
    try:
        choice = input("선택 (1-4, 기본값: 1): ").strip()
        if not choice:
            return "1"
        return choice
    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다.")
        sys.exit(0)


def run_command(command, description):
    """명령 실행"""
    print(f"\n🚀 {description} 시작 중...")
    print(f"명령어: {' '.join(command)}")
    print("─" * 70)
    
    try:
        # 명령 실행
        process = subprocess.run(command, check=False)
        
        if process.returncode == 0:
            print(f"\n✅ {description}이(가) 정상적으로 종료되었습니다.")
        else:
            print(f"\n⚠️ {description}이(가) 종료되었습니다 (코드: {process.returncode})")
            
    except FileNotFoundError:
        print(f"\n❌ 오류: {command[0]} 파일을 찾을 수 없습니다.")
        print("현재 디렉토리에 S-Class DMS v19 파일들이 있는지 확인해주세요.")
    except KeyboardInterrupt:
        print(f"\n👋 사용자에 의해 {description}이(가) 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류가 발생했습니다: {e}")


def check_files():
    """필수 파일 존재 확인"""
    required_files = [
        "run_sclass_dms.py",
        "gui_launcher.py", 
        "app.py",
        "s_class_dms_v19_main.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 다음 필수 파일들이 없습니다:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n현재 디렉토리가 S-Class DMS v19 프로젝트 루트인지 확인해주세요.")
        return False
    
    return True


def show_advanced_options():
    """고급 옵션 표시"""
    print("""
⚡ 고급 실행 옵션:

📊 에디션별 실행:
  python run_sclass_dms.py --edition=COMMUNITY   # 무료 기본 기능
  python run_sclass_dms.py --edition=PRO         # AI 코치 + 헬스케어
  python run_sclass_dms.py --edition=ENTERPRISE  # AR HUD + 감성 케어
  python run_sclass_dms.py --edition=RESEARCH    # 모든 기능

🔧 특정 시스템 제어:
  python run_sclass_dms.py --no-digital-twin     # 디지털 트윈 비활성화
  python run_sclass_dms.py --no-ar-hud           # AR HUD 비활성화

📝 로그 및 설정:
  python run_sclass_dms.py --verbose             # 상세 로그
  python run_sclass_dms.py --user=myuser         # 사용자 ID 설정

💡 도움말:
  python run_sclass_dms.py --help                # 전체 옵션 보기
""")


def main():
    """메인 실행 함수"""
    # 인수 파서 설정
    parser = argparse.ArgumentParser(
        description="S-Class DMS v19.0 통합 런처",
        add_help=False
    )
    parser.add_argument(
        '--advanced', '-a',
        action='store_true',
        help='고급 옵션 표시'
    )
    parser.add_argument(
        '--help', '-h',
        action='store_true',
        help='도움말 표시'
    )
    
    # 인수가 있는 경우 처리
    if len(sys.argv) > 1:
        args = parser.parse_args()
        
        if args.help:
            print(WELCOME_BANNER)
            print_execution_options()
            show_advanced_options()
            return
            
        if args.advanced:
            print(WELCOME_BANNER)
            show_advanced_options()
            return
    
    # 환영 메시지 출력
    print(WELCOME_BANNER)
    
    # 필수 파일 확인
    if not check_files():
        sys.exit(1)
    
    # 실행 옵션 출력
    print_execution_options()
    
    # 사용자 선택 받기
    choice = get_user_choice()
    
    # 선택에 따라 실행
    if choice == "1" or choice.lower() == "cli":
        run_command(
            [sys.executable, "run_sclass_dms.py"],
            "CLI 모드"
        )
        
    elif choice == "2" or choice.lower() == "gui":
        run_command(
            [sys.executable, "gui_launcher.py"],
            "GUI 모드"
        )
        
    elif choice == "3" or choice.lower() == "web":
        print("\n🌐 웹 대시보드 시작 중...")
        print("브라우저에서 http://localhost:5000 을 열어주세요.")
        run_command(
            [sys.executable, "app.py"],
            "웹 대시보드"
        )
        
    elif choice == "4" or choice.lower() == "demo":
        run_command(
            [sys.executable, "run_sclass_dms.py", "--demo"],
            "데모 모드"
        )
        
    else:
        print(f"\n❌ 잘못된 선택입니다: {choice}")
        print("1-4 사이의 숫자를 입력해주세요.")
        sys.exit(1)


def show_help():
    """도움말 표시"""
    print("""
🚀 S-Class DMS v19.0 통합 런처

사용법:
  python main.py              # 대화형 메뉴
  python main.py --advanced   # 고급 옵션 보기
  python main.py --help       # 이 도움말 보기

직접 실행:
  python run_sclass_dms.py    # CLI 모드 직접 실행
  python gui_launcher.py      # GUI 모드 직접 실행
  python app.py               # 웹 대시보드 직접 실행

특별한 점:
  ✅ 단일 통합 시스템 - 더 이상 혼재되지 않음
  ✅ 5대 혁신 시스템 완전 구현
  ✅ 상용화 준비 완료
  ✅ 세계 최초 통합 플랫폼

문의:
  📧 README.md 파일을 참조하세요
  🌐 웹 대시보드: http://localhost:5000
""")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 S-Class DMS v19 런처를 종료합니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        sys.exit(1)
