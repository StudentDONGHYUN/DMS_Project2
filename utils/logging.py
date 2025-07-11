import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import time

def setup_logging_system():
    """
    애플리케이션 전역의 로깅 시스템을 설정합니다.
    파일과 콘솔에 로그를 출력하도록 루트 로거를 설정합니다.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_filename = log_dir / f"dms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

class TerminalLogManager:
    """터미널 로그 관리 - 주기적 초기화 (이 클래스는 로거를 직접 사용하지 않으므로 그대로 둡니다)"""
    def __init__(self):
        self.log_count = 0
        self.max_log_count = 500
        self.last_clear_time = time.time()
        self.min_clear_interval = 300

    def increment_log_count(self):
        self.log_count += 1
        current_time = time.time()
        if (
            self.log_count >= self.max_log_count
            and current_time - self.last_clear_time > self.min_clear_interval
        ):
            self.clear_terminal()
            self.log_count = 0
            self.last_clear_time = current_time

    def clear_terminal(self):
        try:
            os.system("cls" if os.name == "nt" else "clear")
            print("=== 터미널 로그 정리됨 (메모리 관리) ===")
        except:
            pass