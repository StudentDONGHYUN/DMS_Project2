"""
Logging Setup - Enhanced System
통합 시스템과 호환되는 로깅 설정
기존 코드와 개선된 코드를 통합하여 호환성과 성능을 모두 확보
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional
import time
from datetime import datetime

def setup_logging_system(
    log_level: str = "INFO",
    log_dir: str = "logs",
    max_log_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """
    애플리케이션 전역의 로깅 시스템을 설정합니다.
    파일과 콘솔에 로그를 출력하도록 루트 로거를 설정합니다.
    
    Args:
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: 로그 디렉토리
        max_log_size: 최대 로그 파일 크기 (바이트)
        backup_count: 백업 파일 개수
    """
    try:
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # Create rotating file handler
        log_filename = log_path / f"dms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_filename,
            maxBytes=max_log_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(simple_formatter)
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Add handlers to root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Log system startup
        logging.info(f"Logging system initialized - Level: {log_level}")
        logging.info(f"Log file: {log_filename}")
        
    except Exception as e:
        print(f"Failed to setup logging system: {e}")
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
            ],
        )

def get_logger(name: str) -> logging.Logger:
    """
    지정된 이름으로 로거를 가져옵니다.
    
    Args:
        name: 로거 이름
        
    Returns:
        Logger 인스턴스
    """
    return logging.getLogger(name)

def log_performance(metric: str, value: float, unit: str = ""):
    """
    성능 메트릭을 로그에 기록합니다.
    
    Args:
        metric: 메트릭 이름
        value: 메트릭 값
        unit: 단위
    """
    logger = logging.getLogger("performance")
    logger.info(f"METRIC: {metric} = {value:.4f} {unit}")

def log_system_event(event: str, details: Optional[dict] = None):
    """
    시스템 이벤트를 로그에 기록합니다.
    
    Args:
        event: 이벤트 이름
        details: 추가 세부 정보
    """
    logger = logging.getLogger("system")
    if details:
        logger.info(f"EVENT: {event} - {details}")
    else:
        logger.info(f"EVENT: {event}")

def log_error(error: Exception, context: str = ""):
    """
    에러를 상세 정보와 함께 로그에 기록합니다.
    
    Args:
        error: 에러 객체
        context: 에러 발생 컨텍스트
    """
    logger = logging.getLogger("error")
    error_msg = f"ERROR in {context}: {type(error).__name__}: {str(error)}"
    logger.error(error_msg, exc_info=True)

def cleanup_old_logs(log_dir: str = "logs", days_to_keep: int = 30):
    """
    오래된 로그 파일들을 정리합니다.
    
    Args:
        log_dir: 로그 디렉토리
        days_to_keep: 보관할 일수
    """
    try:
        log_path = Path(log_dir)
        if not log_path.exists():
            return
        
        current_time = time.time()
        cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)
        
        for log_file in log_path.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                logging.info(f"Deleted old log file: {log_file}")
                
    except Exception as e:
        logging.error(f"Failed to cleanup old logs: {e}")

class TerminalLogManager:
    """터미널 로그 관리 - 주기적 초기화 (기존 호환성 유지)"""
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
            # Use subprocess for safer command execution
            import subprocess
            if os.name == "nt":
                # Windows
                subprocess.run(["cmd", "/c", "cls"], check=True)
            else:
                # Unix/Linux/macOS
                subprocess.run(["clear"], check=True)
        except Exception as e:
            # Fallback to print method
            print("\n" * 50)