"""
Logging Setup v2
통합 시스템과 호환되는 로깅 설정
기존 코드와 개선된 코드를 통합하여 호환성과 성능을 모두 확보
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

def setup_logging_system(
    log_level: str = "INFO",
    log_dir: str = "logs",
    max_log_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """
    로깅 시스템 설정
    
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
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler for general logs
        general_log_file = log_path / "dms_system.log"
        general_handler = logging.handlers.RotatingFileHandler(
            general_log_file,
            maxBytes=max_log_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        general_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        general_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(general_handler)
        
        # File handler for errors
        error_log_file = log_path / "dms_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_log_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        
        # File handler for performance logs
        perf_log_file = log_path / "dms_performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=max_log_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(detailed_formatter)
        
        # Create performance logger
        perf_logger = logging.getLogger("performance")
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False
        
        # Log startup message
        startup_logger = logging.getLogger("system")
        startup_logger.info("=" * 60)
        startup_logger.info("DMS System Logging Initialized")
        startup_logger.info(f"Log Level: {log_level}")
        startup_logger.info(f"Log Directory: {log_path.absolute()}")
        startup_logger.info(f"Max Log Size: {max_log_size / (1024*1024):.1f} MB")
        startup_logger.info(f"Backup Count: {backup_count}")
        startup_logger.info("=" * 60)
        
        print(f"✅ Logging system initialized - Level: {log_level}, Directory: {log_path.absolute()}")
        
    except Exception as e:
        print(f"❌ Failed to setup logging system: {e}")
        # Fallback to basic logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format='%(levelname)s - %(message)s'
        )

def get_logger(name: str) -> logging.Logger:
    """
    로거 가져오기
    
    Args:
        name: 로거 이름
        
    Returns:
        설정된 로거
    """
    return logging.getLogger(name)

def log_performance(metric: str, value: float, unit: str = ""):
    """
    성능 메트릭 로깅
    
    Args:
        metric: 메트릭 이름
        value: 메트릭 값
        unit: 단위
    """
    perf_logger = logging.getLogger("performance")
    perf_logger.info(f"PERF: {metric} = {value:.2f} {unit}")

def log_system_event(event: str, details: Optional[dict] = None):
    """
    시스템 이벤트 로깅
    
    Args:
        event: 이벤트 이름
        details: 이벤트 세부사항
    """
    system_logger = logging.getLogger("system")
    if details:
        system_logger.info(f"EVENT: {event} - {details}")
    else:
        system_logger.info(f"EVENT: {event}")

def log_error(error: Exception, context: str = ""):
    """
    오류 로깅
    
    Args:
        error: 오류 객체
        context: 오류 컨텍스트
    """
    error_logger = logging.getLogger("errors")
    if context:
        error_logger.error(f"ERROR in {context}: {error}", exc_info=True)
    else:
        error_logger.error(f"ERROR: {error}", exc_info=True)

def cleanup_old_logs(log_dir: str = "logs", days_to_keep: int = 30):
    """
    오래된 로그 파일 정리
    
    Args:
        log_dir: 로그 디렉토리
        days_to_keep: 보관할 일수
    """
    try:
        import time
        from datetime import datetime, timedelta
        
        log_path = Path(log_dir)
        if not log_path.exists():
            return
        
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        for log_file in log_path.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                print(f"Removed old log file: {log_file}")
                
    except Exception as e:
        print(f"Error cleaning up old logs: {e}")