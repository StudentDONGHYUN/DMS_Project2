"""
Sensor Backup Manager v2
통합 시스템과 호환되는 센서 백업 관리자
기존 코드와 개선된 코드를 통합하여 호환성과 성능을 모두 확보
"""

import time
import logging
from typing import Dict, Any, Optional, List
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SensorBackupManager:
    """통합 센서 백업 관리자 - 기존 코드와 개선된 코드의 통합"""

    def __init__(self):
        """센서 백업 관리자 초기화"""
        self.backup_dir = Path("backup")
        self.backup_dir.mkdir(exist_ok=True)
        
        # Backup state
        self.backup_enabled = True
        self.backup_interval = 60.0  # seconds
        self.last_backup_time = 0.0
        
        # Data storage
        self.sensor_data = []
        self.max_data_points = 1000
        
        # Backup statistics
        self.backup_count = 0
        self.total_backup_size = 0
        
        logger.info("Sensor Backup Manager initialized")

    def add_sensor_data(self, data: Dict[str, Any]):
        """센서 데이터 추가"""
        try:
            if not self.backup_enabled:
                return
            
            # Add timestamp
            data_with_timestamp = {
                'timestamp': time.time(),
                'data': data
            }
            
            self.sensor_data.append(data_with_timestamp)
            
            # Keep data size manageable
            if len(self.sensor_data) > self.max_data_points:
                self.sensor_data = self.sensor_data[-self.max_data_points:]
                
        except Exception as e:
            logger.error(f"Error adding sensor data: {e}")

    def should_create_backup(self) -> bool:
        """백업 생성 여부 확인"""
        if not self.backup_enabled:
            return False
        
        current_time = time.time()
        return current_time - self.last_backup_time >= self.backup_interval

    def create_backup(self) -> Optional[str]:
        """백업 생성"""
        try:
            if not self.sensor_data:
                logger.debug("No sensor data to backup")
                return None
            
            # Create backup filename
            timestamp = int(time.time())
            backup_file = self.backup_dir / f"sensor_backup_{timestamp}.json"
            
            # Prepare backup data
            backup_data = {
                'backup_timestamp': timestamp,
                'data_count': len(self.sensor_data),
                'backup_version': '2.0',
                'sensor_data': self.sensor_data
            }
            
            # Write backup file
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            # Update statistics
            self.backup_count += 1
            self.last_backup_time = time.time()
            self.total_backup_size += backup_file.stat().st_size
            
            logger.info(f"Backup created: {backup_file} ({len(self.sensor_data)} data points)")
            
            # Clear data after successful backup
            self.sensor_data.clear()
            
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return None

    def load_backup(self, backup_file: str) -> Optional[List[Dict[str, Any]]]:
        """백업 로드"""
        try:
            backup_path = Path(backup_file)
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_file}")
                return None
            
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # Validate backup data
            if 'sensor_data' not in backup_data:
                logger.error("Invalid backup file format")
                return None
            
            sensor_data = backup_data['sensor_data']
            logger.info(f"Backup loaded: {backup_file} ({len(sensor_data)} data points)")
            
            return sensor_data
            
        except Exception as e:
            logger.error(f"Error loading backup: {e}")
            return None

    def get_backup_info(self) -> Dict[str, Any]:
        """백업 정보 반환"""
        try:
            # Get backup files
            backup_files = list(self.backup_dir.glob("sensor_backup_*.json"))
            
            total_size = sum(f.stat().st_size for f in backup_files)
            
            return {
                'backup_enabled': self.backup_enabled,
                'backup_interval': self.backup_interval,
                'backup_count': self.backup_count,
                'total_backup_size': self.total_backup_size,
                'backup_files_count': len(backup_files),
                'current_data_points': len(self.sensor_data),
                'last_backup_time': self.last_backup_time,
                'total_backup_files_size': total_size
            }
            
        except Exception as e:
            logger.error(f"Error getting backup info: {e}")
            return {}

    def list_backups(self) -> List[Dict[str, Any]]:
        """백업 목록 반환"""
        try:
            backup_files = list(self.backup_dir.glob("sensor_backup_*.json"))
            backups = []
            
            for backup_file in backup_files:
                try:
                    stat = backup_file.stat()
                    backups.append({
                        'filename': backup_file.name,
                        'size': stat.st_size,
                        'created_time': stat.st_mtime,
                        'path': str(backup_file)
                    })
                except Exception as e:
                    logger.warning(f"Error getting info for backup file {backup_file}: {e}")
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x['created_time'], reverse=True)
            
            return backups
            
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []

    def cleanup_old_backups(self, max_backups: int = 10):
        """오래된 백업 정리"""
        try:
            backups = self.list_backups()
            
            if len(backups) <= max_backups:
                return
            
            # Remove oldest backups
            backups_to_remove = backups[max_backups:]
            
            for backup in backups_to_remove:
                try:
                    backup_path = Path(backup['path'])
                    backup_path.unlink()
                    logger.info(f"Removed old backup: {backup['filename']}")
                except Exception as e:
                    logger.warning(f"Error removing backup {backup['filename']}: {e}")
            
            logger.info(f"Cleanup completed: removed {len(backups_to_remove)} old backups")
            
        except Exception as e:
            logger.error(f"Error during backup cleanup: {e}")

    def set_backup_interval(self, interval: float):
        """백업 간격 설정"""
        self.backup_interval = max(10.0, interval)  # Minimum 10 seconds
        logger.info(f"Backup interval set to {self.backup_interval} seconds")

    def enable_backup(self, enabled: bool):
        """백업 활성화/비활성화"""
        self.backup_enabled = enabled
        logger.info(f"Backup {'enabled' if enabled else 'disabled'}")

    def clear_current_data(self):
        """현재 데이터 초기화"""
        self.sensor_data.clear()
        logger.info("Current sensor data cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            'backup_count': self.backup_count,
            'total_backup_size': self.total_backup_size,
            'current_data_points': len(self.sensor_data),
            'backup_enabled': self.backup_enabled,
            'backup_interval': self.backup_interval
        }

    def close(self):
        """백업 관리자 종료"""
        try:
            # Create final backup if there's data
            if self.sensor_data:
                self.create_backup()
            
            # Cleanup old backups
            self.cleanup_old_backups()
            
            logger.info("Sensor Backup Manager closed")
            
        except Exception as e:
            logger.error(f"Error closing Sensor Backup Manager: {e}")