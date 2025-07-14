"""
Sensor Backup Manager - Enhanced System
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
                self.sensor_data.pop(0)
            
            # Check if backup is needed
            if self.should_create_backup():
                self.create_backup()
                
        except Exception as e:
            logger.error(f"Error adding sensor data: {e}")

    def should_create_backup(self) -> bool:
        """백업 생성 여부 확인"""
        try:
            current_time = time.time()
            return (current_time - self.last_backup_time) >= self.backup_interval
        except Exception as e:
            logger.error(f"Error checking backup condition: {e}")
            return False

    def create_backup(self) -> Optional[str]:
        """백업 생성"""
        try:
            if not self.sensor_data:
                logger.debug("No sensor data to backup")
                return None
            
            current_time = time.time()
            backup_filename = f"sensor_backup_{int(current_time)}.json"
            backup_filepath = self.backup_dir / backup_filename
            
            backup_data = {
                'created_at': current_time,
                'data_count': len(self.sensor_data),
                'data': self.sensor_data.copy()
            }
            
            with open(backup_filepath, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            # Update statistics
            file_size = backup_filepath.stat().st_size
            self.backup_count += 1
            self.total_backup_size += file_size
            self.last_backup_time = current_time
            
            logger.info(f"Backup created: {backup_filename} ({file_size} bytes)")
            
            # Clear current data after backup
            self.sensor_data.clear()
            
            return str(backup_filepath)
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return None

    def load_backup(self, backup_file: str) -> Optional[List[Dict[str, Any]]]:
        """백업 로드"""
        try:
            backup_path = Path(backup_file)
            
            if not backup_path.exists():
                backup_path = self.backup_dir / backup_file
            
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_file}")
                return None
            
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            if 'data' in backup_data:
                logger.info(f"Backup loaded: {backup_file} ({len(backup_data['data'])} data points)")
                return backup_data['data']
            else:
                logger.error(f"Invalid backup format: {backup_file}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading backup: {e}")
            return None

    def get_backup_info(self) -> Dict[str, Any]:
        """백업 정보 반환"""
        try:
            return {
                'backup_enabled': self.backup_enabled,
                'backup_interval_seconds': self.backup_interval,
                'last_backup_time': self.last_backup_time,
                'current_data_points': len(self.sensor_data),
                'max_data_points': self.max_data_points,
                'backup_count': self.backup_count,
                'total_backup_size_bytes': self.total_backup_size,
                'backup_directory': str(self.backup_dir),
                'time_until_next_backup': max(0, self.backup_interval - (time.time() - self.last_backup_time))
            }
        except Exception as e:
            logger.error(f"Error getting backup info: {e}")
            return {}

    def list_backups(self) -> List[Dict[str, Any]]:
        """백업 파일 목록 반환"""
        try:
            backups = []
            
            for backup_file in self.backup_dir.glob("sensor_backup_*.json"):
                try:
                    stat = backup_file.stat()
                    
                    # Try to read metadata from file
                    try:
                        with open(backup_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        data_count = data.get('data_count', 0)
                        created_at = data.get('created_at', stat.st_mtime)
                    except:
                        data_count = 0
                        created_at = stat.st_mtime
                    
                    backups.append({
                        'filename': backup_file.name,
                        'filepath': str(backup_file),
                        'size_bytes': stat.st_size,
                        'created_at': created_at,
                        'data_count': data_count
                    })
                except Exception as e:
                    logger.warning(f"Error reading backup file {backup_file}: {e}")
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x['created_at'], reverse=True)
            
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
                    backup_path = Path(backup['filepath'])
                    backup_path.unlink()
                    logger.info(f"Removed old backup: {backup['filename']}")
                except Exception as e:
                    logger.warning(f"Error removing backup {backup['filename']}: {e}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")

    def set_backup_interval(self, interval: float):
        """백업 간격 설정"""
        self.backup_interval = max(10.0, interval)  # Minimum 10 seconds
        logger.info(f"Backup interval set to {self.backup_interval} seconds")

    def enable_backup(self, enabled: bool):
        """백업 활성화/비활성화"""
        self.backup_enabled = enabled
        logger.info(f"Backup {'enabled' if enabled else 'disabled'}")

    def clear_current_data(self):
        """현재 센서 데이터 지우기"""
        self.sensor_data.clear()
        logger.info("Current sensor data cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            'backup_count': self.backup_count,
            'total_backup_size_bytes': self.total_backup_size,
            'current_data_points': len(self.sensor_data),
            'backup_enabled': self.backup_enabled,
            'backup_interval_seconds': self.backup_interval
        }

    def close(self):
        """백업 관리자 종료"""
        try:
            # Create final backup if there's data
            if self.sensor_data and self.backup_enabled:
                self.create_backup()
            
            logger.info("Sensor Backup Manager closed")
            
        except Exception as e:
            logger.error(f"Error closing backup manager: {e}")

    # Legacy compatibility methods
    def backup_data(self, data):
        """Legacy compatibility method"""
        self.add_sensor_data(data)

    def get_backup_status(self):
        """Legacy compatibility method"""
        return {
            'enabled': self.backup_enabled,
            'last_backup': self.last_backup_time,
            'data_points': len(self.sensor_data)
        }
