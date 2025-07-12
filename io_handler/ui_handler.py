"""
Integrated UI Handler
통합된 UI 핸들러 - 기존 UI 기능과 개선된 시스템 아키텍처 통합
"""

import asyncio
import cv2
import numpy as np
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# Core imports
from models.data_structures import UIState, UIMode, AlertType
from io_handler.ui import SClassAdvancedUIManager

logger = logging.getLogger(__name__)


class UIHandler:
    """통합 UI 핸들러 - 기존 UI 기능과 개선된 시스템 아키텍처 통합"""

    def __init__(self):
        """UI 핸들러 초기화"""
        self.is_running = False
        self.is_initialized = False
        
        # UI components
        self.ui_manager = None
        self.display_window = None
        
        # State management
        self.current_ui_state = None
        self.last_update_time = 0.0
        self.update_interval = 0.033  # ~30 FPS
        
        # Display settings
        self.window_name = "S-Class DMS - Integrated System"
        self.window_width = 1280
        self.window_height = 720
        
        # Performance tracking
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        logger.info("UI Handler initialized")

    async def start(self) -> bool:
        """UI 핸들러 시작"""
        try:
            logger.info("Starting UI Handler...")
            
            # Initialize UI manager
            self.ui_manager = SClassAdvancedUIManager()
            
            # Create display window
            self._create_display_window()
            
            self.is_initialized = True
            self.is_running = True
            
            logger.info("UI Handler started successfully")
            return True
            
        except Exception as e:
            logger.error(f"UI Handler startup failed: {e}")
            return False

    def _create_display_window(self):
        """디스플레이 윈도우 생성"""
        try:
            # Create named window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            
            # Set window size
            cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
            
            # Set window properties
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
            
            logger.info(f"Display window created: {self.window_name}")
            
        except Exception as e:
            logger.error(f"Display window creation failed: {e}")

    async def update(self, ui_state: UIState):
        """UI 상태 업데이트"""
        try:
            if not self.is_running or not self.is_initialized:
                return
            
            current_time = time.time()
            
            # Check update interval
            if current_time - self.last_update_time < self.update_interval:
                return
            
            # Update current state
            self.current_ui_state = ui_state
            self.last_update_time = current_time
            
            # Update FPS counter
            self._update_fps_counter()
            
            # Render UI
            await self._render_ui()
            
        except Exception as e:
            logger.error(f"UI update error: {e}")

    def _update_fps_counter(self):
        """FPS 카운터 업데이트"""
        current_time = time.time()
        self.fps_counter += 1
        
        if current_time - self.last_fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.last_fps_time)
            logger.debug(f"UI FPS: {fps:.1f}")
            
            self.fps_counter = 0
            self.last_fps_time = current_time

    async def _render_ui(self):
        """UI 렌더링"""
        try:
            if not self.current_ui_state:
                return
            
            # Create a blank frame for UI rendering
            frame = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
            
            # Render UI state using the existing UI manager
            rendered_frame = self.ui_manager.render_ui_state(frame, self.current_ui_state)
            
            # Display the frame
            cv2.imshow(self.window_name, rendered_frame)
            
            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # Key pressed
                await self._handle_key_event(key)
            
        except Exception as e:
            logger.error(f"UI rendering error: {e}")

    async def _handle_key_event(self, key: int):
        """키 이벤트 처리"""
        try:
            # Handle UI mode changes
            if key == ord('m') or key == ord('M'):
                # UI mode cycling handled by UI manager
                pass
            
            # Handle auto mode
            elif key == ord('a') or key == ord('A'):
                # Auto mode handled by UI manager
                pass
            
            # Handle exit
            elif key == ord('q') or key == ord('Q') or key == 27:  # ESC
                logger.info("Exit requested via keyboard")
                self.is_running = False
            
            # Handle other keys
            else:
                logger.debug(f"Key pressed: {key}")
            
        except Exception as e:
            logger.error(f"Key event handling error: {e}")

    async def stop(self):
        """UI 핸들러 중지"""
        try:
            logger.info("Stopping UI Handler...")
            
            self.is_running = False
            
            # Close display window
            if self.display_window:
                cv2.destroyWindow(self.window_name)
            
            # Close all windows
            cv2.destroyAllWindows()
            
            logger.info("UI Handler stopped successfully")
            
        except Exception as e:
            logger.error(f"UI Handler shutdown error: {e}")

    def get_ui_state(self) -> Optional[UIState]:
        """현재 UI 상태 반환"""
        return self.current_ui_state

    def get_performance_info(self) -> Dict[str, Any]:
        """UI 성능 정보 반환"""
        return {
            'is_running': self.is_running,
            'is_initialized': self.is_initialized,
            'frame_count': self.frame_count,
            'last_update_time': self.last_update_time,
            'window_name': self.window_name,
            'window_size': f"{self.window_width}x{self.window_height}"
        }

    def set_display_settings(self, width: int = None, height: int = None, window_name: str = None):
        """디스플레이 설정 변경"""
        try:
            if width is not None:
                self.window_width = width
            
            if height is not None:
                self.window_height = height
            
            if window_name is not None:
                self.window_name = window_name
                if self.is_initialized:
                    cv2.destroyWindow(self.window_name)
                    self._create_display_window()
            
            logger.info(f"Display settings updated: {self.window_width}x{self.window_height}, {self.window_name}")
            
        except Exception as e:
            logger.error(f"Display settings update failed: {e}")

    def set_update_interval(self, interval: float):
        """업데이트 간격 설정"""
        self.update_interval = max(0.001, interval)  # Minimum 1ms
        logger.info(f"Update interval set to {self.update_interval:.3f}s")