# utils/opencv_safe.py
"""
OpenCV 안전 처리 유틸리티
UMat 변환 및 프레임 처리 시 발생하는 오류를 방지하는 안전한 레이어
"""

import cv2
import numpy as np
import logging
from typing import Union, Optional, Any

logger = logging.getLogger(__name__)


class OpenCVSafeHandler:
    """OpenCV 안전 처리 핸들러"""
    
    @staticmethod
    def safe_umat_convert(frame: Union[np.ndarray, cv2.UMat]) -> Union[np.ndarray, cv2.UMat]:
        """
        안전한 UMat 변환
        
        Args:
            frame: 입력 프레임 (numpy array 또는 UMat)
            
        Returns:
            처리 가능한 프레임 (변환 실패 시 원본 반환)
        """
        if frame is None:
            logger.warning("None 프레임 입력, None 반환")
            return None
            
        try:
            # 이미 UMat인 경우 그대로 반환
            if isinstance(frame, cv2.UMat):
                return frame
                
            # numpy array인 경우 안전한 UMat 변환 시도
            if isinstance(frame, np.ndarray):
                # ✅ FIXED: 올바른 UMat 생성자 사용
                if frame.ndim == 3 and frame.shape[2] == 3:  # BGR 이미지
                    return cv2.UMat(frame)
                elif frame.ndim == 2:  # 그레이스케일 이미지
                    return cv2.UMat(frame)
                else:
                    logger.warning(f"지원되지 않는 프레임 형태: {frame.shape}")
                    return frame
            else:
                logger.warning(f"지원되지 않는 프레임 타입: {type(frame)}")
                return frame
                
        except Exception as e:
            logger.debug(f"UMat 변환 실패, numpy array 사용: {e}")
            # 변환 실패 시 원본 numpy array 반환
            return frame
    
    @staticmethod
    def safe_frame_annotation(frame: Union[np.ndarray, cv2.UMat], 
                             text: str, 
                             position: tuple = (10, 30),
                             font_scale: float = 1.0,
                             color: tuple = (0, 255, 0),
                             thickness: int = 2) -> Union[np.ndarray, cv2.UMat]:
        """
        안전한 프레임 텍스트 주석
        
        Args:
            frame: 입력 프레임
            text: 표시할 텍스트
            position: 텍스트 위치 (x, y)
            font_scale: 폰트 크기
            color: 텍스트 색상 (B, G, R)
            thickness: 텍스트 두께
            
        Returns:
            주석이 추가된 프레임 (실패 시 원본 반환)
        """
        if frame is None:
            return None
            
        try:
            # 프레임 타입에 관계없이 안전하게 텍스트 추가
            cv2.putText(
                frame,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness
            )
            return frame
            
        except Exception as e:
            logger.warning(f"프레임 주석 실패: {e}")
            return frame
    
    @staticmethod
    def safe_frame_flags_writeable(frame: Union[np.ndarray, cv2.UMat], 
                                  writeable: bool = False) -> Union[np.ndarray, cv2.UMat]:
        """
        안전한 프레임 flags.writeable 설정
        
        Args:
            frame: 입력 프레임
            writeable: writeable 플래그 값
            
        Returns:
            플래그가 설정된 프레임 (실패 시 원본 반환)
        """
        if frame is None:
            return None
            
        try:
            # MediaPipe 성능 최적화를 위한 writeable 플래그 설정
            if hasattr(frame, 'flags') and hasattr(frame.flags, 'writeable'):
                frame.flags.writeable = writeable
            else:
                logger.debug(f"프레임 타입 {type(frame)}에서 flags.writeable 속성 없음")
            return frame
            
        except Exception as e:
            logger.debug(f"프레임 flags 설정 실패: {e}")
            return frame
    
    @staticmethod
    def create_fallback_frame(width: int = 640, 
                             height: int = 480, 
                             channels: int = 3,
                             color: tuple = (64, 64, 64)) -> np.ndarray:
        """
        폴백 프레임 생성 (오류 상황에서 사용)
        
        Args:
            width: 프레임 너비
            height: 프레임 높이
            channels: 채널 수 (3=BGR, 1=그레이스케일)
            color: 배경 색상 (B, G, R)
            
        Returns:
            생성된 폴백 프레임
        """
        try:
            if channels == 3:
                frame = np.full((height, width, 3), color, dtype=np.uint8)
            else:
                frame = np.full((height, width), color[0], dtype=np.uint8)
            return frame
        except Exception as e:
            logger.error(f"폴백 프레임 생성 실패: {e}")
            # 최소한의 검은 프레임 반환
            return np.zeros((480, 640, 3), dtype=np.uint8)


# 편의 함수들
def safe_create_basic_info_overlay(frame, frame_count, perf_stats=None):
    """
    안전한 기본 정보 오버레이 생성
    """
    try:
        # ✅ FIXED: 안전한 UMat 변환 사용
        annotated_frame = OpenCVSafeHandler.safe_umat_convert(frame)
        
        # 프레임 번호 표시
        annotated_frame = OpenCVSafeHandler.safe_frame_annotation(
            annotated_frame,
            f"Frame: {frame_count}",
            position=(10, 30),
            color=(0, 255, 0)
        )
        
        # 성능 정보 표시
        if perf_stats is not None:
            fps = perf_stats.get("fps", 0.0)
            annotated_frame = OpenCVSafeHandler.safe_frame_annotation(
                annotated_frame,
                f"FPS: {fps:.1f}",
                position=(10, 60),
                font_scale=0.8,
                color=(255, 255, 0)
            )
        
        return annotated_frame
        
    except Exception as e:
        logger.error(f"안전한 오버레이 생성 실패: {e}")
        # 최종 폴백: 원본 프레임 반환 또는 폴백 프레임 생성
        if frame is not None:
            return frame
        else:
            return OpenCVSafeHandler.create_fallback_frame()


def safe_frame_preprocessing_for_mediapipe(frame):
    """
    MediaPipe 처리 전 안전한 프레임 전처리
    """
    if frame is None:
        return None
        
    try:
        # MediaPipe 성능 최적화를 위한 writeable=False 설정
        processed_frame = OpenCVSafeHandler.safe_frame_flags_writeable(frame, writeable=False)
        return processed_frame
        
    except Exception as e:
        logger.warning(f"MediaPipe 전처리 실패: {e}")
        return frame
