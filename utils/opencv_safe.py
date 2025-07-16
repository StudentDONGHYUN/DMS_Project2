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
    
    @staticmethod
    def safe_umat_to_numpy(umat_frame):
        """안전한 UMat → numpy 변환"""
        try:
            if umat_frame is None:
                return None
                
            if isinstance(umat_frame, cv2.UMat):
                return umat_frame.get()
            elif isinstance(umat_frame, np.ndarray):
                return umat_frame
            else:
                logger.warning(f"알 수 없는 프레임 타입: {type(umat_frame)}")
                return None
                
        except Exception as e:
            logger.error(f"UMat→numpy 변환 실패: {e}")
            return None
    
    @staticmethod
    def safe_numpy_to_umat(numpy_frame):
        """안전한 numpy → UMat 변환"""
        try:
            if numpy_frame is None:
                return None
                
            if isinstance(numpy_frame, np.ndarray):
                # 프레임 검증
                if (numpy_frame.ndim == 3 and 
                    numpy_frame.shape[2] == 3 and 
                    numpy_frame.dtype == np.uint8):
                    return cv2.UMat(numpy_frame)
                elif (numpy_frame.ndim == 2 and 
                      numpy_frame.dtype == np.uint8):
                    return cv2.UMat(numpy_frame)
                else:
                    logger.debug(f"UMat 변환 불가능한 형태: {numpy_frame.shape}, {numpy_frame.dtype}")
                    return numpy_frame
            elif isinstance(numpy_frame, cv2.UMat):
                return numpy_frame
            else:
                logger.warning(f"알 수 없는 프레임 타입: {type(numpy_frame)}")
                return numpy_frame
                
        except Exception as e:
            logger.error(f"numpy→UMat 변환 실패: {e}")
            return numpy_frame
    
    @staticmethod
    def validate_frame_for_display(frame):
        """화면 표시용 프레임 검증"""
        try:
            if frame is None:
                return False, "None 프레임"
                
            # UMat을 numpy로 변환
            if isinstance(frame, cv2.UMat):
                try:
                    frame = frame.get()
                except Exception as e:
                    return False, f"UMat 변환 실패: {e}"
            
            if not isinstance(frame, np.ndarray):
                return False, f"numpy 배열이 아님: {type(frame)}"
                
            if frame.size == 0:
                return False, "빈 배열"
                
            if frame.ndim not in [2, 3]:
                return False, f"잘못된 차원: {frame.ndim}"
                
            if frame.ndim == 3 and frame.shape[2] != 3:
                return False, f"잘못된 채널 수: {frame.shape[2]}"
                
            if frame.dtype != np.uint8:
                return False, f"잘못된 데이터 타입: {frame.dtype}"
                
            return True, "유효한 프레임"
            
        except Exception as e:
            return False, f"검증 중 오류: {e}"


# 편의 함수들
def safe_create_basic_info_overlay(frame, frame_count, perf_stats=None):
    """
    안전한 기본 정보 오버레이 생성 (완전 수정)
    """
    try:
        # ✅ FIXED: 입력 검증 강화
        if frame is None:
            logger.warning("None 프레임 입력 - 폴백 프레임 생성")
            frame = OpenCVSafeHandler.create_fallback_frame()
            
        # numpy 배열 검증
        if not isinstance(frame, (np.ndarray, cv2.UMat)):
            logger.warning(f"지원되지 않는 프레임 타입: {type(frame)}")
            frame = OpenCVSafeHandler.create_fallback_frame()
            
        # UMat을 numpy로 변환 (안전한 처리)
        if isinstance(frame, cv2.UMat):
            try:
                frame = frame.get()
            except Exception as umat_e:
                logger.warning(f"UMat→numpy 변환 실패: {umat_e}")
                frame = OpenCVSafeHandler.create_fallback_frame()
        
        # 프레임 차원 검증
        if frame.ndim not in [2, 3]:
            logger.warning(f"잘못된 프레임 차원: {frame.ndim}")
            frame = OpenCVSafeHandler.create_fallback_frame()
            
        if frame.ndim == 3 and frame.shape[2] not in [1, 3, 4]:
            logger.warning(f"잘못된 채널 수: {frame.shape[2]}")
            frame = OpenCVSafeHandler.create_fallback_frame()
        
        # 안전한 프레임 복사
        try:
            annotated_frame = frame.copy()
        except Exception as copy_e:
            logger.warning(f"프레임 복사 실패: {copy_e}")
            annotated_frame = frame
        
        # 프레임 번호 표시 (안전한 처리)
        try:
            annotated_frame = OpenCVSafeHandler.safe_frame_annotation(
                annotated_frame,
                f"Frame: {frame_count}",
                position=(10, 30),
                color=(0, 255, 0),
                font_scale=0.8
            )
        except Exception as frame_text_e:
            logger.debug(f"프레임 번호 표시 실패: {frame_text_e}")
        
        # 성능 정보 표시 (안전한 처리)
        if perf_stats is not None:
            try:
                fps = float(perf_stats.get("fps", 0.0))
                annotated_frame = OpenCVSafeHandler.safe_frame_annotation(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    position=(10, 60),
                    font_scale=0.7,
                    color=(255, 255, 0)
                )
                
                # 추가 성능 정보
                if "processing_time" in perf_stats:
                    proc_time = float(perf_stats["processing_time"]) * 1000  # ms 변환
                    annotated_frame = OpenCVSafeHandler.safe_frame_annotation(
                        annotated_frame,
                        f"Process: {proc_time:.1f}ms",
                        position=(10, 90),
                        font_scale=0.6,
                        color=(128, 255, 255)
                    )
                    
            except Exception as perf_e:
                logger.debug(f"성능 정보 표시 실패: {perf_e}")
        
        # 시스템 상태 표시
        try:
            annotated_frame = OpenCVSafeHandler.safe_frame_annotation(
                annotated_frame,
                "S-Class DMS v19.0 Active",
                position=(10, annotated_frame.shape[0] - 30),
                font_scale=0.6,
                color=(128, 255, 128)
            )
        except Exception as status_e:
            logger.debug(f"상태 정보 표시 실패: {status_e}")
        
        return annotated_frame
        
    except Exception as e:
        logger.error(f"안전한 오버레이 생성 완전 실패: {e}")
        
        # 최종 폴백: 기본 프레임 반환 또는 생성
        try:
            if frame is not None and isinstance(frame, np.ndarray):
                return frame
            else:
                return OpenCVSafeHandler.create_fallback_frame()
        except Exception as final_e:
            logger.error(f"최종 폴백도 실패: {final_e}")
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
